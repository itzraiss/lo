import numpy as np
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import random

from config import config
from utils.logger import loto_logger, async_performance_logger, performance_logger

@dataclass
class MonteCarloConfig:
    """Configuração para simulação Monte Carlo"""
    num_simulations: int = config.model.monte_carlo_sims
    parallel_processing: bool = True
    batch_size: int = 10000
    confidence_level: float = 0.95
    
    # Configurações de performance
    max_workers: int = config.performance.max_workers
    chunk_size: int = 1000
    
    # Critérios de convergência
    convergence_threshold: float = 0.001
    min_simulations: int = 10000
    max_simulations: int = 1000000
    
    # Métricas a calcular
    target_hits: List[int] = None
    
    def __post_init__(self):
        if self.target_hits is None:
            self.target_hits = [16, 17, 18, 19, 20]

class LotomaniaMonteCarloSimulator:
    """Simulador Monte Carlo para estimativa de probabilidades da Lotomania"""
    
    def __init__(self, mc_config: MonteCarloConfig = None):
        self.config = mc_config or MonteCarloConfig()
        self.simulation_cache = {}
        self.last_simulation_time = 0
        
    @async_performance_logger
    async def simulate_ticket_probabilities(self, tickets: List[List[int]], 
                                          number_probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Simula probabilidades de acertos para os tickets gerados"""
        try:
            loto_logger.log_info(f"Iniciando simulação Monte Carlo com {self.config.num_simulations} simulações")
            
            # Preparar dados
            prob_array = np.array([number_probabilities[str(i)] for i in range(1, 101)])
            tickets_array = [np.array(ticket) - 1 for ticket in tickets]  # Convert to 0-based indexing
            
            # Normalizar probabilidades
            prob_array = prob_array / prob_array.sum() * 20  # 20 números sorteados
            
            # Executar simulação
            if self.config.parallel_processing and len(tickets) > 1:
                results = await self._run_parallel_simulation(tickets_array, prob_array)
            else:
                results = await self._run_sequential_simulation(tickets_array, prob_array)
            
            # Processar resultados
            processed_results = self._process_simulation_results(results, tickets)
            
            # Calcular métricas de confiança
            confidence_metrics = self._calculate_confidence_metrics(processed_results)
            
            # Combinar resultados
            final_results = {
                **processed_results,
                'confidence_metrics': confidence_metrics,
                'simulation_config': {
                    'num_simulations': self.config.num_simulations,
                    'parallel_processing': self.config.parallel_processing,
                    'target_hits': self.config.target_hits
                }
            }
            
            # Log resultados
            self._log_simulation_results(final_results)
            
            return final_results
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na simulação Monte Carlo")
            # Retornar estimativas conservadoras
            return self._generate_fallback_estimates(tickets)
    
    async def _run_parallel_simulation(self, tickets: List[np.ndarray], 
                                     probabilities: np.ndarray) -> Dict[str, Any]:
        """Executa simulação em paralelo"""
        try:
            # Dividir simulações em batches
            total_sims = self.config.num_simulations
            batch_size = self.config.batch_size
            num_batches = (total_sims + batch_size - 1) // batch_size
            
            # Preparar argumentos para processamento paralelo
            simulation_args = []
            for i in range(num_batches):
                start_sim = i * batch_size
                end_sim = min((i + 1) * batch_size, total_sims)
                batch_sims = end_sim - start_sim
                
                simulation_args.append({
                    'tickets': tickets,
                    'probabilities': probabilities,
                    'num_simulations': batch_sims,
                    'random_seed': random.randint(1, 1000000) + i
                })
            
            # Executar batches em paralelo
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self._run_simulation_batch, args) 
                    for args in simulation_args
                ]
                
                # Coletar resultados
                batch_results = []
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 min timeout per batch
                        batch_results.append(result)
                    except Exception as e:
                        loto_logger.log_error(e, f"Erro em batch de simulação")
                        continue
            
            # Agregar resultados dos batches
            aggregated_results = self._aggregate_batch_results(batch_results)
            
            return aggregated_results
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na simulação paralela")
            # Fallback para simulação sequencial
            return await self._run_sequential_simulation(tickets, probabilities)
    
    async def _run_sequential_simulation(self, tickets: List[np.ndarray], 
                                       probabilities: np.ndarray) -> Dict[str, Any]:
        """Executa simulação sequencial"""
        try:
            args = {
                'tickets': tickets,
                'probabilities': probabilities,
                'num_simulations': self.config.num_simulations,
                'random_seed': random.randint(1, 1000000)
            }
            
            # Executar simulação
            results = self._run_simulation_batch(args)
            
            return results
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na simulação sequencial")
            raise
    
    @performance_logger
    def _run_simulation_batch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executa um batch de simulações"""
        try:
            tickets = args['tickets']
            probabilities = args['probabilities']
            num_simulations = args['num_simulations']
            random_seed = args['random_seed']
            
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Inicializar contadores
            hit_counts = {ticket_idx: {hits: 0 for hits in self.config.target_hits} 
                         for ticket_idx in range(len(tickets))}
            
            total_hits_distribution = {ticket_idx: [] for ticket_idx in range(len(tickets))}
            
            # Executar simulações
            for sim in range(num_simulations):
                # Gerar números sorteados usando probabilidades
                drawn_numbers = self._sample_drawn_numbers(probabilities)
                
                # Verificar acertos para cada ticket
                for ticket_idx, ticket in enumerate(tickets):
                    hits = len(np.intersect1d(ticket, drawn_numbers))
                    total_hits_distribution[ticket_idx].append(hits)
                    
                    # Contar hits específicos
                    for target_hits in self.config.target_hits:
                        if hits >= target_hits:
                            hit_counts[ticket_idx][target_hits] += 1
            
            # Calcular estatísticas
            results = {
                'hit_counts': hit_counts,
                'hit_probabilities': {},
                'total_hits_distribution': total_hits_distribution,
                'num_simulations': num_simulations,
                'statistics': {}
            }
            
            # Calcular probabilidades
            for ticket_idx in range(len(tickets)):
                results['hit_probabilities'][ticket_idx] = {}
                for target_hits in self.config.target_hits:
                    prob = hit_counts[ticket_idx][target_hits] / num_simulations
                    results['hit_probabilities'][ticket_idx][target_hits] = prob
                
                # Estatísticas da distribuição
                hits_array = np.array(total_hits_distribution[ticket_idx])
                results['statistics'][ticket_idx] = {
                    'mean_hits': float(np.mean(hits_array)),
                    'std_hits': float(np.std(hits_array)),
                    'min_hits': int(np.min(hits_array)),
                    'max_hits': int(np.max(hits_array)),
                    'median_hits': float(np.median(hits_array))
                }
            
            return results
            
        except Exception as e:
            loto_logger.log_error(e, "Erro no batch de simulação")
            raise
    
    def _sample_drawn_numbers(self, probabilities: np.ndarray) -> np.ndarray:
        """Amostra 20 números sem reposição usando probabilidades"""
        # Método 1: Amostragem direta (mais rápido)
        try:
            # Normalizar probabilidades
            probs = probabilities / probabilities.sum()
            
            # Amostragem sem reposição
            drawn_indices = np.random.choice(
                100, 
                size=20, 
                replace=False, 
                p=probs
            )
            
            return drawn_indices
            
        except Exception:
            # Fallback: amostragem uniforme
            return np.random.choice(100, size=20, replace=False)
    
    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Agrega resultados de múltiplos batches"""
        try:
            if not batch_results:
                raise ValueError("Nenhum resultado de batch disponível")
            
            # Inicializar estruturas agregadas
            num_tickets = len(batch_results[0]['hit_counts'])
            total_simulations = sum(result['num_simulations'] for result in batch_results)
            
            aggregated_hit_counts = {
                ticket_idx: {hits: 0 for hits in self.config.target_hits}
                for ticket_idx in range(num_tickets)
            }
            
            aggregated_distributions = {ticket_idx: [] for ticket_idx in range(num_tickets)}
            
            # Agregar contadores e distribuições
            for result in batch_results:
                for ticket_idx in range(num_tickets):
                    # Somar contadores
                    for target_hits in self.config.target_hits:
                        aggregated_hit_counts[ticket_idx][target_hits] += \
                            result['hit_counts'][ticket_idx][target_hits]
                    
                    # Combinar distribuições
                    aggregated_distributions[ticket_idx].extend(
                        result['total_hits_distribution'][ticket_idx]
                    )
            
            # Calcular probabilidades agregadas
            aggregated_probabilities = {}
            aggregated_statistics = {}
            
            for ticket_idx in range(num_tickets):
                aggregated_probabilities[ticket_idx] = {}
                for target_hits in self.config.target_hits:
                    prob = aggregated_hit_counts[ticket_idx][target_hits] / total_simulations
                    aggregated_probabilities[ticket_idx][target_hits] = prob
                
                # Estatísticas agregadas
                hits_array = np.array(aggregated_distributions[ticket_idx])
                aggregated_statistics[ticket_idx] = {
                    'mean_hits': float(np.mean(hits_array)),
                    'std_hits': float(np.std(hits_array)),
                    'min_hits': int(np.min(hits_array)),
                    'max_hits': int(np.max(hits_array)),
                    'median_hits': float(np.median(hits_array))
                }
            
            return {
                'hit_counts': aggregated_hit_counts,
                'hit_probabilities': aggregated_probabilities,
                'total_hits_distribution': aggregated_distributions,
                'num_simulations': total_simulations,
                'statistics': aggregated_statistics
            }
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao agregar resultados dos batches")
            raise
    
    def _process_simulation_results(self, simulation_results: Dict[str, Any], 
                                  original_tickets: List[List[int]]) -> Dict[str, Any]:
        """Processa e formata resultados da simulação"""
        try:
            num_tickets = len(original_tickets)
            
            # Calcular probabilidades para pelo menos um ticket acertar
            combined_probabilities = {}
            for target_hits in self.config.target_hits:
                # P(pelo menos um ticket ≥ target_hits) = 1 - P(todos tickets < target_hits)
                prob_all_miss = 1.0
                for ticket_idx in range(num_tickets):
                    prob_hit = simulation_results['hit_probabilities'][ticket_idx][target_hits]
                    prob_miss = 1.0 - prob_hit
                    prob_all_miss *= prob_miss
                
                combined_probabilities[f"at_least_{target_hits}"] = 1.0 - prob_all_miss
            
            # Calcular estatísticas por ticket
            ticket_statistics = []
            for ticket_idx in range(num_tickets):
                ticket_stats = {
                    'ticket_number': ticket_idx + 1,
                    'numbers': original_tickets[ticket_idx],
                    'hit_probabilities': simulation_results['hit_probabilities'][ticket_idx],
                    'statistics': simulation_results['statistics'][ticket_idx]
                }
                ticket_statistics.append(ticket_stats)
            
            # Calcular métricas globais
            best_ticket_idx = self._find_best_ticket(simulation_results['hit_probabilities'])
            worst_ticket_idx = self._find_worst_ticket(simulation_results['hit_probabilities'])
            
            processed_results = {
                'individual_tickets': ticket_statistics,
                'combined_probabilities': combined_probabilities,
                'best_ticket': {
                    'index': best_ticket_idx,
                    'numbers': original_tickets[best_ticket_idx],
                    'probabilities': simulation_results['hit_probabilities'][best_ticket_idx]
                },
                'worst_ticket': {
                    'index': worst_ticket_idx,
                    'numbers': original_tickets[worst_ticket_idx],
                    'probabilities': simulation_results['hit_probabilities'][worst_ticket_idx]
                },
                'total_simulations': simulation_results['num_simulations'],
                'simulation_quality': self._assess_simulation_quality(simulation_results)
            }
            
            return processed_results
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao processar resultados da simulação")
            raise
    
    def _find_best_ticket(self, hit_probabilities: Dict[int, Dict[int, float]]) -> int:
        """Encontra o ticket com melhor probabilidade"""
        best_score = -1
        best_ticket = 0
        
        for ticket_idx, probs in hit_probabilities.items():
            # Score ponderado: mais peso para hits maiores
            score = sum(probs[hits] * hits for hits in self.config.target_hits)
            if score > best_score:
                best_score = score
                best_ticket = ticket_idx
        
        return best_ticket
    
    def _find_worst_ticket(self, hit_probabilities: Dict[int, Dict[int, float]]) -> int:
        """Encontra o ticket com pior probabilidade"""
        worst_score = float('inf')
        worst_ticket = 0
        
        for ticket_idx, probs in hit_probabilities.items():
            # Score ponderado: mais peso para hits maiores
            score = sum(probs[hits] * hits for hits in self.config.target_hits)
            if score < worst_score:
                worst_score = score
                worst_ticket = ticket_idx
        
        return worst_ticket
    
    def _assess_simulation_quality(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia a qualidade da simulação"""
        try:
            num_simulations = simulation_results['num_simulations']
            
            # Calcular erro padrão aproximado
            standard_errors = {}
            for ticket_idx in simulation_results['hit_probabilities']:
                standard_errors[ticket_idx] = {}
                for target_hits in self.config.target_hits:
                    p = simulation_results['hit_probabilities'][ticket_idx][target_hits]
                    # Erro padrão da proporção
                    se = np.sqrt(p * (1 - p) / num_simulations)
                    standard_errors[ticket_idx][target_hits] = se
            
            # Avaliar convergência
            max_se = max(
                se for ticket_se in standard_errors.values() 
                for se in ticket_se.values()
            )
            
            convergence_quality = "excellent" if max_se < 0.001 else \
                                "good" if max_se < 0.005 else \
                                "fair" if max_se < 0.01 else "poor"
            
            quality_metrics = {
                'convergence_quality': convergence_quality,
                'max_standard_error': max_se,
                'standard_errors': standard_errors,
                'simulation_efficiency': min(1.0, num_simulations / self.config.min_simulations)
            }
            
            return quality_metrics
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao avaliar qualidade da simulação")
            return {'convergence_quality': 'unknown', 'max_standard_error': 0}
    
    def _calculate_confidence_metrics(self, processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas de confiança para as estimativas"""
        try:
            confidence_level = self.config.confidence_level
            alpha = 1 - confidence_level
            z_score = 1.96  # Para 95% de confiança
            
            confidence_intervals = {}
            
            # Intervalos de confiança para probabilidades combinadas
            for key, prob in processed_results['combined_probabilities'].items():
                n = processed_results['total_simulations']
                se = np.sqrt(prob * (1 - prob) / n)
                margin_error = z_score * se
                
                ci_lower = max(0, prob - margin_error)
                ci_upper = min(1, prob + margin_error)
                
                confidence_intervals[key] = {
                    'estimate': prob,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'margin_error': margin_error
                }
            
            # Determinar se passa nos critérios de confiança
            passes_criteria = self._check_confidence_criteria(processed_results['combined_probabilities'])
            
            confidence_metrics = {
                'confidence_level': confidence_level,
                'confidence_intervals': confidence_intervals,
                'passes_criteria': passes_criteria,
                'recommendation': self._generate_recommendation(passes_criteria, processed_results)
            }
            
            return confidence_metrics
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao calcular métricas de confiança")
            return {}
    
    def _check_confidence_criteria(self, combined_probabilities: Dict[str, float]) -> Dict[str, bool]:
        """Verifica se as probabilidades passam nos critérios de confiança"""
        criteria_passed = {}
        
        # Critérios configuráveis
        criteria_passed['18_hits'] = combined_probabilities.get('at_least_18', 0) >= config.model.confidence_threshold_18
        criteria_passed['17_hits'] = combined_probabilities.get('at_least_17', 0) >= config.model.confidence_threshold_17
        criteria_passed['16_hits'] = combined_probabilities.get('at_least_16', 0) >= config.model.confidence_threshold_16
        
        criteria_passed['any_criterion'] = any(criteria_passed.values())
        criteria_passed['all_criteria'] = all(criteria_passed.values())
        
        return criteria_passed
    
    def _generate_recommendation(self, passes_criteria: Dict[str, bool], 
                               processed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera recomendação baseada nos resultados"""
        try:
            if passes_criteria.get('any_criterion', False):
                recommendation = {
                    'action': 'PLAY',
                    'confidence': 'HIGH' if passes_criteria.get('18_hits', False) else 'MEDIUM',
                    'reason': 'Probabilidades atingem critérios mínimos de confiança'
                }
            else:
                recommendation = {
                    'action': 'WAIT',
                    'confidence': 'LOW',
                    'reason': 'Probabilidades abaixo dos critérios mínimos'
                }
            
            # Adicionar detalhes
            best_prob_18 = processed_results['combined_probabilities'].get('at_least_18', 0)
            best_prob_17 = processed_results['combined_probabilities'].get('at_least_17', 0)
            
            recommendation['details'] = {
                'best_18_probability': best_prob_18,
                'best_17_probability': best_prob_17,
                'expected_value': self._calculate_expected_value(processed_results),
                'risk_assessment': self._assess_risk(processed_results)
            }
            
            return recommendation
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar recomendação")
            return {'action': 'UNKNOWN', 'confidence': 'LOW'}
    
    def _calculate_expected_value(self, processed_results: Dict[str, Any]) -> float:
        """Calcula valor esperado aproximado"""
        try:
            # Valores de prêmio aproximados (simplificado)
            prize_values = {20: 500000, 19: 50000, 18: 1000, 17: 50, 16: 5}
            
            expected_value = 0
            for hits, prize in prize_values.items():
                prob_key = f"at_least_{hits}"
                if prob_key in processed_results['combined_probabilities']:
                    prob = processed_results['combined_probabilities'][prob_key]
                    expected_value += prob * prize
            
            # Subtrair custo das apostas (3 jogos × R$ 2,50)
            cost = 3 * 2.50
            net_expected_value = expected_value - cost
            
            return net_expected_value
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao calcular valor esperado")
            return 0.0
    
    def _assess_risk(self, processed_results: Dict[str, Any]) -> str:
        """Avalia o risco da estratégia"""
        try:
            prob_16_plus = processed_results['combined_probabilities'].get('at_least_16', 0)
            prob_17_plus = processed_results['combined_probabilities'].get('at_least_17', 0)
            prob_18_plus = processed_results['combined_probabilities'].get('at_least_18', 0)
            
            if prob_18_plus > 0.05:
                return "LOW"
            elif prob_17_plus > 0.15:
                return "MEDIUM"
            elif prob_16_plus > 0.50:
                return "HIGH"
            else:
                return "VERY_HIGH"
                
        except Exception:
            return "UNKNOWN"
    
    def _generate_fallback_estimates(self, tickets: List[List[int]]) -> Dict[str, Any]:
        """Gera estimativas conservadoras em caso de erro"""
        try:
            # Estimativas baseadas em probabilidade teórica simples
            fallback_estimates = {
                'individual_tickets': [],
                'combined_probabilities': {
                    'at_least_16': 0.01,
                    'at_least_17': 0.005,
                    'at_least_18': 0.001,
                    'at_least_19': 0.0001,
                    'at_least_20': 0.00001
                },
                'total_simulations': 0,
                'simulation_quality': {'convergence_quality': 'fallback'},
                'confidence_metrics': {
                    'passes_criteria': {'any_criterion': False},
                    'recommendation': {'action': 'WAIT', 'confidence': 'LOW'}
                }
            }
            
            # Estimativas por ticket
            for i, ticket in enumerate(tickets):
                ticket_estimate = {
                    'ticket_number': i + 1,
                    'numbers': ticket,
                    'hit_probabilities': {16: 0.005, 17: 0.002, 18: 0.0005, 19: 0.0001, 20: 0.00001},
                    'statistics': {'mean_hits': 10.0, 'std_hits': 2.0}
                }
                fallback_estimates['individual_tickets'].append(ticket_estimate)
            
            return fallback_estimates
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar estimativas de fallback")
            return {}
    
    def _log_simulation_results(self, results: Dict[str, Any]):
        """Log detalhado dos resultados da simulação"""
        try:
            # Log métricas principais
            combined_probs = results.get('combined_probabilities', {})
            
            loto_logger.log_monte_carlo_simulation(
                {
                    'probabilities': combined_probs,
                    'total_simulations': results.get('total_simulations', 0),
                    'quality': results.get('simulation_quality', {}),
                    'recommendation': results.get('confidence_metrics', {}).get('recommendation', {})
                },
                0  # Duration será logado pelo decorator
            )
            
            # Log detalhado
            if combined_probs:
                loto_logger.log_info(
                    f"Simulação Monte Carlo: P(≥18)={combined_probs.get('at_least_18', 0):.4f}, "
                    f"P(≥17)={combined_probs.get('at_least_17', 0):.4f}, "
                    f"P(≥16)={combined_probs.get('at_least_16', 0):.4f}"
                )
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao fazer log dos resultados")

# Instância global
monte_carlo_simulator = LotomaniaMonteCarloSimulator()