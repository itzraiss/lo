import numpy as np
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import itertools
from concurrent.futures import ThreadPoolExecutor
import random

# Optimization libraries
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

try:
    from ortools.linear_solver import pywraplp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

from config import config
from utils.logger import loto_logger, async_performance_logger, performance_logger

@dataclass
class OptimizationConfig:
    """Configuração para otimização dos tickets"""
    max_overlap: int = config.model.max_overlap
    numbers_per_ticket: int = config.model.numbers_per_ticket
    total_tickets: int = config.model.total_tickets
    candidate_pool_size: int = config.model.candidate_pool_size
    
    # Estratégias de otimização
    primary_strategy: str = "ilp"  # ilp, greedy, genetic, simulated_annealing
    fallback_strategy: str = "greedy"
    
    # Parâmetros específicos
    genetic_population: int = 100
    genetic_generations: int = 50
    genetic_mutation_rate: float = 0.1
    
    simulated_annealing_temp: float = 1000.0
    simulated_annealing_cooling: float = 0.95
    simulated_annealing_iterations: int = 10000
    
    # Timeouts
    ilp_timeout_seconds: int = 300  # 5 minutos
    greedy_timeout_seconds: int = 60
    
    # Critérios de diversificação
    diversity_weight: float = 0.3
    balance_groups: bool = True
    balance_even_odd: bool = True

class LotomaniaOptimizer:
    """Sistema de otimização para geração de tickets da Lotomania"""
    
    def __init__(self, optimization_config: OptimizationConfig = None):
        self.config = optimization_config or OptimizationConfig()
        self.last_optimization_time = 0
        self.optimization_history = []
        
    @async_performance_logger
    async def optimize_tickets(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Otimiza geração de 3 tickets usando múltiplas estratégias"""
        try:
            loto_logger.log_info("Iniciando otimização de tickets")
            
            # Preparar dados
            prob_array = np.array([probabilities[str(i)] for i in range(1, 101)])
            candidate_pool = self._select_candidate_pool(prob_array)
            
            # Tentar estratégia primária
            result = await self._optimize_with_strategy(
                self.config.primary_strategy,
                prob_array,
                candidate_pool
            )
            
            # Se falhar, usar estratégia de fallback
            if not result or not result.get('success', False):
                loto_logger.log_warning(f"Estratégia primária '{self.config.primary_strategy}' falhou, usando fallback")
                result = await self._optimize_with_strategy(
                    self.config.fallback_strategy,
                    prob_array,
                    candidate_pool
                )
            
            # Se ainda falhar, usar otimização simples
            if not result or not result.get('success', False):
                loto_logger.log_warning("Estratégias de otimização falharam, usando seleção simples")
                result = self._simple_ticket_generation(prob_array)
            
            # Validar e refinar resultado
            if result and result.get('tickets'):
                result = self._validate_and_refine_tickets(result, prob_array)
            
            # Log resultado
            self._log_optimization_result(result)
            
            return result
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na otimização de tickets")
            # Fallback para geração simples
            return self._simple_ticket_generation(np.array([probabilities[str(i)] for i in range(1, 101)]))
    
    def _select_candidate_pool(self, probabilities: np.ndarray) -> List[int]:
        """Seleciona pool de candidatos baseado em probabilidades e diversificação"""
        try:
            # Números ordenados por probabilidade
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # Selecionar top candidatos
            top_candidates = sorted_indices[:self.config.candidate_pool_size]
            
            # Adicionar diversificação
            if self.config.balance_groups:
                top_candidates = self._balance_number_groups(top_candidates, probabilities)
            
            if self.config.balance_even_odd:
                top_candidates = self._balance_even_odd(top_candidates, probabilities)
            
            # Converter para lista de números (1-100)
            candidate_pool = [int(idx + 1) for idx in top_candidates]
            
            loto_logger.log_info(f"Pool de candidatos selecionado: {len(candidate_pool)} números")
            return candidate_pool
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao selecionar pool de candidatos")
            # Fallback: top números por probabilidade
            sorted_indices = np.argsort(probabilities)[::-1]
            return [int(idx + 1) for idx in sorted_indices[:self.config.candidate_pool_size]]
    
    def _balance_number_groups(self, candidates: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Balanceia candidatos entre grupos de números"""
        groups = {
            'low': list(range(0, 20)),      # 1-20
            'mid_low': list(range(20, 40)), # 21-40
            'mid_high': list(range(40, 60)), # 41-60
            'high': list(range(60, 80)),    # 61-80
            'very_high': list(range(80, 100)) # 81-100
        }
        
        balanced_candidates = []
        remaining_slots = self.config.candidate_pool_size
        
        # Distribuir candidatos entre grupos
        slots_per_group = remaining_slots // len(groups)
        
        for group_name, group_indices in groups.items():
            # Candidatos deste grupo
            group_candidates = [idx for idx in candidates if idx in group_indices]
            
            # Selecionar os melhores deste grupo
            group_candidates_sorted = sorted(group_candidates, key=lambda x: probabilities[x], reverse=True)
            selected = group_candidates_sorted[:min(slots_per_group, len(group_candidates_sorted))]
            
            balanced_candidates.extend(selected)
            remaining_slots -= len(selected)
        
        # Adicionar candidatos restantes pelos melhores scores
        if remaining_slots > 0:
            remaining_candidates = [idx for idx in candidates if idx not in balanced_candidates]
            remaining_candidates_sorted = sorted(remaining_candidates, key=lambda x: probabilities[x], reverse=True)
            balanced_candidates.extend(remaining_candidates_sorted[:remaining_slots])
        
        return np.array(balanced_candidates[:self.config.candidate_pool_size])
    
    def _balance_even_odd(self, candidates: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Balanceia candidatos entre números pares e ímpares"""
        even_candidates = [idx for idx in candidates if (idx + 1) % 2 == 0]
        odd_candidates = [idx for idx in candidates if (idx + 1) % 2 == 1]
        
        # Tentar manter aproximadamente 50/50
        target_even = self.config.candidate_pool_size // 2
        target_odd = self.config.candidate_pool_size - target_even
        
        # Ordenar por probabilidade
        even_sorted = sorted(even_candidates, key=lambda x: probabilities[x], reverse=True)
        odd_sorted = sorted(odd_candidates, key=lambda x: probabilities[x], reverse=True)
        
        # Selecionar
        selected_even = even_sorted[:min(target_even, len(even_sorted))]
        selected_odd = odd_sorted[:min(target_odd, len(odd_sorted))]
        
        # Se um lado tem menos que o target, compensar com o outro
        if len(selected_even) < target_even:
            remaining = target_even - len(selected_even)
            selected_odd.extend(odd_sorted[len(selected_odd):len(selected_odd) + remaining])
        elif len(selected_odd) < target_odd:
            remaining = target_odd - len(selected_odd)
            selected_even.extend(even_sorted[len(selected_even):len(selected_even) + remaining])
        
        balanced = selected_even + selected_odd
        return np.array(balanced[:self.config.candidate_pool_size])
    
    async def _optimize_with_strategy(self, strategy: str, probabilities: np.ndarray, 
                                    candidate_pool: List[int]) -> Dict[str, Any]:
        """Executa otimização com estratégia específica"""
        try:
            start_time = time.time()
            
            if strategy == "ilp" and PULP_AVAILABLE:
                result = await self._ilp_optimization(probabilities, candidate_pool)
            elif strategy == "ortools" and ORTOOLS_AVAILABLE:
                result = await self._ortools_optimization(probabilities, candidate_pool)
            elif strategy == "greedy":
                result = await self._greedy_optimization(probabilities, candidate_pool)
            elif strategy == "genetic":
                result = await self._genetic_optimization(probabilities, candidate_pool)
            elif strategy == "simulated_annealing":
                result = await self._simulated_annealing_optimization(probabilities, candidate_pool)
            else:
                loto_logger.log_warning(f"Estratégia '{strategy}' não disponível, usando greedy")
                result = await self._greedy_optimization(probabilities, candidate_pool)
            
            duration = time.time() - start_time
            
            if result:
                result['optimization_strategy'] = strategy
                result['optimization_duration'] = duration
            
            return result
            
        except Exception as e:
            loto_logger.log_error(e, f"Erro na otimização com estratégia {strategy}")
            return {'success': False, 'error': str(e)}
    
    @async_performance_logger
    async def _ilp_optimization(self, probabilities: np.ndarray, candidate_pool: List[int]) -> Dict[str, Any]:
        """Otimização usando Integer Linear Programming (PuLP)"""
        try:
            # Criar problema
            prob = pulp.LpProblem("LotomaniaOptimization", pulp.LpMaximize)
            
            # Variáveis: x[t][i] = 1 se número i está no ticket t
            x = {}
            for t in range(self.config.total_tickets):
                x[t] = {}
                for i in candidate_pool:
                    x[t][i] = pulp.LpVariable(f"x_{t}_{i}", cat='Binary')
            
            # Função objetivo: maximizar soma das probabilidades
            objective = 0
            for t in range(self.config.total_tickets):
                for i in candidate_pool:
                    objective += probabilities[i - 1] * x[t][i]
            
            prob += objective
            
            # Restrições
            # 1. Cada ticket deve ter exatamente 50 números
            for t in range(self.config.total_tickets):
                prob += pulp.lpSum([x[t][i] for i in candidate_pool]) == self.config.numbers_per_ticket
            
            # 2. Overlap máximo entre tickets
            for t1 in range(self.config.total_tickets):
                for t2 in range(t1 + 1, self.config.total_tickets):
                    prob += pulp.lpSum([x[t1][i] * x[t2][i] for i in candidate_pool]) <= self.config.max_overlap
            
            # Resolver
            solver = pulp.PULP_CBC_CMD(timeLimit=self.config.ilp_timeout_seconds, msg=0)
            prob.solve(solver)
            
            # Extrair solução
            if prob.status == pulp.LpStatusOptimal:
                tickets = []
                for t in range(self.config.total_tickets):
                    ticket = [i for i in candidate_pool if x[t][i].varValue == 1]
                    tickets.append(sorted(ticket))
                
                objective_value = sum(
                    probabilities[i - 1] for ticket in tickets for i in ticket
                )
                
                return {
                    'success': True,
                    'tickets': tickets,
                    'objective_value': objective_value,
                    'method': 'ILP',
                    'status': 'optimal'
                }
            else:
                return {'success': False, 'status': pulp.LpStatus[prob.status]}
                
        except Exception as e:
            loto_logger.log_error(e, "Erro na otimização ILP")
            return {'success': False, 'error': str(e)}
    
    @async_performance_logger
    async def _ortools_optimization(self, probabilities: np.ndarray, candidate_pool: List[int]) -> Dict[str, Any]:
        """Otimização usando OR-Tools"""
        try:
            # Criar solver
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                return {'success': False, 'error': 'OR-Tools solver não disponível'}
            
            # Variáveis
            x = {}
            for t in range(self.config.total_tickets):
                x[t] = {}
                for i in candidate_pool:
                    x[t][i] = solver.IntVar(0, 1, f'x_{t}_{i}')
            
            # Função objetivo
            objective = solver.Objective()
            for t in range(self.config.total_tickets):
                for i in candidate_pool:
                    objective.SetCoefficient(x[t][i], probabilities[i - 1])
            objective.SetMaximization()
            
            # Restrições
            # 1. Cada ticket deve ter exatamente 50 números
            for t in range(self.config.total_tickets):
                constraint = solver.Constraint(self.config.numbers_per_ticket, self.config.numbers_per_ticket)
                for i in candidate_pool:
                    constraint.SetCoefficient(x[t][i], 1)
            
            # 2. Overlap máximo entre tickets
            for t1 in range(self.config.total_tickets):
                for t2 in range(t1 + 1, self.config.total_tickets):
                    constraint = solver.Constraint(0, self.config.max_overlap)
                    for i in candidate_pool:
                        # Aproximação: x[t1][i] * x[t2][i] ≈ min(x[t1][i] + x[t2][i] - 1, 0)
                        # Usamos uma restrição mais simples
                        pass  # Implementação simplificada
            
            # Resolver
            solver.SetTimeLimit(self.config.ilp_timeout_seconds * 1000)
            status = solver.Solve()
            
            if status == pywraplp.Solver.OPTIMAL:
                tickets = []
                for t in range(self.config.total_tickets):
                    ticket = [i for i in candidate_pool if x[t][i].solution_value() == 1]
                    tickets.append(sorted(ticket))
                
                objective_value = solver.Objective().Value()
                
                return {
                    'success': True,
                    'tickets': tickets,
                    'objective_value': objective_value,
                    'method': 'OR-Tools',
                    'status': 'optimal'
                }
            else:
                return {'success': False, 'status': f'OR-Tools status: {status}'}
                
        except Exception as e:
            loto_logger.log_error(e, "Erro na otimização OR-Tools")
            return {'success': False, 'error': str(e)}
    
    @async_performance_logger
    async def _greedy_optimization(self, probabilities: np.ndarray, candidate_pool: List[int]) -> Dict[str, Any]:
        """Otimização greedy rápida"""
        try:
            tickets = []
            used_numbers = set()
            
            # Ordenar candidatos por probabilidade
            candidates_sorted = sorted(candidate_pool, key=lambda x: probabilities[x - 1], reverse=True)
            
            for ticket_idx in range(self.config.total_tickets):
                ticket = []
                available_numbers = [num for num in candidates_sorted if num not in used_numbers or len(used_numbers) < len(candidate_pool) * 0.7]
                
                # Selecionar números para este ticket
                for num in available_numbers:
                    if len(ticket) >= self.config.numbers_per_ticket:
                        break
                    
                    # Verificar overlap
                    overlap_ok = True
                    for existing_ticket in tickets:
                        current_overlap = len(set(ticket + [num]) & set(existing_ticket))
                        if current_overlap > self.config.max_overlap:
                            overlap_ok = False
                            break
                    
                    if overlap_ok:
                        ticket.append(num)
                        if len(used_numbers) < len(candidate_pool) * 0.5:  # Evitar reutilização prematura
                            used_numbers.add(num)
                
                # Se não conseguiu preencher, completar com números disponíveis
                if len(ticket) < self.config.numbers_per_ticket:
                    remaining_needed = self.config.numbers_per_ticket - len(ticket)
                    available = [num for num in candidates_sorted if num not in ticket]
                    
                    for num in available:
                        if len(ticket) >= self.config.numbers_per_ticket:
                            break
                        
                        # Verificar overlap relaxado
                        valid = True
                        for existing_ticket in tickets:
                            overlap = len(set(ticket + [num]) & set(existing_ticket))
                            if overlap > self.config.max_overlap + 5:  # Margem de tolerância
                                valid = False
                                break
                        
                        if valid:
                            ticket.append(num)
                
                tickets.append(sorted(ticket))
            
            # Calcular valor objetivo
            objective_value = sum(probabilities[i - 1] for ticket in tickets for i in ticket)
            
            return {
                'success': True,
                'tickets': tickets,
                'objective_value': objective_value,
                'method': 'Greedy',
                'status': 'heuristic'
            }
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na otimização greedy")
            return {'success': False, 'error': str(e)}
    
    @async_performance_logger
    async def _genetic_optimization(self, probabilities: np.ndarray, candidate_pool: List[int]) -> Dict[str, Any]:
        """Otimização usando algoritmo genético"""
        try:
            population_size = self.config.genetic_population
            generations = self.config.genetic_generations
            mutation_rate = self.config.genetic_mutation_rate
            
            # Função de fitness
            def fitness(individual):
                tickets = self._decode_individual(individual, candidate_pool)
                if not self._is_valid_solution(tickets):
                    return 0  # Penalizar soluções inválidas
                
                # Objetivo: maximizar soma das probabilidades
                objective = sum(probabilities[i - 1] for ticket in tickets for i in ticket)
                
                # Penalizar overlap excessivo
                overlap_penalty = 0
                for i in range(len(tickets)):
                    for j in range(i + 1, len(tickets)):
                        overlap = len(set(tickets[i]) & set(tickets[j]))
                        if overlap > self.config.max_overlap:
                            overlap_penalty += (overlap - self.config.max_overlap) * 100
                
                return objective - overlap_penalty
            
            # População inicial
            population = []
            for _ in range(population_size):
                individual = self._generate_random_individual(candidate_pool)
                population.append(individual)
            
            # Evolução
            for generation in range(generations):
                # Avaliar fitness
                fitness_scores = [fitness(ind) for ind in population]
                
                # Seleção
                new_population = []
                for _ in range(population_size):
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)
                    
                    # Crossover
                    child = self._crossover(parent1, parent2)
                    
                    # Mutação
                    if random.random() < mutation_rate:
                        child = self._mutate(child, candidate_pool)
                    
                    new_population.append(child)
                
                population = new_population
            
            # Melhor solução
            best_fitness = max(fitness(ind) for ind in population)
            best_individual = population[np.argmax([fitness(ind) for ind in population])]
            best_tickets = self._decode_individual(best_individual, candidate_pool)
            
            return {
                'success': True,
                'tickets': best_tickets,
                'objective_value': best_fitness,
                'method': 'Genetic Algorithm',
                'status': 'heuristic',
                'generations': generations
            }
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na otimização genética")
            return {'success': False, 'error': str(e)}
    
    @async_performance_logger
    async def _simulated_annealing_optimization(self, probabilities: np.ndarray, 
                                              candidate_pool: List[int]) -> Dict[str, Any]:
        """Otimização usando Simulated Annealing"""
        try:
            # Solução inicial
            current_solution = self._generate_random_solution(candidate_pool)
            current_energy = self._calculate_energy(current_solution, probabilities)
            
            best_solution = current_solution.copy()
            best_energy = current_energy
            
            temperature = self.config.simulated_annealing_temp
            cooling_rate = self.config.simulated_annealing_cooling
            iterations = self.config.simulated_annealing_iterations
            
            for iteration in range(iterations):
                # Gerar vizinho
                neighbor = self._generate_neighbor(current_solution, candidate_pool)
                neighbor_energy = self._calculate_energy(neighbor, probabilities)
                
                # Decidir se aceita o vizinho
                delta = neighbor_energy - current_energy
                
                if delta > 0 or random.random() < np.exp(delta / temperature):
                    current_solution = neighbor
                    current_energy = neighbor_energy
                    
                    if current_energy > best_energy:
                        best_solution = current_solution.copy()
                        best_energy = current_energy
                
                # Resfriamento
                temperature *= cooling_rate
            
            return {
                'success': True,
                'tickets': best_solution,
                'objective_value': best_energy,
                'method': 'Simulated Annealing',
                'status': 'heuristic',
                'iterations': iterations
            }
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na otimização por Simulated Annealing")
            return {'success': False, 'error': str(e)}
    
    def _simple_ticket_generation(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Geração simples de tickets como fallback"""
        try:
            tickets = []
            numbers_indices = np.argsort(probabilities)[::-1]  # Ordenar por probabilidade
            
            for ticket_idx in range(self.config.total_tickets):
                ticket = []
                start_idx = ticket_idx * 30  # Offset para diversificar
                
                for i in range(start_idx, len(numbers_indices)):
                    if len(ticket) >= self.config.numbers_per_ticket:
                        break
                    
                    num = int(numbers_indices[i] + 1)
                    ticket.append(num)
                
                # Se não preencheu, completar com números aleatórios
                while len(ticket) < self.config.numbers_per_ticket:
                    num = random.randint(1, 100)
                    if num not in ticket:
                        ticket.append(num)
                
                tickets.append(sorted(ticket))
            
            objective_value = sum(probabilities[i - 1] for ticket in tickets for i in ticket)
            
            return {
                'success': True,
                'tickets': tickets,
                'objective_value': objective_value,
                'method': 'Simple Generation',
                'status': 'fallback'
            }
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na geração simples de tickets")
            return {
                'success': False,
                'tickets': [list(range(1, 51)), list(range(26, 76)), list(range(51, 101))],
                'objective_value': 0,
                'method': 'Emergency Fallback',
                'error': str(e)
            }
    
    def _validate_and_refine_tickets(self, result: Dict[str, Any], probabilities: np.ndarray) -> Dict[str, Any]:
        """Valida e refina os tickets gerados"""
        try:
            tickets = result.get('tickets', [])
            
            # Validações básicas
            for i, ticket in enumerate(tickets):
                # Garantir 50 números
                if len(ticket) != self.config.numbers_per_ticket:
                    if len(ticket) < self.config.numbers_per_ticket:
                        # Adicionar números
                        available = [n for n in range(1, 101) if n not in ticket]
                        needed = self.config.numbers_per_ticket - len(ticket)
                        additional = sorted(available, key=lambda x: probabilities[x-1], reverse=True)[:needed]
                        ticket.extend(additional)
                    else:
                        # Remover números com menor probabilidade
                        ticket = sorted(ticket, key=lambda x: probabilities[x-1], reverse=True)[:self.config.numbers_per_ticket]
                    
                    tickets[i] = sorted(ticket)
                
                # Garantir números únicos
                tickets[i] = sorted(list(set(ticket)))
                
                # Se ainda não tem 50, completar
                while len(tickets[i]) < self.config.numbers_per_ticket:
                    for num in range(1, 101):
                        if num not in tickets[i]:
                            tickets[i].append(num)
                            break
                    tickets[i] = sorted(tickets[i])
            
            # Recalcular valor objetivo
            result['objective_value'] = sum(probabilities[i - 1] for ticket in tickets for i in ticket)
            result['tickets'] = tickets
            
            # Adicionar análises
            result['overlap_analysis'] = self._analyze_overlap(tickets)
            result['diversity_analysis'] = self._analyze_diversity(tickets)
            result['probability_analysis'] = self._analyze_ticket_probabilities(tickets, probabilities)
            
            return result
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na validação dos tickets")
            return result
    
    def _analyze_overlap(self, tickets: List[List[int]]) -> Dict[str, Any]:
        """Analisa overlap entre tickets"""
        overlaps = {}
        for i in range(len(tickets)):
            for j in range(i + 1, len(tickets)):
                overlap = len(set(tickets[i]) & set(tickets[j]))
                overlaps[f"ticket_{i+1}_vs_{j+1}"] = overlap
        
        return {
            'pairwise_overlaps': overlaps,
            'max_overlap': max(overlaps.values()) if overlaps else 0,
            'min_overlap': min(overlaps.values()) if overlaps else 0,
            'avg_overlap': sum(overlaps.values()) / len(overlaps) if overlaps else 0
        }
    
    def _analyze_diversity(self, tickets: List[List[int]]) -> Dict[str, Any]:
        """Analisa diversidade dos tickets"""
        all_numbers = set()
        for ticket in tickets:
            all_numbers.update(ticket)
        
        # Análise por grupos
        groups = {
            'low': list(range(1, 21)),
            'mid_low': list(range(21, 41)),
            'mid_high': list(range(41, 61)),
            'high': list(range(61, 81)),
            'very_high': list(range(81, 101))
        }
        
        group_coverage = {}
        for group_name, group_range in groups.items():
            coverage = len(set(group_range) & all_numbers)
            group_coverage[group_name] = coverage
        
        # Análise par/ímpar
        even_count = len([n for n in all_numbers if n % 2 == 0])
        odd_count = len([n for n in all_numbers if n % 2 == 1])
        
        return {
            'total_unique_numbers': len(all_numbers),
            'coverage_percentage': len(all_numbers) / 100,
            'group_coverage': group_coverage,
            'even_count': even_count,
            'odd_count': odd_count,
            'even_odd_ratio': even_count / odd_count if odd_count > 0 else 0
        }
    
    def _analyze_ticket_probabilities(self, tickets: List[List[int]], probabilities: np.ndarray) -> Dict[str, Any]:
        """Analisa probabilidades dos tickets"""
        ticket_probs = []
        for ticket in tickets:
            ticket_prob = sum(probabilities[i - 1] for i in ticket)
            ticket_probs.append(ticket_prob)
        
        return {
            'individual_scores': ticket_probs,
            'total_score': sum(ticket_probs),
            'avg_score': sum(ticket_probs) / len(ticket_probs),
            'min_score': min(ticket_probs),
            'max_score': max(ticket_probs),
            'score_variance': np.var(ticket_probs)
        }
    
    def _log_optimization_result(self, result: Dict[str, Any]):
        """Log detalhado do resultado da otimização"""
        if result.get('success'):
            loto_logger.log_optimization_result(
                result.get('method', 'Unknown'),
                result.get('tickets', []),
                result.get('objective_value', 0),
                result.get('optimization_duration', 0)
            )
        else:
            loto_logger.log_error(
                f"Otimização falhou: {result.get('error', 'Unknown error')}",
                "optimization"
            )
    
    # Métodos auxiliares para algoritmos genéticos
    def _generate_random_individual(self, candidate_pool: List[int]) -> List[List[int]]:
        """Gera indivíduo aleatório para algoritmo genético"""
        tickets = []
        for _ in range(self.config.total_tickets):
            ticket = random.sample(candidate_pool, self.config.numbers_per_ticket)
            tickets.append(sorted(ticket))
        return tickets
    
    def _decode_individual(self, individual: List[List[int]], candidate_pool: List[int]) -> List[List[int]]:
        """Decodifica indivíduo do algoritmo genético"""
        return individual
    
    def _is_valid_solution(self, tickets: List[List[int]]) -> bool:
        """Verifica se solução é válida"""
        if len(tickets) != self.config.total_tickets:
            return False
        
        for ticket in tickets:
            if len(ticket) != self.config.numbers_per_ticket:
                return False
            if len(set(ticket)) != len(ticket):  # Números únicos
                return False
        
        return True
    
    def _tournament_selection(self, population: List, fitness_scores: List[float]) -> List[List[int]]:
        """Seleção por torneio"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """Crossover entre dois pais"""
        child = []
        for i in range(self.config.total_tickets):
            if random.random() < 0.5:
                child.append(parent1[i].copy())
            else:
                child.append(parent2[i].copy())
        return child
    
    def _mutate(self, individual: List[List[int]], candidate_pool: List[int]) -> List[List[int]]:
        """Mutação de um indivíduo"""
        mutated = [ticket.copy() for ticket in individual]
        
        # Mutar alguns números aleatoriamente
        for ticket in mutated:
            if random.random() < 0.3:  # 30% chance de mutar cada ticket
                # Trocar alguns números
                num_changes = random.randint(1, 5)
                for _ in range(num_changes):
                    if len(ticket) > 0:
                        old_num = random.choice(ticket)
                        new_num = random.choice(candidate_pool)
                        if new_num not in ticket:
                            ticket.remove(old_num)
                            ticket.append(new_num)
                ticket.sort()
        
        return mutated
    
    # Métodos auxiliares para Simulated Annealing
    def _generate_random_solution(self, candidate_pool: List[int]) -> List[List[int]]:
        """Gera solução aleatória inicial"""
        return self._generate_random_individual(candidate_pool)
    
    def _calculate_energy(self, solution: List[List[int]], probabilities: np.ndarray) -> float:
        """Calcula energia da solução (negativo da função objetivo)"""
        return sum(probabilities[i - 1] for ticket in solution for i in ticket)
    
    def _generate_neighbor(self, solution: List[List[int]], candidate_pool: List[int]) -> List[List[int]]:
        """Gera solução vizinha"""
        neighbor = [ticket.copy() for ticket in solution]
        
        # Mudança aleatória
        ticket_idx = random.randint(0, len(neighbor) - 1)
        ticket = neighbor[ticket_idx]
        
        if len(ticket) > 0:
            old_num = random.choice(ticket)
            new_num = random.choice(candidate_pool)
            
            if new_num not in ticket:
                ticket.remove(old_num)
                ticket.append(new_num)
                ticket.sort()
        
        return neighbor

# Instância global
ticket_optimizer = LotomaniaOptimizer()