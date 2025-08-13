"""
Dados de exemplo para demonstração do sistema Lotomania AI
Usado quando MongoDB não está configurado
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

def generate_sample_contests(count: int = 1000) -> List[Dict[str, Any]]:
    """Gera concursos de exemplo para demonstração"""
    contests = []
    start_date = datetime(2020, 1, 1)
    
    for i in range(count):
        # Gerar 20 números aleatórios de 1 a 100
        numbers = sorted(random.sample(range(1, 101), 20))
        
        # Data incremental (concursos semanais)
        contest_date = start_date + timedelta(days=i * 3)  # A cada 3 dias
        
        contest = {
            'contest': 2000 + i,
            'date': contest_date.strftime('%Y-%m-%d'),
            'numbers': numbers,
            'raw': {
                'generated': True,
                'sample_data': True
            },
            'created_at': datetime.utcnow()
        }
        
        contests.append(contest)
    
    return contests

def get_demo_probabilities() -> Dict[str, float]:
    """Gera probabilidades de exemplo"""
    probabilities = {}
    
    for i in range(1, 101):
        # Probabilidade base com variação
        base_prob = 0.20  # 20% para cada número
        
        # Alguns números "mais prováveis" para demonstração
        if i in [7, 13, 21, 33, 42, 57, 66, 71, 84, 99]:
            variation = random.uniform(0.02, 0.08)
        else:
            variation = random.uniform(-0.05, 0.05)
            
        prob = max(0.01, min(0.35, base_prob + variation))
        probabilities[str(i)] = prob
    
    return probabilities

def get_demo_tickets() -> List[List[int]]:
    """Gera tickets de exemplo otimizados"""
    # Simular 3 tickets com overlap controlado
    all_numbers = list(range(1, 101))
    
    # Ticket 1: números com probabilidades mais altas
    high_prob_numbers = [7, 13, 21, 33, 42, 57, 66, 71, 84, 99]
    ticket1 = high_prob_numbers + random.sample([n for n in all_numbers if n not in high_prob_numbers], 40)
    
    # Ticket 2: mix de altos e médios
    ticket2 = high_prob_numbers[:5] + random.sample([n for n in all_numbers if n not in high_prob_numbers[:5]], 45)
    
    # Ticket 3: estratégia diferente
    ticket3 = high_prob_numbers[5:] + random.sample([n for n in all_numbers if n not in high_prob_numbers[5:]], 45)
    
    return [sorted(ticket1), sorted(ticket2), sorted(ticket3)]

# Dados de exemplo pré-gerados para demonstração rápida
SAMPLE_PREDICTION = {
    'generated_at': datetime.utcnow(),
    'model_version': 'demo_v1.0.0',
    'probabilities': get_demo_probabilities(),
    'tickets': get_demo_tickets(),
    'optimization_result': {
        'success': True,
        'method': 'Demo Optimization',
        'objective_value': 45.6,
        'optimization_duration': 2.3,
        'overlap_analysis': {
            'pairwise_overlaps': {
                'ticket_1_vs_2': 25,
                'ticket_1_vs_3': 28,
                'ticket_2_vs_3': 22
            },
            'max_overlap': 28,
            'avg_overlap': 25
        },
        'diversity_analysis': {
            'total_unique_numbers': 95,
            'coverage_percentage': 0.95,
            'even_count': 47,
            'odd_count': 48
        }
    },
    'simulation_result': {
        'individual_tickets': [
            {
                'ticket_number': 1,
                'numbers': get_demo_tickets()[0],
                'hit_probabilities': {16: 0.15, 17: 0.08, 18: 0.03, 19: 0.01, 20: 0.002},
                'statistics': {'mean_hits': 10.2, 'std_hits': 2.1}
            },
            {
                'ticket_number': 2,
                'numbers': get_demo_tickets()[1],
                'hit_probabilities': {16: 0.12, 17: 0.06, 18: 0.025, 19: 0.008, 20: 0.001},
                'statistics': {'mean_hits': 9.8, 'std_hits': 2.0}
            },
            {
                'ticket_number': 3,
                'numbers': get_demo_tickets()[2],
                'hit_probabilities': {16: 0.14, 17: 0.07, 18: 0.028, 19: 0.009, 20: 0.0015},
                'statistics': {'mean_hits': 10.0, 'std_hits': 2.1}
            }
        ],
        'combined_probabilities': {
            'at_least_16': 0.35,
            'at_least_17': 0.18,
            'at_least_18': 0.075,
            'at_least_19': 0.025,
            'at_least_20': 0.004
        },
        'total_simulations': 100000,
        'simulation_quality': {'convergence_quality': 'excellent'},
        'confidence_metrics': {
            'passes_criteria': {
                '18_hits': True,  # 7.5% > 2%
                '17_hits': True,  # 18% > 10%
                '16_hits': False, # 35% < 50%
                'any_criterion': True
            },
            'recommendation': {
                'action': 'PLAY',
                'confidence': 'HIGH',
                'reason': 'Probabilidades atingem critérios mínimos de confiança',
                'details': {
                    'best_18_probability': 0.075,
                    'best_17_probability': 0.18,
                    'expected_value': 156.75,
                    'risk_assessment': 'MEDIUM'
                }
            }
        }
    },
    'confidence_passed': True,
    'recommendation': {
        'action': 'PLAY',
        'confidence': 'HIGH',
        'reason': 'Probabilidades atingem critérios mínimos de confiança'
    },
    'cycle_duration': 45.2
}