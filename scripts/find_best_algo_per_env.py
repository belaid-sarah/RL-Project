"""
Script pour trouver le meilleur algorithme pour chaque environnement

Teste tous les algorithmes sur tous les environnements et identifie le meilleur.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime
import traceback

# Tous les environnements
from envs.lineworld_simple import LineWorldSimple
from envs.gridworld_simple import GridWorldSimple
from envs.rps import TwoRoundRPS
from envs.monty_hall_level1 import MontyHallLevel1
from envs.monty_hall_level2 import MontyHallLevel2

# Tous les algorithmes
from algos.policy_iteration import PolicyIteration
from algos.value_iteration import ValueIteration
from algos.monte_carlo import MonteCarloES, OnPolicyMonteCarlo, OffPolicyMonteCarlo
from algos.sarsa import SARSAAgent
from algos.q_learning import QLearningAgent
from algos.expected_sarsa import ExpectedSARSAAgent
from algos.dyna_q import DynaQAgent
from algos.dyna_q_plus import DynaQPlusAgent

# Configuration
CONFIG = {
    'num_episodes': {
        'LineWorldSimple': 2000,
        'GridWorldSimple': 3000,
        'TwoRoundRPS': 1000,
        'MontyHallLevel1': 1000,
        'MontyHallLevel2': 2000,
    },
    'evaluation_episodes': 100,
    'verbose': False,
}

# Hyperparamètres optimisés
HYPERPARAMETERS = {
    'Q-Learning': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05},
    'SARSA': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05},
    'Expected SARSA': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05},
    'Dyna-Q': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05, 'n_planning_steps': 5},
    'Dyna-Q+': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05, 'n_planning_steps': 5},
    'Policy Iteration': {'gamma': 0.99, 'theta': 1e-5},
    'Value Iteration': {'gamma': 0.99, 'theta': 1e-5},
    'Monte Carlo ES': {'gamma': 0.99},
    'On-Policy Monte Carlo': {'gamma': 0.99, 'epsilon': 0.2},
    'Off-Policy Monte Carlo': {'gamma': 0.99, 'epsilon': 0.2},
}

# Mapping algorithmes -> classes
ALGORITHMS = {
    'Q-Learning': QLearningAgent,
    'SARSA': SARSAAgent,
    'Expected SARSA': ExpectedSARSAAgent,
    'Dyna-Q': DynaQAgent,
    'Dyna-Q+': DynaQPlusAgent,
    'Policy Iteration': PolicyIteration,
    'Value Iteration': ValueIteration,
    'Monte Carlo ES': MonteCarloES,
    'On-Policy Monte Carlo': OnPolicyMonteCarlo,
    'Off-Policy Monte Carlo': OffPolicyMonteCarlo,
}

def create_env(env_name):
    """Crée une instance d'environnement"""
    if env_name == 'LineWorldSimple':
        return LineWorldSimple(length=15)  # Réduire pour Policy/Value Iteration
    elif env_name == 'GridWorldSimple':
        return GridWorldSimple(width=8, height=8)  # Réduire pour Policy/Value Iteration
    elif env_name == 'TwoRoundRPS':
        return TwoRoundRPS()
    elif env_name == 'MontyHallLevel1':
        return MontyHallLevel1()
    elif env_name == 'MontyHallLevel2':
        return MontyHallLevel2()
    else:
        raise ValueError(f"Environnement inconnu: {env_name}")

def test_algorithm_on_environment(algo_name, env_name):
    """Teste un algorithme sur un environnement"""
    try:
        # Créer environnement
        env = create_env(env_name)
        
        # Récupérer hyperparamètres
        hyperparams = HYPERPARAMETERS.get(algo_name, {})
        
        # Créer agent
        algo_class = ALGORITHMS[algo_name]
        agent = algo_class(env, **hyperparams)
        
        # Nombre d'épisodes selon l'environnement
        num_episodes = CONFIG['num_episodes'].get(env_name, 1000)
        
        # Policy/Value Iteration n'ont pas besoin de beaucoup d'épisodes
        if algo_name in ['Policy Iteration', 'Value Iteration']:
            num_episodes = 50  # Juste pour l'interface
        
        # Entraînement
        start_time = time.time()
        agent.train(num_episodes=num_episodes, verbose=CONFIG['verbose'])
        training_time = time.time() - start_time
        
        # Évaluation
        eval_results = agent.evaluate(num_episodes=CONFIG['evaluation_episodes'], verbose=False)
        
        # Résultats
        results = {
            'algorithm': algo_name,
            'environment': env_name,
            'training_time': training_time,
            'mean_reward': float(eval_results['mean_reward']),
            'std_reward': float(eval_results['std_reward']),
            'success_rate': float(eval_results['success_rate']),
            'mean_steps': float(eval_results['mean_steps']),
            'hyperparameters': hyperparams,
        }
        
        return results
        
    except Exception as e:
        return {
            'algorithm': algo_name,
            'environment': env_name,
            'error': str(e),
        }

def find_best_algorithm(results_by_env):
    """Trouve le meilleur algorithme pour chaque environnement"""
    best_by_env = {}
    
    for env_name, results in results_by_env.items():
        # Filtrer les résultats sans erreur
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            best_by_env[env_name] = None
            continue
        
        # Critère : Success rate d'abord, puis mean reward
        best = max(valid_results, key=lambda x: (
            x.get('success_rate', 0),  # Priorité 1: success rate
            x.get('mean_reward', -float('inf'))  # Priorité 2: mean reward
        ))
        
        best_by_env[env_name] = best
    
    return best_by_env

def main():
    """Teste tous les algorithmes et trouve le meilleur pour chaque environnement"""
    print("="*70)
    print("RECHERCHE DU MEILLEUR ALGORITHME PAR ENVIRONNEMENT")
    print("="*70)
    
    # Environnements à tester
    environments = ['LineWorldSimple', 'GridWorldSimple', 'TwoRoundRPS', 'MontyHallLevel1', 'MontyHallLevel2']
    
    # Algorithmes à tester
    algorithms = list(ALGORITHMS.keys())
    
    all_results = []
    results_by_env = {env: [] for env in environments}
    
    total_tests = len(environments) * len(algorithms)
    current_test = 0
    
    for env_name in environments:
        print(f"\n{'='*70}")
        print(f"ENVIRONNEMENT: {env_name}")
        print(f"{'='*70}")
        
        for algo_name in algorithms:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] {algo_name} sur {env_name}...", end=" ")
            
            # Skip Policy/Value Iteration sur envs trop complexes
            if algo_name in ['Policy Iteration', 'Value Iteration']:
                if env_name in ['TwoRoundRPS', 'MontyHallLevel1', 'MontyHallLevel2']:
                    print("SKIPPED (trop complexe pour DP)")
                    continue
            
            results = test_algorithm_on_environment(algo_name, env_name)
            all_results.append(results)
            results_by_env[env_name].append(results)
            
            if 'error' in results:
                print(f"ERREUR: {results['error']}")
            else:
                print(f"OK - Success: {results['success_rate']*100:.1f}%, Reward: {results['mean_reward']:.2f}")
            
            time.sleep(0.1)  # Pause courte
    
    # Trouver le meilleur algorithme pour chaque environnement
    best_by_env = find_best_algorithm(results_by_env)
    
    # Afficher les résultats
    print(f"\n\n{'='*70}")
    print("MEILLEUR ALGORITHME PAR ENVIRONNEMENT")
    print(f"{'='*70}\n")
    
    for env_name in environments:
        best = best_by_env.get(env_name)
        if best:
            print(f"{env_name}:")
            print(f"   [MEILLEUR] {best['algorithm']}")
            print(f"   Success Rate: {best['success_rate']*100:.1f}%")
            print(f"   Mean Reward: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
            print(f"   Training Time: {best['training_time']:.2f}s")
            print()
            
            # Afficher le top 3
            valid_results = [r for r in results_by_env[env_name] if 'error' not in r]
            valid_results.sort(key=lambda x: (x.get('success_rate', 0), x.get('mean_reward', -float('inf'))), reverse=True)
            
            if len(valid_results) > 1:
                print(f"   Top 3:")
                for i, r in enumerate(valid_results[:3], 1):
                    print(f"      {i}. {r['algorithm']}: {r['success_rate']*100:.1f}% success, {r['mean_reward']:.2f} reward")
            print()
        else:
            print(f"{env_name}: Aucun algorithme n'a fonctionne")
            print()
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/best_algo_per_env_{timestamp}.json"
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'timestamp': timestamp,
        'best_by_environment': {
            env: {
                'algorithm': best['algorithm'] if best else None,
                'success_rate': best['success_rate'] if best else None,
                'mean_reward': best['mean_reward'] if best else None,
                'training_time': best['training_time'] if best else None,
            } for env, best in best_by_env.items()
        },
        'all_results': all_results,
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"{'='*70}")
    print(f"OK - Resultats sauvegardes: {results_file}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

