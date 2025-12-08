"""
Script de test complet pour tester tous les algorithmes sur tous les environnements

Ce script permet de :
1. Tester chaque algorithme sur chaque environnement
2. Comparer les performances
3. Sauvegarder les résultats
4. Générer des rapports
"""

import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

# Importer tous les environnements
from envs.lineworld import LineWorld
from envs.gridworld import GridWorld
from envs.rps import TwoRoundRPS
from envs.monty_hall_level1 import MontyHallLevel1
from envs.monty_hall_level2 import MontyHallLevel2

# Importer tous les algorithmes
from algos.policy_iteration import PolicyIteration
from algos.value_iteration import ValueIteration
from algos.monte_carlo import MonteCarloES, OnPolicyMonteCarlo, OffPolicyMonteCarlo
from algos.sarsa import SARSAAgent
from algos.q_learning import QLearningAgent
from algos.expected_sarsa import ExpectedSARSAAgent
from algos.dyna_q import DynaQAgent
from algos.dyna_q_plus import DynaQPlusAgent

# Configuration des tests
CONFIG = {
    'num_episodes': {
        'LineWorld': 1000,
        'GridWorld': 2000,
        'TwoRoundRPS': 500,
        'MontyHallLevel1': 500,
        'MontyHallLevel2': 1000,
    },
    'evaluation_episodes': 100,
    'verbose': False,  # Mettre à True pour voir les détails
    'save_results': True,
    'results_dir': 'results'
}

# Hyperparamètres par algorithme et environnement
HYPERPARAMETERS = {
    'PolicyIteration': {
        'gamma': 0.99,
        'theta': 1e-5,
    },
    'ValueIteration': {
        'gamma': 0.99,
        'theta': 1e-5,
    },
    'MonteCarloES': {
        'gamma': 0.99,
        'epsilon': 0.1,
    },
    'OnPolicyMonteCarlo': {
        'gamma': 0.99,
        'epsilon': 0.1,
    },
    'OffPolicyMonteCarlo': {
        'gamma': 0.99,
        'epsilon': 0.1,
    },
    'SARSA': {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
    },
    'Q-Learning': {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
    },
    'ExpectedSARSA': {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
    },
    'Dyna-Q': {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
        'n_planning_steps': 5,
    },
    'Dyna-Q+': {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon': 0.1,
        'n_planning_steps': 5,
        'kappa': 1e-3,
        'tau': 1000,
    },
}

# Mapping des algorithmes
ALGORITHMS = {
    'PolicyIteration': PolicyIteration,
    'ValueIteration': ValueIteration,
    'MonteCarloES': MonteCarloES,
    'OnPolicyMonteCarlo': OnPolicyMonteCarlo,
    'OffPolicyMonteCarlo': OffPolicyMonteCarlo,
    'SARSA': SARSAAgent,
    'Q-Learning': QLearningAgent,
    'ExpectedSARSA': ExpectedSARSAAgent,
    'Dyna-Q': DynaQAgent,
    'Dyna-Q+': DynaQPlusAgent,
}

# Mapping des environnements
ENVIRONMENTS = {
    'LineWorld': lambda: LineWorld(length=10),  # Réduire pour les tests rapides
    'GridWorld': lambda: GridWorld(width=5, height=5),  # Réduire pour les tests rapides
    'TwoRoundRPS': lambda: TwoRoundRPS(),
    'MontyHallLevel1': lambda: MontyHallLevel1(),
    'MontyHallLevel2': lambda: MontyHallLevel2(),
}

def test_algorithm_on_environment(algo_name, env_name, num_episodes, verbose=False):
    """
    Teste un algorithme sur un environnement
    
    Returns:
        dict: Résultats du test
    """
    print(f"\n{'='*60}")
    print(f"Testing {algo_name} on {env_name}")
    print(f"{'='*60}")
    
    try:
        # Créer l'environnement
        env = ENVIRONMENTS[env_name]()
        
        # Créer l'agent
        hyperparams = HYPERPARAMETERS[algo_name].copy()
        agent = ALGORITHMS[algo_name](env, **hyperparams)
        
        # Entraîner
        print(f"Training for {num_episodes} episodes...")
        start_time = time.time()
        agent.train(num_episodes=num_episodes, verbose=verbose)
        training_time = time.time() - start_time
        
        # Évaluer
        print(f"Evaluating for {CONFIG['evaluation_episodes']} episodes...")
        eval_results = agent.evaluate(num_episodes=CONFIG['evaluation_episodes'], verbose=verbose)
        
        # Statistiques d'entraînement
        training_stats = agent.get_training_stats()
        
        results = {
            'algorithm': algo_name,
            'environment': env_name,
            'hyperparameters': hyperparams,
            'training': {
                'num_episodes': num_episodes,
                'training_time': training_time,
                'convergence_episode': agent.convergence_episode,
                'final_mean_reward': training_stats['final_mean_reward'] if training_stats else None,
                'best_reward': training_stats['best_reward'] if training_stats else None,
            },
            'evaluation': eval_results,
            'success': True,
            'error': None
        }
        
        print(f"✅ Success!")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Eval mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"   Success rate: {eval_results['success_rate']*100:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'algorithm': algo_name,
            'environment': env_name,
            'success': False,
            'error': str(e)
        }

def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*60)
    print("REINFORCEMENT LEARNING - COMPLETE TEST SUITE")
    print("="*60)
    
    all_results = []
    total_tests = len(ALGORITHMS) * len(ENVIRONMENTS)
    current_test = 0
    
    # Créer le dossier de résultats
    if CONFIG['save_results']:
        results_dir = Path(CONFIG['results_dir'])
        results_dir.mkdir(exist_ok=True)
    
    for algo_name in ALGORITHMS.keys():
        for env_name in ENVIRONMENTS.keys():
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] {algo_name} on {env_name}")
            
            num_episodes = CONFIG['num_episodes'].get(env_name, 1000)
            result = test_algorithm_on_environment(
                algo_name, 
                env_name, 
                num_episodes, 
                verbose=CONFIG['verbose']
            )
            
            all_results.append(result)
            
            # Sauvegarder individuellement
            if CONFIG['save_results'] and result['success']:
                filename = f"{algo_name}_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = results_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
    
    # Sauvegarder le rapport complet
    if CONFIG['save_results']:
        report_path = results_dir / f"complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n✅ Complete report saved to {report_path}")
    
    # Générer un résumé
    print_summary(all_results)
    
    return all_results

def print_summary(results):
    """Affiche un résumé des résultats"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Grouper par environnement
    by_env = {}
    for result in results:
        if result['success']:
            env = result['environment']
            if env not in by_env:
                by_env[env] = []
            by_env[env].append(result)
    
    for env_name, env_results in by_env.items():
        print(f"\n{env_name}:")
        print("-" * 60)
        
        # Trier par performance
        sorted_results = sorted(
            env_results, 
            key=lambda x: x['evaluation']['mean_reward'], 
            reverse=True
        )
        
        for i, result in enumerate(sorted_results, 1):
            algo = result['algorithm']
            eval_res = result['evaluation']
            print(f"{i:2d}. {algo:20s} | "
                  f"Reward: {eval_res['mean_reward']:7.2f} ± {eval_res['std_reward']:5.2f} | "
                  f"Success: {eval_res['success_rate']*100:5.1f}% | "
                  f"Time: {result['training']['training_time']:6.2f}s")
    
    # Statistiques globales
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed tests:")
        for result in failed:
            print(f"  - {result['algorithm']} on {result['environment']}: {result['error']}")

def test_single_combination(algo_name, env_name, num_episodes=1000, verbose=True):
    """Teste une seule combinaison algorithme/environnement"""
    return test_algorithm_on_environment(algo_name, env_name, num_episodes, verbose)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RL algorithms on environments')
    parser.add_argument('--algo', type=str, help='Algorithm name to test')
    parser.add_argument('--env', type=str, help='Environment name to test')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.all:
        CONFIG['verbose'] = args.verbose
        run_all_tests()
    elif args.algo and args.env:
        CONFIG['verbose'] = args.verbose
        test_single_combination(args.algo, args.env, args.episodes, args.verbose)
    else:
        print("Usage:")
        print("  python test_all_algos_envs.py --all                    # Run all tests")
        print("  python test_all_algos_envs.py --algo Q-Learning --env LineWorld --episodes 1000")
        print("\nAvailable algorithms:", list(ALGORITHMS.keys()))
        print("Available environments:", list(ENVIRONMENTS.keys()))



