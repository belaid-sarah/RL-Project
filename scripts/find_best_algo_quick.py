"""
Version rapide pour trouver le meilleur algorithme par environnement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.lineworld_simple import LineWorldSimple
from envs.gridworld_simple import GridWorldSimple
from envs.rps import TwoRoundRPS
from envs.monty_hall_level1 import MontyHallLevel1

from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
from algos.expected_sarsa import ExpectedSARSAAgent
from algos.dyna_q import DynaQAgent
from algos.policy_iteration import PolicyIteration
from algos.value_iteration import ValueIteration

# Test rapide des algorithmes principaux
ALGORITHMS = {
    'Q-Learning': QLearningAgent,
    'SARSA': SARSAAgent,
    'Expected SARSA': ExpectedSARSAAgent,
    'Dyna-Q': DynaQAgent,
    'Policy Iteration': PolicyIteration,
    'Value Iteration': ValueIteration,
}

ENVIRONMENTS = {
    'LineWorldSimple': lambda: LineWorldSimple(length=15),
    'GridWorldSimple': lambda: GridWorldSimple(width=8, height=8),
    'TwoRoundRPS': lambda: TwoRoundRPS(),
    'MontyHallLevel1': lambda: MontyHallLevel1(),
}

HYPERPARAMS = {
    'Q-Learning': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05},
    'SARSA': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05},
    'Expected SARSA': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05},
    'Dyna-Q': {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 1.0, 'epsilon_decay': 0.995, 'epsilon_min': 0.05, 'n_planning_steps': 5},
    'Policy Iteration': {'gamma': 0.99, 'theta': 1e-5},
    'Value Iteration': {'gamma': 0.99, 'theta': 1e-5},
}

def test_algo_env(algo_name, env_name):
    try:
        env = ENVIRONMENTS[env_name]()
        algo_class = ALGORITHMS[algo_name]
        agent = algo_class(env, **HYPERPARAMS[algo_name])
        
        num_episodes = 1000 if env_name in ['TwoRoundRPS', 'MontyHallLevel1'] else 2000
        if algo_name in ['Policy Iteration', 'Value Iteration']:
            num_episodes = 50
        
        agent.train(num_episodes=num_episodes, verbose=False)
        results = agent.evaluate(num_episodes=50, verbose=False)
        
        return {
            'success_rate': results['success_rate'],
            'mean_reward': results['mean_reward'],
        }
    except Exception as e:
        return {'error': str(e)}

print("="*70)
print("RECHERCHE DU MEILLEUR ALGORITHME PAR ENVIRONNEMENT")
print("="*70)

all_results = {}

for env_name in ENVIRONMENTS.keys():
    print(f"\n{env_name}:")
    print("-"*70)
    env_results = {}
    
    for algo_name in ALGORITHMS.keys():
        if algo_name in ['Policy Iteration', 'Value Iteration'] and env_name in ['TwoRoundRPS', 'MontyHallLevel1']:
            continue
        
        print(f"  Testing {algo_name}...", end=" ")
        result = test_algo_env(algo_name, env_name)
        
        if 'error' in result:
            print(f"ERROR")
        else:
            print(f"Success: {result['success_rate']*100:.1f}%, Reward: {result['mean_reward']:.2f}")
            env_results[algo_name] = result
    
    all_results[env_name] = env_results

# Trouver le meilleur
print("\n" + "="*70)
print("MEILLEUR ALGORITHME PAR ENVIRONNEMENT")
print("="*70)

for env_name, results in all_results.items():
    if not results:
        continue
    
    best = max(results.items(), key=lambda x: (x[1].get('success_rate', 0), x[1].get('mean_reward', -float('inf'))))
    
    print(f"\n{env_name}:")
    print(f"  [MEILLEUR] {best[0]}")
    print(f"  Success Rate: {best[1]['success_rate']*100:.1f}%")
    print(f"  Mean Reward: {best[1]['mean_reward']:.2f}")
    
    # Top 3
    sorted_results = sorted(results.items(), key=lambda x: (x[1].get('success_rate', 0), x[1].get('mean_reward', -float('inf'))), reverse=True)
    if len(sorted_results) > 1:
        print(f"  Top 3:")
        for i, (algo, res) in enumerate(sorted_results[:3], 1):
            print(f"    {i}. {algo}: {res['success_rate']*100:.1f}% success, {res['mean_reward']:.2f} reward")

print("\n" + "="*70)

