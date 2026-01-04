"""
Teste UN algorithme sur TOUS les environnements
Parfait pour commencer et voir les différences entre environnements
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.lineworld import LineWorld
from envs.lineworld_simple import LineWorldSimple
from envs.gridworld import GridWorld
from envs.gridworld_simple import GridWorldSimple
from envs.rps import TwoRoundRPS
from envs.monty_hall_level1 import MontyHallLevel1
from envs.monty_hall_level2 import MontyHallLevel2
from algos.q_learning import QLearningAgent
import json
from datetime import datetime

def test_one_algo_all_envs(algo_class, algo_name, hyperparams, num_episodes=1000):
    """Teste un algorithme sur tous les environnements"""
    
    # Tous les environnements
    all_environments = {
        'LineWorldSimple': lambda: LineWorldSimple(length=20),  # Version simple, peut être agrandie
        'GridWorldSimple': lambda: GridWorldSimple(width=8, height=8),  # Version simple, peut être agrandie
        'TwoRoundRPS': lambda: TwoRoundRPS(),
        'MontyHallLevel1': lambda: MontyHallLevel1(),
        'MontyHallLevel2': lambda: MontyHallLevel2(),
    }
    
    # Pour Policy Iteration et Value Iteration, utiliser les versions simples
    if algo_name in ['PolicyIteration', 'ValueIteration']:
        environments = {
            'LineWorldSimple': lambda: LineWorldSimple(length=20),
            'GridWorldSimple': lambda: GridWorldSimple(width=8, height=8),
            'TwoRoundRPS': all_environments['TwoRoundRPS'],
        }
        print("[NOTE] Utilisation des environnements simplifies (rewards standards Sutton & Barto)")
    else:
        environments = all_environments
    
    # Nombre d'épisodes par environnement (certains sont plus simples)
    episodes_by_env = {
        'LineWorld': 200,  # Réduit car LineWorld est complexe
        'GridWorld': 1000,  # Réduit pour accélérer
        'TwoRoundRPS': 500,
        'MontyHallLevel1': 500,
        'MontyHallLevel2': 500,  # Réduit
    }
    
    # Pour Policy/Value Iteration, utiliser moins d'itérations
    if algo_name in ['PolicyIteration', 'ValueIteration']:
        episodes_by_env = {
            'LineWorldSimple': 20,  # Itérations de convergence
            'GridWorldSimple': 30,  # Itérations de convergence
            'TwoRoundRPS': 20,
        }
    
    # Limite de steps par épisode (pour éviter les boucles infinies)
    max_steps_by_env = {
        'LineWorld': 200,
        'GridWorld': 100,
        'TwoRoundRPS': 10,
        'MontyHallLevel1': 10,
        'MontyHallLevel2': 20,
    }
    
    print(f"\n{'='*70}")
    print(f"TEST: {algo_name} sur TOUS les environnements")
    print(f"Hyperparamètres: {hyperparams}")
    print(f"{'='*70}\n")
    
    results = []
    
    for env_name, env_factory in environments.items():
        print(f"\n{'-'*70}")
        print(f"Environnement: {env_name}")
        print(f"{'-'*70}")
        
        try:
            # Créer l'environnement
            env = env_factory()
            
            # Créer l'agent
            agent = algo_class(env, **hyperparams)
            
            # Entraîner
            num_eps = episodes_by_env.get(env_name, num_episodes)
            max_steps = max_steps_by_env.get(env_name, 1000)
            
            # Policy Iteration et Value Iteration utilisent num_iterations au lieu de num_episodes
            if algo_name in ['PolicyIteration', 'ValueIteration']:
                print(f"[ENTRAINEMENT] {num_eps} iterations...")
                agent.train(num_episodes=num_eps, verbose=False)
            else:
                print(f"[ENTRAINEMENT] {num_eps} episodes (max {max_steps} steps/episode)...")
                agent.train(num_episodes=num_eps, verbose=False, max_steps_per_episode=max_steps)
            
            # Évaluer
            print(f"[EVALUATION] 100 episodes...")
            eval_results = agent.evaluate(num_episodes=100, verbose=False)
            
            # Afficher les résultats
            print(f"[OK] Resultats:")
            print(f"   Récompense moyenne: {eval_results['mean_reward']:7.2f} ± {eval_results['std_reward']:5.2f}")
            print(f"   Taux de succès:      {eval_results['success_rate']*100:6.1f}%")
            print(f"   Steps moyens:        {eval_results['mean_steps']:6.1f}")
            print(f"   Temps d'entraînement: {agent.training_time:.2f}s")
            
            results.append({
                'algorithm': algo_name,
                'environment': env_name,
                'hyperparameters': hyperparams,
                'num_episodes': num_eps,
                'training_time': agent.training_time,
                'evaluation': eval_results
            })
            
        except Exception as e:
            print(f"[ERREUR] {str(e)}")
            results.append({
                'algorithm': algo_name,
                'environment': env_name,
                'error': str(e)
            })
    
    # Afficher le résumé
    print(f"\n\n{'='*70}")
    print("RESUME - Performance par environnement")
    print(f"{'='*70}")
    print(f"{'Environnement':<20} | {'Reward Moy':>12} | {'Success %':>10} | {'Temps (s)':>10}")
    print(f"{'-'*70}")
    
    successful = [r for r in results if 'evaluation' in r]
    for result in successful:
        eval_res = result['evaluation']
        print(f"{result['environment']:<20} | "
              f"{eval_res['mean_reward']:>12.2f} | "
              f"{eval_res['success_rate']*100:>9.1f}% | "
              f"{result['training_time']:>10.2f}")
    
    # Meilleur et pire environnement
    if successful:
        best = max(successful, key=lambda x: x['evaluation']['mean_reward'])
        worst = min(successful, key=lambda x: x['evaluation']['mean_reward'])
        
        print(f"\n[MEILLEUR] Environnement: {best['environment']}")
        print(f"   Reward: {best['evaluation']['mean_reward']:.2f}")
        
        print(f"\n[PLUS DIFFICILE] Environnement: {worst['environment']}")
        print(f"   Reward: {worst['evaluation']['mean_reward']:.2f}")
    
    # Sauvegarder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    filename = results_dir / f"{algo_name}_all_envs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[OK] Resultats sauvegardes: {filename}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tester un algorithme sur tous les environnements')
    parser.add_argument('--algo', type=str, default='Q-Learning', 
                       choices=['Q-Learning', 'SARSA', 'MonteCarlo', 'Dyna-Q', 'PolicyIteration', 'ValueIteration'],
                       help='Algorithme à tester')
    parser.add_argument('--alpha', type=float, default=0.1, help='Taux d\'apprentissage')
    parser.add_argument('--gamma', type=float, default=0.99, help='Facteur d\'actualisation')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration')
    parser.add_argument('--episodes', type=int, default=1000, help='Nombre d\'épisodes par défaut')
    
    args = parser.parse_args()
    
    # Configuration selon l'algorithme
    if args.algo == 'Q-Learning':
        from algos.q_learning import QLearningAgent
        algo_class = QLearningAgent
        hyperparams = {'alpha': args.alpha, 'gamma': args.gamma, 'epsilon': args.epsilon}
    elif args.algo == 'SARSA':
        from algos.sarsa import SARSAAgent
        algo_class = SARSAAgent
        hyperparams = {'alpha': args.alpha, 'gamma': args.gamma, 'epsilon': args.epsilon}
    elif args.algo == 'MonteCarlo':
        from algos.monte_carlo import OnPolicyMonteCarlo
        algo_class = OnPolicyMonteCarlo
        hyperparams = {'gamma': args.gamma, 'epsilon': args.epsilon}
    elif args.algo == 'Dyna-Q':
        from algos.dyna_q import DynaQAgent
        algo_class = DynaQAgent
        hyperparams = {'alpha': args.alpha, 'gamma': args.gamma, 'epsilon': args.epsilon, 'n_planning_steps': 5}
    elif args.algo == 'PolicyIteration':
        from algos.policy_iteration import PolicyIteration
        algo_class = PolicyIteration
        hyperparams = {'gamma': args.gamma, 'theta': 1e-5}
    elif args.algo == 'ValueIteration':
        from algos.value_iteration import ValueIteration
        algo_class = ValueIteration
        hyperparams = {'gamma': args.gamma, 'theta': 1e-5}
    else:
        print(f"Algorithme {args.algo} non supporté")
        exit(1)
    
    test_one_algo_all_envs(algo_class, args.algo, hyperparams, args.episodes)

