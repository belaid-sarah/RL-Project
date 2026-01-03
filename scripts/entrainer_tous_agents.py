"""
Script pour entraîner et sauvegarder tous les agents importants
Exécutez ce script une fois pour préparer tous les agents pour la soutenance
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time

# Créer le dossier models
Path('models').mkdir(exist_ok=True)

print("="*70)
print("ENTRAÎNEMENT DE TOUS LES AGENTS")
print("="*70)
print("\nCe script va entraîner et sauvegarder tous les agents importants.")
print("Cela peut prendre du temps (10-30 minutes selon votre machine).")
print("\nVous pouvez arrêter avec Ctrl+C à tout moment.")
print("="*70)

agents_to_train = []

# 1. Q-Learning sur LineWorldSimple
print("\n" + "="*70)
print("1. Q-Learning sur LineWorldSimple")
print("="*70)
try:
    from envs.lineworld_simple import LineWorldSimple
    from algos.q_learning import QLearningAgent
    
    env = LineWorldSimple(length=25)
    # Epsilon decay: commence à 1.0, décroît vers 0.05
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05)
    
    print("Entraînement en cours (1000 épisodes)...")
    agent.train(num_episodes=1000, verbose=False)
    
    model_path = 'models/qlearning_lineworld.pkl'
    agent.save(model_path)
    
    # Évaluer
    eval_results = agent.evaluate(num_episodes=50, verbose=False)
    print(f"[OK] Sauvegarde: {model_path}")
    print(f"   Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Success: {eval_results['success_rate']*100:.1f}%")
    
    agents_to_train.append(('Q-Learning', 'LineWorldSimple', model_path))
except Exception as e:
    print(f"[ERROR] Erreur: {e}")

# 2. Q-Learning sur GridWorldSimple
print("\n" + "="*70)
print("2. Q-Learning sur GridWorldSimple")
print("="*70)
try:
    from envs.gridworld_simple import GridWorldSimple
    
    env = GridWorldSimple(width=10, height=10)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05)
    
    print("Entraînement en cours (5000 épisodes - GridWorld nécessite plus d'épisodes)...")
    agent.train(num_episodes=5000, verbose=False)
    
    model_path = 'models/qlearning_gridworld.pkl'
    agent.save(model_path)
    
    eval_results = agent.evaluate(num_episodes=50, verbose=False)
    print(f"[OK] Sauvegarde: {model_path}")
    print(f"   Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Success: {eval_results['success_rate']*100:.1f}%")
    
    agents_to_train.append(('Q-Learning', 'GridWorldSimple', model_path))
except Exception as e:
    print(f"[ERROR] Erreur: {e}")

# 3. SARSA sur LineWorldSimple
print("\n" + "="*70)
print("3. SARSA sur LineWorldSimple")
print("="*70)
try:
    from algos.sarsa import SARSAAgent
    
    env = LineWorldSimple(length=25)
    agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05)
    
    print("Entraînement en cours (1000 épisodes)...")
    agent.train(num_episodes=1000, verbose=False)
    
    model_path = 'models/sarsa_lineworld.pkl'
    agent.save(model_path)
    
    eval_results = agent.evaluate(num_episodes=50, verbose=False)
    print(f"[OK] Sauvegarde: {model_path}")
    print(f"   Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Success: {eval_results['success_rate']*100:.1f}%")
    
    agents_to_train.append(('SARSA', 'LineWorldSimple', model_path))
except Exception as e:
    print(f"[ERROR] Erreur: {e}")

# 4. Dyna-Q sur GridWorldSimple
print("\n" + "="*70)
print("4. Dyna-Q sur GridWorldSimple")
print("="*70)
try:
    from algos.dyna_q import DynaQAgent
    
    env = GridWorldSimple(width=10, height=10)
    agent = DynaQAgent(env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, n_planning_steps=5)
    
    print("Entraînement en cours (3000 épisodes - GridWorld nécessite plus d'épisodes)...")
    agent.train(num_episodes=3000, verbose=False)
    
    model_path = 'models/dynaq_gridworld.pkl'
    agent.save(model_path)
    
    eval_results = agent.evaluate(num_episodes=50, verbose=False)
    print(f"[OK] Sauvegarde: {model_path}")
    print(f"   Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Success: {eval_results['success_rate']*100:.1f}%")
    
    agents_to_train.append(('Dyna-Q', 'GridWorldSimple', model_path))
except Exception as e:
    print(f"[ERROR] Erreur: {e}")

# 5. Policy Iteration sur LineWorldSimple
print("\n" + "="*70)
print("5. Policy Iteration sur LineWorldSimple")
print("="*70)
try:
    from algos.policy_iteration import PolicyIteration
    
    env = LineWorldSimple(length=25)
    agent = PolicyIteration(env, gamma=0.99, theta=1e-5)
    
    print("Entraînement en cours (convergence rapide)...")
    agent.train(num_episodes=50, verbose=False)
    
    model_path = 'models/policy_iteration_lineworld.pkl'
    agent.save(model_path)
    
    eval_results = agent.evaluate(num_episodes=50, verbose=False)
    print(f"[OK] Sauvegarde: {model_path}")
    print(f"   Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Success: {eval_results['success_rate']*100:.1f}%")
    
    agents_to_train.append(('Policy Iteration', 'LineWorldSimple', model_path))
except Exception as e:
    print(f"[ERROR] Erreur: {e}")

# 6. Value Iteration sur GridWorldSimple
print("\n" + "="*70)
print("6. Value Iteration sur GridWorldSimple")
print("="*70)
try:
    from algos.value_iteration import ValueIteration
    
    env = GridWorldSimple(width=10, height=10)
    agent = ValueIteration(env, gamma=0.99, theta=1e-5)
    
    print("Entraînement en cours (1000 épisodes - GridWorld nécessite plus d'épisodes)...")
    agent.train(num_episodes=1000, verbose=False)
    
    model_path = 'models/value_iteration_gridworld.pkl'
    agent.save(model_path)
    
    eval_results = agent.evaluate(num_episodes=50, verbose=False)
    print(f"[OK] Sauvegarde: {model_path}")
    print(f"   Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Success: {eval_results['success_rate']*100:.1f}%")
    
    agents_to_train.append(('Value Iteration', 'GridWorldSimple', model_path))
except Exception as e:
    print(f"[ERROR] Erreur: {e}")

# Résumé
print("\n" + "="*70)
print("RÉSUMÉ - AGENTS SAUVEGARDÉS")
print("="*70)

if agents_to_train:
    print(f"\n[OK] {len(agents_to_train)} agents sauvegardes avec succes:\n")
    for algo, env, path in agents_to_train:
        print(f"  • {algo:20s} sur {env:20s} → {path}")
    
    print("\n" + "="*70)
    print("UTILISATION POUR LA SOUTENANCE")
    print("="*70)
    print("\nPour rejouer un agent pas à pas:")
    print("  python replay_policy.py --env LineWorldSimple --algo Q-Learning --model models/qlearning_lineworld.pkl")
    print("\nPour visualiser un agent:")
    print("  python visualize_rl.py --env LineWorldSimple --algo Q-Learning --mode eval --episodes 100")
    print("\n[OK] Tous les agents sont prets pour la soutenance!")
else:
    print("\n[ERROR] Aucun agent n'a pu etre sauvegarde. Verifiez les erreurs ci-dessus.")

print("="*70)

