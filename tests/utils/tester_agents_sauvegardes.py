"""
Script pour tester si les agents sauvegardés ont vraiment appris
Vérifie si les agents atteignent le goal et combien de steps ils prennent
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from envs.lineworld_simple import LineWorldSimple
from envs.gridworld_simple import GridWorldSimple
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
from algos.dyna_q import DynaQAgent
from algos.policy_iteration import PolicyIteration
from algos.value_iteration import ValueIteration

def test_agent(env, agent_class, model_path, env_name, algo_name, num_tests=10):
    """Teste un agent sauvegardé"""
    print(f"\n{'='*70}")
    print(f"Test: {algo_name} sur {env_name}")
    print(f"{'='*70}")
    
    try:
        # Créer l'agent
        if algo_name == "Q-Learning":
            agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
        elif algo_name == "SARSA":
            agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
        elif algo_name == "Dyna-Q":
            agent = DynaQAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5)
        elif algo_name == "Policy Iteration":
            agent = PolicyIteration(env, gamma=0.99, theta=1e-5)
        elif algo_name == "Value Iteration":
            agent = ValueIteration(env, gamma=0.99, theta=1e-5)
        else:
            print(f"[ERROR] Algorithme non reconnu: {algo_name}")
            return
        
        # Charger le modèle
        agent.load(model_path)
        
        # Tester
        success_count = 0
        total_steps = []
        total_rewards = []
        
        for test_num in range(num_tests):
            state = env.reset()
            steps = 0
            reward_total = 0
            done = False
            max_steps = 200  # Augmenté pour permettre plus de steps
            
            # Trajectoire
            trajectory = [state]
            
            while not done and steps < max_steps:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                trajectory.append(next_state)
                reward_total += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Vérifier si goal atteint
            if hasattr(env, 'goal'):
                goal_reached = (state == env.goal) or done
            else:
                goal_reached = done and reward_total > -50  # Heuristique
            
            if goal_reached:
                success_count += 1
            
            total_steps.append(steps)
            total_rewards.append(reward_total)
            
            # Afficher le premier test en détail
            if test_num == 0:
                print(f"\n  Test 1 - Trajectoire (premiers 10 steps):")
                print(f"    {trajectory[:10]}...")
                print(f"    Steps: {steps}, Reward: {reward_total:.2f}, Goal atteint: {goal_reached}")
        
        # Statistiques
        success_rate = success_count / num_tests
        avg_steps = sum(total_steps) / len(total_steps)
        avg_reward = sum(total_rewards) / len(total_rewards)
        
        print(f"\n  Resultats sur {num_tests} tests:")
        print(f"    [OK] Taux de succes: {success_rate*100:.1f}% ({success_count}/{num_tests})")
        print(f"    Steps moyens: {avg_steps:.1f}")
        print(f"    Reward moyen: {avg_reward:.2f}")
        
        if success_rate > 0.5:
            print(f"  [OK] Agent fonctionne bien!")
        elif success_rate > 0:
            print(f"  [WARN] Agent apprend mais pas optimal")
        else:
            print(f"  [ERROR] Agent n'a pas appris - besoin de plus d'entrainement")
        
        return success_rate, avg_steps, avg_reward
        
    except Exception as e:
        print(f"  [ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Tests
print("="*70)
print("TEST DES AGENTS SAUVEGARDÉS")
print("="*70)
print("\nCe script teste si vos agents sauvegardés ont vraiment appris.")
print("Il vérifie si les agents atteignent le goal et combien de steps ils prennent.")
print("="*70)

results = []

# 1. Q-Learning LineWorldSimple
try:
    env = LineWorldSimple(length=25)
    success, steps, reward = test_agent(
        env, QLearningAgent, 
        'models/qlearning_lineworld.pkl',
        'LineWorldSimple', 'Q-Learning', num_tests=20
    )
    if success is not None:
        results.append(('Q-Learning', 'LineWorldSimple', success, steps, reward))
except Exception as e:
    print(f"[ERROR] Erreur avec Q-Learning LineWorld: {e}")

# 2. Q-Learning GridWorldSimple
try:
    env = GridWorldSimple(width=10, height=10)
    success, steps, reward = test_agent(
        env, QLearningAgent,
        'models/qlearning_gridworld.pkl',
        'GridWorldSimple', 'Q-Learning', num_tests=20
    )
    if success is not None:
        results.append(('Q-Learning', 'GridWorldSimple', success, steps, reward))
except Exception as e:
    print(f"[ERROR] Erreur avec Q-Learning GridWorld: {e}")

# 3. SARSA LineWorldSimple
try:
    env = LineWorldSimple(length=25)
    success, steps, reward = test_agent(
        env, SARSAAgent,
        'models/sarsa_lineworld.pkl',
        'LineWorldSimple', 'SARSA', num_tests=20
    )
    if success is not None:
        results.append(('SARSA', 'LineWorldSimple', success, steps, reward))
except Exception as e:
    print(f"[ERROR] Erreur avec SARSA: {e}")

# 4. Dyna-Q GridWorldSimple
try:
    env = GridWorldSimple(width=10, height=10)
    success, steps, reward = test_agent(
        env, DynaQAgent,
        'models/dynaq_gridworld.pkl',
        'GridWorldSimple', 'Dyna-Q', num_tests=20
    )
    if success is not None:
        results.append(('Dyna-Q', 'GridWorldSimple', success, steps, reward))
except Exception as e:
    print(f"[ERROR] Erreur avec Dyna-Q: {e}")

# 5. Policy Iteration LineWorldSimple
try:
    env = LineWorldSimple(length=25)
    success, steps, reward = test_agent(
        env, PolicyIteration,
        'models/policy_iteration_lineworld.pkl',
        'LineWorldSimple', 'Policy Iteration', num_tests=20
    )
    if success is not None:
        results.append(('Policy Iteration', 'LineWorldSimple', success, steps, reward))
except Exception as e:
    print(f"[ERROR] Erreur avec Policy Iteration: {e}")

# 6. Value Iteration GridWorldSimple
try:
    env = GridWorldSimple(width=10, height=10)
    success, steps, reward = test_agent(
        env, ValueIteration,
        'models/value_iteration_gridworld.pkl',
        'GridWorldSimple', 'Value Iteration', num_tests=20
    )
    if success is not None:
        results.append(('Value Iteration', 'GridWorldSimple', success, steps, reward))
except Exception as e:
    print(f"[ERROR] Erreur avec Value Iteration: {e}")

# Résumé
print("\n" + "="*70)
print("RÉSUMÉ DES TESTS")
print("="*70)

if results:
    print(f"\n{'Algorithme':<20} {'Environnement':<20} {'Success':<10} {'Steps':<10} {'Reward':<10}")
    print("-"*70)
    for algo, env_name, success, steps, reward in results:
        print(f"{algo:<20} {env_name:<20} {success*100:>6.1f}%   {steps:>6.1f}    {reward:>8.2f}")
    
    # Meilleur agent
    best = max(results, key=lambda x: x[2])  # Par success rate
    print(f"\n[BEST] Meilleur agent: {best[0]} sur {best[1]} ({best[2]*100:.1f}% success)")
else:
    print("\n[ERROR] Aucun resultat disponible")

print("="*70)

