"""
Teste un algorithme avec logs détaillés des actions prises
Affiche les actions (gauche/droite pour LineWorld, directions pour GridWorld)
"""

from envs.lineworld_simple import LineWorldSimple
from envs.gridworld_simple import GridWorldSimple
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
import argparse

def get_action_symbol(env_name, action):
    """Convertit action en symbole lisible"""
    if env_name == 'LineWorldSimple':
        if action == 0:
            return "← (gauche)"
        elif action == 1:
            return "→ (droite)"
    elif env_name == 'GridWorldSimple':
        if action == 0:
            return "↑ (haut)"
        elif action == 1:
            return "↓ (bas)"
        elif action == 2:
            return "← (gauche)"
        elif action == 3:
            return "→ (droite)"
    return f"Action {action}"

def test_with_logs(algo_class, algo_name, env_name, env_factory, num_episodes=100, show_all_episodes=False):
    """Teste avec logs détaillés des actions"""
    
    print(f"\n{'='*70}")
    print(f"TEST: {algo_name} sur {env_name}")
    print(f"{'='*70}\n")
    
    # Créer environnement et agent
    env = env_factory()
    
    if algo_name == 'Q-Learning':
        agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    elif algo_name == 'SARSA':
        agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    else:
        print(f"Algorithme {algo_name} non supporte pour les logs")
        return
    
    # Entraînement avec logs
    print(f"[ENTRAINEMENT] {num_episodes} episodes...\n")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_actions = []
        episode_rewards = []
        
        while not done and steps < 1000:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            # Mettre à jour l'agent
            if algo_name == 'Q-Learning':
                agent.update(state, action, reward, next_state, done)
            elif algo_name == 'SARSA':
                if not done:
                    next_action = agent.select_action(next_state, training=True)
                    agent.update(state, action, reward, next_state, next_action, done)
                else:
                    agent.update(state, action, reward, next_state, 0, done)
            
            episode_actions.append(action)
            episode_rewards.append(reward)
            total_reward += reward
            steps += 1
            state = next_state
        
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)
        
        # Afficher les rewards pour certains épisodes
        if show_all_episodes or episode < 3 or episode % 20 == 0 or episode == num_episodes - 1:
            print(f"Episode {episode+1:3d} | Total Reward: {total_reward:6.1f} | Steps: {steps:3d}")
            print(f"  Rewards par step: ", end="")
            
            # Afficher les rewards (limiter à 50 pour lisibilité)
            display_rewards = episode_rewards[:50]
            for i, r in enumerate(display_rewards):
                print(f"{r:+5.1f}", end=" ")
                if (i + 1) % 10 == 0 and i < len(display_rewards) - 1:
                    print("\n                     ", end="")
            
            if len(episode_rewards) > 50:
                print(f"... ({len(episode_rewards) - 50} rewards de plus)")
            else:
                print()
            
            # Statistiques des rewards
            reward_sum = sum(episode_rewards)
            reward_avg = reward_sum / len(episode_rewards) if episode_rewards else 0
            positive_rewards = sum(1 for r in episode_rewards if r > 0)
            negative_rewards = sum(1 for r in episode_rewards if r < 0)
            zero_rewards = sum(1 for r in episode_rewards if r == 0)
            
            print(f"  Stats: Total={reward_sum:6.1f}, Moyenne={reward_avg:5.2f}, "
                  f"Positifs={positive_rewards}, Negatifs={negative_rewards}, Zeros={zero_rewards}")
            print()
    
    # Évaluation avec logs détaillés
    print(f"\n{'='*70}")
    print(f"[EVALUATION] 5 episodes (mode greedy, pas d'exploration)...")
    print(f"{'='*70}\n")
    
    for eval_ep in range(5):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_actions = []
        episode_rewards = []
        episode_states = [state]
        
        while not done and steps < 1000:
            action = agent.select_action(state, training=False)  # Mode greedy
            next_state, reward, done, _ = env.step(action)
            
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_states.append(next_state)
            total_reward += reward
            steps += 1
            state = next_state
        
        print(f"Eval Episode {eval_ep+1} | Total Reward: {total_reward:6.1f} | Steps: {steps:3d}")
        print(f"  Rewards par step: ", end="")
        
        # Afficher les rewards
        display_rewards = episode_rewards[:50]
        for i, r in enumerate(display_rewards):
            print(f"{r:+5.1f}", end=" ")
            if (i + 1) % 10 == 0 and i < len(display_rewards) - 1:
                print("\n                     ", end="")
        
        if len(episode_rewards) > 50:
            print(f"... ({len(episode_rewards) - 50} rewards de plus)")
        else:
            print()
        
        # Afficher aussi les positions pour LineWorld
        if env_name == 'LineWorldSimple':
            print(f"  Positions: ", end="")
            for i, s in enumerate(episode_states[:30]):
                print(f"{s}", end="")
                if i < len(episode_states) - 1 and i < 29:
                    print(" -> ", end="")
            if len(episode_states) > 30:
                print(f"... (total {len(episode_states)} positions)")
            else:
                print()
        
        print()
    
    # Résumé final
    print(f"{'='*70}")
    print(f"RESUME")
    print(f"{'='*70}")
    print(f"Episodes d'entrainement: {num_episodes}")
    print(f"Reward moyen (derniers 10): {sum(agent.episode_rewards[-10:]) / min(10, len(agent.episode_rewards)):.2f}")
    print(f"Steps moyen (derniers 10): {sum(agent.episode_lengths[-10:]) / min(10, len(agent.episode_lengths)):.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tester avec logs des actions')
    parser.add_argument('--algo', type=str, default='Q-Learning', 
                       choices=['Q-Learning', 'SARSA'],
                       help='Algorithme a tester')
    parser.add_argument('--env', type=str, default='LineWorldSimple',
                       choices=['LineWorldSimple', 'GridWorldSimple'],
                       help='Environnement a tester')
    parser.add_argument('--episodes', type=int, default=100, help='Nombre d\'episodes')
    parser.add_argument('--all', action='store_true', help='Afficher tous les episodes')
    
    args = parser.parse_args()
    
    # Créer l'environnement
    if args.env == 'LineWorldSimple':
        env_factory = lambda: LineWorldSimple(length=10)
    elif args.env == 'GridWorldSimple':
        env_factory = lambda: GridWorldSimple(width=5, height=5)
    
    # Créer l'algorithme
    if args.algo == 'Q-Learning':
        algo_class = QLearningAgent
    elif args.algo == 'SARSA':
        algo_class = SARSAAgent
    
    test_with_logs(algo_class, args.algo, args.env, env_factory, 
                   num_episodes=args.episodes, show_all_episodes=args.all)

