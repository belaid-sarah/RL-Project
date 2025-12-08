"""
Script de test avec visualisation pygame et comparaison d'algorithmes
"""

import pygame
import sys
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
from algos.monte_carlo import OnPolicyMonteCarlo
from algos.dyna_q import DynaQAgent
import time

def visualize_training(env, agent, num_episodes=100, display_every=10):
    """Visualise l'entraînement avec pygame"""
    
    pygame.init()
    cell_size = 60
    screen_width = env.width * cell_size
    screen_height = env.height * cell_size + 150
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"Training: {agent.name}")
    
    font_small = pygame.font.Font(None, 20)
    font_medium = pygame.font.Font(None, 24)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 500:
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return episode_rewards
            
            # Action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            # Mise à jour de l'agent
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Afficher tous les X épisodes
            if episode % display_every == 0:
                screen.fill((20, 20, 25))
                
                # Dessiner la grille
                for x in range(env.width):
                    for y in range(env.height):
                        color = (80, 80, 90)
                        if (x, y) in env.obstacles:
                            color = (40, 40, 40)
                        elif (x, y) in env.traps:
                            color = (180, 30, 30)
                        elif (x, y) in env.rewards_cells:
                            color = (255, 200, 50)
                        elif (x, y) == env.goal:
                            color = (50, 200, 50)
                        elif (x, y) == env.start:
                            color = (50, 150, 255)
                        
                        # Agent
                        if (x, y) == state:
                            color = (100, 255, 100)
                        
                        pygame.draw.rect(screen, color,
                                       (x*cell_size + 2, y*cell_size + 2,
                                        cell_size-4, cell_size-4))
                
                # Infos
                info_y = env.height * cell_size + 10
                texts = [
                    f"Episode: {episode+1}/{num_episodes}",
                    f"Reward: {total_reward:.1f}",
                    f"Steps: {steps}",
                    f"Algorithm: {agent.name}",
                ]
                for i, text in enumerate(texts):
                    text_surface = font_medium.render(text, True, (255, 255, 255))
                    screen.blit(text_surface, (10, info_y + i*25))
                
                pygame.display.flip()
                pygame.time.wait(50)
        
        episode_rewards.append(total_reward)
    
    pygame.quit()
    return episode_rewards

def test_algorithm(env_class, algo_class, algo_name, hyperparams, num_episodes=500):
    """Teste un algorithme et retourne les résultats"""
    print(f"\n{'='*60}")
    print(f"Testing {algo_name}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"{'='*60}")
    
    env = env_class()
    agent = algo_class(env, **hyperparams)
    
    start_time = time.time()
    agent.train(num_episodes=num_episodes, verbose=False)
    training_time = time.time() - start_time
    
    # Évaluer
    eval_results = agent.evaluate(num_episodes=100, verbose=False)
    
    return {
        'algorithm': algo_name,
        'hyperparameters': hyperparams,
        'training_time': training_time,
        'num_episodes': num_episodes,
        'evaluation': eval_results,
        'agent': agent
    }

def compare_algorithms():
    """Compare plusieurs algorithmes avec différentes configurations"""
    
    # Configuration des tests
    env_class = GridWorld
    env_params = {'width': 5, 'height': 5}
    
    # Algorithmes à tester avec leurs hyperparamètres
    test_configs = [
        # Q-Learning avec différents alpha
        ('Q-Learning', QLearningAgent, {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1}),
        ('Q-Learning', QLearningAgent, {'alpha': 0.2, 'gamma': 0.99, 'epsilon': 0.1}),
        ('Q-Learning', QLearningAgent, {'alpha': 0.05, 'gamma': 0.99, 'epsilon': 0.1}),
        
        # SARSA
        ('SARSA', SARSAAgent, {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1}),
        ('SARSA', SARSAAgent, {'alpha': 0.2, 'gamma': 0.99, 'epsilon': 0.1}),
        
        # Monte Carlo
        ('Monte Carlo', OnPolicyMonteCarlo, {'gamma': 0.99, 'epsilon': 0.1}),
        
        # Dyna-Q
        ('Dyna-Q', DynaQAgent, {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1, 'n_planning_steps': 5}),
        ('Dyna-Q', DynaQAgent, {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1, 'n_planning_steps': 10}),
    ]
    
    results = []
    
    for algo_name, algo_class, hyperparams in test_configs:
        # Créer un nom unique avec les hyperparamètres
        if 'alpha' in hyperparams:
            unique_name = f"{algo_name} (α={hyperparams['alpha']})"
        elif 'n_planning_steps' in hyperparams:
            unique_name = f"{algo_name} (n={hyperparams['n_planning_steps']})"
        else:
            unique_name = algo_name
        
        result = test_algorithm(
            lambda: GridWorld(**env_params),
            algo_class,
            unique_name,
            hyperparams,
            num_episodes=500
        )
        results.append(result)
        
        print(f"✅ {unique_name}: Reward={result['evaluation']['mean_reward']:.2f}, "
              f"Success={result['evaluation']['success_rate']*100:.1f}%, "
              f"Time={result['training_time']:.2f}s")
    
    # Afficher le classement
    print(f"\n{'='*60}")
    print("CLASSEMENT DES ALGORITHMES")
    print(f"{'='*60}")
    
    sorted_results = sorted(results, key=lambda x: x['evaluation']['mean_reward'], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:2d}. {result['algorithm']:30s} | "
              f"Reward: {result['evaluation']['mean_reward']:7.2f} ± {result['evaluation']['std_reward']:5.2f} | "
              f"Success: {result['evaluation']['success_rate']*100:5.1f}% | "
              f"Time: {result['training_time']:6.2f}s")
    
    # Sauvegarder les résultats
    import json
    from pathlib import Path
    from datetime import datetime
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Préparer les données pour JSON
    save_data = []
    for result in results:
        save_data.append({
            'algorithm': result['algorithm'],
            'hyperparameters': result['hyperparameters'],
            'training_time': result['training_time'],
            'num_episodes': result['num_episodes'],
            'evaluation': result['evaluation']
        })
    
    filename = results_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n✅ Résultats sauvegardés dans: {filename}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true', help='Comparer tous les algorithmes')
    parser.add_argument('--visualize', action='store_true', help='Visualiser l\'entraînement')
    parser.add_argument('--algo', type=str, default='Q-Learning', help='Algorithme à visualiser')
    parser.add_argument('--episodes', type=int, default=500, help='Nombre d\'épisodes')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms()
    elif args.visualize:
        env = GridWorld(width=5, height=5)
        if args.algo == 'Q-Learning':
            agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
        elif args.algo == 'SARSA':
            agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
        else:
            agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
        
        print("Visualisation de l'entraînement...")
        print("Fermez la fenêtre pour arrêter")
        visualize_training(env, agent, num_episodes=args.episodes, display_every=1)
    else:
        print("Usage:")
        print("  python test_with_visualization.py --compare          # Comparer tous les algorithmes")
        print("  python test_with_visualization.py --visualize --algo Q-Learning --episodes 200")

