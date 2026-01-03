"""
Visualisation interactive des environnements RL avec Pygame
Permet de voir l'agent entraîné en action
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
import argparse
import time

from envs.lineworld_simple import LineWorldSimple
from envs.gridworld_simple import GridWorldSimple
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent

# Couleurs
COLORS = {
    'background': (30, 30, 40),
    'cell': (60, 60, 80),
    'agent': (100, 255, 100),
    'goal': (50, 200, 50),
    'obstacle': (40, 40, 40),
    'start': (100, 150, 255),
    'text': (255, 255, 255),
    'reward_pos': (255, 200, 50),
    'reward_neg': (200, 50, 50),
}

def visualize_lineworld(env, agent, mode='eval', speed=2, algo_name='Q-Learning'):
    """Visualise LineWorld avec Pygame"""
    pygame.init()
    
    cell_size = 50
    width = env.length * cell_size
    height = cell_size + 150  # +150 pour les infos
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"LineWorld - {agent.name} - Mode: {mode}")
    
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    step_delay = 1000 // speed  # ms entre chaque step
    
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    last_step_time = time.time() * 1000
    max_steps_per_episode = env.max_steps if hasattr(env, 'max_steps') else 1000
    last_states = []  # Pour détecter les oscillations
    
    while running:
        current_time = time.time() * 1000
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    episode_num += 1
                elif event.key == pygame.K_UP:
                    speed = min(20, speed + 1)
                    step_delay = 1000 // speed
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)
                    step_delay = 1000 // speed
                elif event.key == pygame.K_q:
                    running = False
        
        # Step automatique
        if not paused and not env.done and (current_time - last_step_time) >= step_delay:
            action = agent.select_action(state, training=(mode == 'train'))
            next_state, reward, done, info = env.step(action)
            
            if mode == 'train':
                if algo_name == 'SARSA':
                    if not done:
                        next_action = agent.select_action(next_state, training=True)
                        agent.update(state, action, reward, next_state, next_action, done)
                    else:
                        agent.update(state, action, reward, next_state, 0, done)
                else:
                    agent.update(state, action, reward, next_state, done)
            
            # Détecter les oscillations
            if len(last_states) >= 2 and last_states[-1] == state and last_states[-2] == next_state:
                if episode_steps >= 10:  # Seuil pour éviter reset trop rapide
                    done = True
            
            # Mettre à jour l'historique des états
            last_states.append(state)
            if len(last_states) > 5:
                last_states.pop(0)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            last_step_time = current_time
            
            # Reset si terminé ou trop de steps
            if done or episode_steps >= max_steps_per_episode:
                if episode_steps >= max_steps_per_episode:
                    print(f"Limite de steps atteinte ({max_steps_per_episode}). Reset...")
                pygame.time.wait(1000)  # Pause avant reset
                state = env.reset()
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
                last_states = []
        
        # Dessiner
        screen.fill(COLORS['background'])
        
        # Dessiner la ligne
        for pos in range(env.length):
            x = pos * cell_size
            color = COLORS['cell']
            
            # Types de cellules
            if pos == env.start:
                color = COLORS['start']
            elif pos == env.goal:
                color = COLORS['goal']
            elif pos in env.obstacles:
                color = COLORS['obstacle']
            elif pos in getattr(env, 'rewards_cells', {}):
                color = COLORS['reward_pos']
            elif pos in getattr(env, 'traps', set()):
                color = COLORS['reward_neg']
            elif pos in getattr(env, 'bonus_zones', {}):
                color = (150, 200, 255)  # Bleu clair pour bonus
            
            # Agent
            if pos == state:
                color = COLORS['agent']
            
            pygame.draw.rect(screen, color, (x + 2, 2, cell_size - 4, cell_size - 4))
            
            # Numéro de position
            text = font_small.render(str(pos), True, COLORS['text'])
            text_rect = text.get_rect(center=(x + cell_size // 2, cell_size // 2))
            screen.blit(text, text_rect)
        
        # Infos en bas
        y_offset = cell_size + 10
        oscillation_warning = ""
        if len(last_states) >= 2 and last_states[-1] == last_states[-2]:
            oscillation_warning = " | [OSCILLATION DETECTEE]"
        info_lines = [
            f"Episode: {episode_num} | Steps: {episode_steps}/{max_steps_per_episode} | Reward: {episode_reward:.1f}{oscillation_warning}",
            f"Position: {state} | Goal: {env.goal} | Speed: {speed}",
            f"Mode: {mode.upper()} | {'PAUSED' if paused else 'RUNNING'} | [SPACE] Pause | [R] Reset | [Q] Quit",
        ]
        
        for i, line in enumerate(info_lines):
            text = font_small.render(line, True, COLORS['text'])
            screen.blit(text, (10, y_offset + i * 20))
        
        # Q-values pour la position actuelle
        if hasattr(agent, 'get_q'):
            q_left = agent.get_q(state, 0)
            q_right = agent.get_q(state, 1)
            q_text = font_small.render(f"Q({state}, gauche)={q_left:.2f} | Q({state}, droite)={q_right:.2f}", 
                                      True, COLORS['text'])
            screen.blit(q_text, (10, y_offset + 60))
        
        # Légende des couleurs (si espace disponible)
        if width > 400:  # Afficher légende seulement si assez large
            legend_y = y_offset + 80
            legend_items = [
                ("Agent", COLORS['agent']),
                ("Start", COLORS['start']),
                ("Goal", COLORS['goal']),
                ("Obstacle", COLORS['obstacle']),
            ]
            if hasattr(env, 'rewards_cells') and env.rewards_cells:
                legend_items.append(("Reward", COLORS['reward_pos']))
            if hasattr(env, 'traps') and env.traps:
                legend_items.append(("Trap", COLORS['reward_neg']))
            
            legend_x = width - 150
            for i, (label, color) in enumerate(legend_items):
                pygame.draw.rect(screen, color, (legend_x, legend_y + i * 15, 15, 10))
                legend_text = font_small.render(label, True, COLORS['text'])
                screen.blit(legend_text, (legend_x + 20, legend_y + i * 15 - 2))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


def visualize_gridworld(env, agent, mode='eval', speed=2, algo_name='Q-Learning'):
    """Visualise GridWorld avec Pygame"""
    pygame.init()
    
    cell_size = 50
    width = env.width * cell_size
    height = env.height * cell_size + 150
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"GridWorld - {agent.name} - Mode: {mode}")
    
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    step_delay = 1000 // speed
    
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    last_step_time = time.time() * 1000
    max_steps_per_episode = 200  # Limite pour éviter les boucles infinies
    last_states = []  # Pour détecter les oscillations
    oscillation_threshold = 10  # Nombre de répétitions avant reset
    
    # Directions pour afficher
    action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    while running:
        current_time = time.time() * 1000
        
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    episode_num += 1
                    last_states = []
                elif event.key == pygame.K_UP:
                    speed = min(20, speed + 1)
                    step_delay = 1000 // speed
                elif event.key == pygame.K_DOWN:
                    speed = max(1, speed - 1)
                    step_delay = 1000 // speed
                elif event.key == pygame.K_q:
                    running = False
        
        # Step automatique
        if not paused and not env.done and (current_time - last_step_time) >= step_delay:
            action = agent.select_action(state, training=(mode == 'train'))
            next_state, reward, done, info = env.step(action)
            
            if mode == 'train':
                if algo_name == 'SARSA':
                    if not done:
                        next_action = agent.select_action(next_state, training=True)
                        agent.update(state, action, reward, next_state, next_action, done)
                    else:
                        agent.update(state, action, reward, next_state, 0, done)
                else:
                    agent.update(state, action, reward, next_state, done)
            
            # Détecter les oscillations (même état répété)
            if len(last_states) >= 2 and last_states[-1] == state and last_states[-2] == next_state:
                # Oscillation détectée
                if episode_steps >= oscillation_threshold:
                    print(f"Oscillation detectee apres {episode_steps} steps. Reset...")
                    done = True
            
            # Mettre à jour l'historique des états
            last_states.append(state)
            if len(last_states) > 5:
                last_states.pop(0)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            last_step_time = current_time
            
            # Reset si trop de steps ou oscillation
            if done or episode_steps >= max_steps_per_episode:
                if episode_steps >= max_steps_per_episode:
                    print(f"Limite de steps atteinte ({max_steps_per_episode}). Reset...")
                pygame.time.wait(500)
                state = env.reset()
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
                last_states = []
        
        # Dessiner
        screen.fill(COLORS['background'])
        
        # Dessiner la grille
        for x in range(env.width):
            for y in range(env.height):
                pos = (x, y)
                rect_x = x * cell_size
                rect_y = y * cell_size
                color = COLORS['cell']
                
                # Types de cellules
                if pos == env.start:
                    color = COLORS['start']
                elif pos == env.goal:
                    color = COLORS['goal']
                elif pos in env.obstacles:
                    color = COLORS['obstacle']
                elif pos in getattr(env, 'rewards_cells', {}):
                    color = COLORS['reward_pos']
                elif pos in getattr(env, 'traps', set()):
                    color = COLORS['reward_neg']
                
                # Agent
                if pos == state:
                    color = COLORS['agent']
                
                pygame.draw.rect(screen, color, 
                               (rect_x + 2, rect_y + 2, cell_size - 4, cell_size - 4))
        
        # Infos en bas
        y_offset = env.height * cell_size + 10
        oscillation_warning = ""
        if len(last_states) >= 2 and last_states[-1] == last_states[-2]:
            oscillation_warning = " | [OSCILLATION DETECTEE]"
        info_lines = [
            f"Episode: {episode_num} | Steps: {episode_steps}/{max_steps_per_episode} | Reward: {episode_reward:.1f}{oscillation_warning}",
            f"Position: {state} | Goal: {env.goal} | Speed: {speed}",
            f"Mode: {mode.upper()} | {'PAUSED' if paused else 'RUNNING'} | [SPACE] Pause | [R] Reset | [Q] Quit",
        ]
        
        for i, line in enumerate(info_lines):
            text = font_small.render(line, True, COLORS['text'])
            screen.blit(text, (10, y_offset + i * 20))
        
        # Q-values pour la position actuelle
        if hasattr(agent, 'get_q'):
            q_values = [agent.get_q(state, a) for a in range(4)]
            action_names = ['↑', '↓', '←', '→']
            q_text = font_small.render(
                f"Q-values: ↑={q_values[0]:.2f} ↓={q_values[1]:.2f} ←={q_values[2]:.2f} →={q_values[3]:.2f}",
                True, COLORS['text'])
            screen.blit(q_text, (10, y_offset + 60))
            
            # Afficher la meilleure action
            best_action = max(range(4), key=lambda a: q_values[a])
            best_text = font_small.render(f"Best action: {action_names[best_action]} (Q={q_values[best_action]:.2f})",
                                         True, (100, 255, 100))
            screen.blit(best_text, (10, y_offset + 80))
        
        # Légende des couleurs
        legend_y = y_offset + 100
        legend_items = [
            ("Agent", COLORS['agent']),
            ("Start", COLORS['start']),
            ("Goal", COLORS['goal']),
            ("Obstacle", COLORS['obstacle']),
        ]
        if hasattr(env, 'rewards_cells') and env.rewards_cells:
            legend_items.append(("Reward", COLORS['reward_pos']))
        if hasattr(env, 'traps') and env.traps:
            legend_items.append(("Trap", COLORS['reward_neg']))
        
        legend_x = width - 150
        for i, (label, color) in enumerate(legend_items):
            pygame.draw.rect(screen, color, (legend_x, legend_y + i * 15, 15, 10))
            legend_text = font_small.render(label, True, COLORS['text'])
            screen.blit(legend_text, (legend_x + 20, legend_y + i * 15 - 2))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Visualiser les environnements RL')
    parser.add_argument('--env', type=str, default='LineWorldSimple',
                       choices=['LineWorldSimple', 'GridWorldSimple'],
                       help='Environnement a visualiser')
    parser.add_argument('--algo', type=str, default='Q-Learning',
                       choices=['Q-Learning', 'SARSA'],
                       help='Algorithme a utiliser')
    parser.add_argument('--mode', type=str, default='eval',
                       choices=['train', 'eval'],
                       help='Mode: train (avec apprentissage) ou eval (sans apprentissage)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Nombre d\'episodes d\'entrainement (si mode=eval, agent deja entraine)')
    parser.add_argument('--speed', type=int, default=2,
                       help='Vitesse d\'affichage (1-20)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Taux d\'apprentissage')
    parser.add_argument('--gamma', type=float, default=0.99, help='Facteur d\'actualisation')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration')
    
    args = parser.parse_args()
    
    # Créer l'environnement
    if args.env == 'LineWorldSimple':
        env = LineWorldSimple(length=10)
        visualize_func = visualize_lineworld
    elif args.env == 'GridWorldSimple':
        env = GridWorldSimple(width=8, height=8)
        visualize_func = visualize_gridworld
    
    # Créer l'agent
    if args.algo == 'Q-Learning':
        agent = QLearningAgent(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    elif args.algo == 'SARSA':
        agent = SARSAAgent(env, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    
    # Entraîner l'agent si nécessaire
    if args.mode == 'eval':
        print(f"Entrainement de {args.algo} sur {args.env}...")
        agent.train(num_episodes=args.episodes, verbose=True, max_steps_per_episode=1000)
        print("Entrainement termine. Lancement de la visualisation...")
        time.sleep(1)
    
    # Lancer la visualisation
    visualize_func(env, agent, mode=args.mode, speed=args.speed, algo_name=args.algo)


if __name__ == "__main__":
    main()

