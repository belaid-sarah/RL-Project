"""
Script pour dérouler une stratégie apprise pas à pas
Parfait pour la soutenance : montre la politique sans avoir à réentraîner
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
import argparse
import pickle

from envs.lineworld_simple import LineWorldSimple
from envs.gridworld_simple import GridWorldSimple
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
from algos.policy_iteration import PolicyIteration
from algos.value_iteration import ValueIteration

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

def replay_lineworld(env, agent, model_path=None):
    """Rejoue une politique sur LineWorld pas à pas"""
    pygame.init()
    
    cell_size = 50
    width = env.length * cell_size
    height = cell_size + 200
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Replay Policy - {agent.name}")
    
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    clock = pygame.time.Clock()
    running = True
    paused = True  # Commence en pause pour contrôle manuel
    
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:  # Flèche droite = step suivant
                    if not env.done:
                        action = agent.select_action(state, training=False)
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        episode_reward += reward
                        episode_steps += 1
                        
                        if done:
                            paused = True
                elif event.key == pygame.K_r:
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    episode_num += 1
                    paused = True
                elif event.key == pygame.K_q:
                    running = False
        
        # Dessiner
        screen.fill(COLORS['background'])
        
        # Dessiner la ligne
        for pos in range(env.length):
            x = pos * cell_size
            color = COLORS['cell']
            
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
            
            if pos == state:
                color = COLORS['agent']
            
            pygame.draw.rect(screen, color, (x + 2, 2, cell_size - 4, cell_size - 4))
            
            # Numéro
            text = font_small.render(str(pos), True, COLORS['text'])
            text_rect = text.get_rect(center=(x + cell_size // 2, cell_size // 2))
            screen.blit(text, text_rect)
        
        # Infos
        y_offset = cell_size + 10
        info_lines = [
            f"Episode: {episode_num} | Steps: {episode_steps} | Reward: {episode_reward:.1f}",
            f"Position: {state} | Goal: {env.goal}",
            f"{'PAUSED' if paused else 'RUNNING'} | [→] Next Step | [SPACE] Auto | [R] Reset | [Q] Quit",
        ]
        
        for i, line in enumerate(info_lines):
            text = font_small.render(line, True, COLORS['text'])
            screen.blit(text, (10, y_offset + i * 20))
        
        # Q-values
        if hasattr(agent, 'get_q'):
            q_left = agent.get_q(state, 0)
            q_right = agent.get_q(state, 1)
            best_action = "gauche" if q_left > q_right else "droite"
            q_text = font_small.render(
                f"Q({state}, gauche)={q_left:.2f} | Q({state}, droite)={q_right:.2f} | Best: {best_action}",
                True, COLORS['text'])
            screen.blit(q_text, (10, y_offset + 60))
        
        # Auto-play si pas en pause
        if not paused and not env.done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                paused = True
                pygame.time.wait(1000)
                state = env.reset()
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
        
        pygame.display.flip()
        clock.tick(10 if paused else 2)  # Plus lent en mode pas à pas
    
    pygame.quit()


def replay_gridworld(env, agent, model_path=None):
    """Rejoue une politique sur GridWorld pas à pas"""
    pygame.init()
    
    cell_size = 50
    width = env.width * cell_size
    height = env.height * cell_size + 200
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Replay Policy - {agent.name}")
    
    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    clock = pygame.time.Clock()
    running = True
    paused = True
    
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    
    action_names = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    if not env.done:
                        action = agent.select_action(state, training=False)
                        next_state, reward, done, info = env.step(action)
                        state = next_state
                        episode_reward += reward
                        episode_steps += 1
                        
                        if done:
                            paused = True
                elif event.key == pygame.K_r:
                    state = env.reset()
                    episode_reward = 0
                    episode_steps = 0
                    episode_num += 1
                    paused = True
                elif event.key == pygame.K_q:
                    running = False
        
        # Dessiner
        screen.fill(COLORS['background'])
        
        for x in range(env.width):
            for y in range(env.height):
                pos = (x, y)
                rect_x = x * cell_size
                rect_y = y * cell_size
                color = COLORS['cell']
                
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
                
                if pos == state:
                    color = COLORS['agent']
                
                pygame.draw.rect(screen, color, 
                               (rect_x + 2, rect_y + 2, cell_size - 4, cell_size - 4))
        
        # Infos
        y_offset = env.height * cell_size + 10
        info_lines = [
            f"Episode: {episode_num} | Steps: {episode_steps} | Reward: {episode_reward:.1f}",
            f"Position: {state} | Goal: {env.goal}",
            f"{'PAUSED' if paused else 'RUNNING'} | [→] Next Step | [SPACE] Auto | [R] Reset | [Q] Quit",
        ]
        
        for i, line in enumerate(info_lines):
            text = font_small.render(line, True, COLORS['text'])
            screen.blit(text, (10, y_offset + i * 20))
        
        # Q-values
        if hasattr(agent, 'get_q'):
            q_values = [agent.get_q(state, a) for a in range(4)]
            best_action_idx = max(range(4), key=lambda a: q_values[a])
            q_text = font_small.render(
                f"Q-values: ↑={q_values[0]:.2f} ↓={q_values[1]:.2f} ←={q_values[2]:.2f} →={q_values[3]:.2f} | Best: {action_names[best_action_idx]}",
                True, COLORS['text'])
            screen.blit(q_text, (10, y_offset + 60))
        
        # Auto-play
        if not paused and not env.done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                paused = True
                pygame.time.wait(1000)
                state = env.reset()
                episode_num += 1
                episode_reward = 0
                episode_steps = 0
        
        pygame.display.flip()
        clock.tick(10 if paused else 2)
    
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Rejouer une politique apprise pas à pas')
    parser.add_argument('--env', type=str, required=True,
                       choices=['LineWorldSimple', 'GridWorldSimple'],
                       help='Environnement')
    parser.add_argument('--algo', type=str, required=True,
                       choices=['Q-Learning', 'SARSA', 'PolicyIteration', 'ValueIteration'],
                       help='Algorithme')
    parser.add_argument('--model', type=str, required=True,
                       help='Chemin vers le modèle sauvegardé (.pkl)')
    parser.add_argument('--length', type=int, default=25, help='Longueur pour LineWorld')
    parser.add_argument('--width', type=int, default=10, help='Largeur pour GridWorld')
    parser.add_argument('--height', type=int, default=10, help='Hauteur pour GridWorld')
    
    args = parser.parse_args()
    
    # Créer l'environnement
    if args.env == 'LineWorldSimple':
        env = LineWorldSimple(length=args.length)
        replay_func = replay_lineworld
    elif args.env == 'GridWorldSimple':
        env = GridWorldSimple(width=args.width, height=args.height)
        replay_func = replay_gridworld
    
    # Créer l'agent (structure minimale)
    if args.algo == 'Q-Learning':
        agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    elif args.algo == 'SARSA':
        from algos.sarsa import SARSAAgent
        agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    elif args.algo == 'PolicyIteration':
        agent = PolicyIteration(env, gamma=0.99, theta=1e-5)
    elif args.algo == 'ValueIteration':
        agent = ValueIteration(env, gamma=0.99, theta=1e-5)
    
    # Charger le modèle
    if not Path(args.model).exists():
        print(f"[ERROR] Erreur: Le fichier {args.model} n'existe pas")
        print(f"   Entraînez d'abord un agent et sauvegardez-le:")
        print(f"   agent.train(num_episodes=1000)")
        print(f"   agent.save('{args.model}')")
        sys.exit(1)
    
    print(f"Chargement du modèle: {args.model}")
    agent.load(args.model)
    print("[OK] Modele charge! Pret pour le replay pas a pas.")
    print("\nContrôles:")
    print("  [→] : Step suivant")
    print("  [SPACE] : Auto-play / Pause")
    print("  [R] : Reset")
    print("  [Q] : Quitter")
    print("\nLancement de la visualisation...")
    
    # Lancer le replay
    replay_func(env, agent, args.model)


if __name__ == "__main__":
    main()

