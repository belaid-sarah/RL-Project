import pygame
import sys
import time
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent

# Configuration des couleurs et tailles
CELL_SIZE = 60
COLOR_BG = (30, 30, 30)
COLOR_WALL = (50, 50, 50)
COLOR_TRAP = (200, 50, 50)
COLOR_GOAL = (50, 200, 50)
COLOR_REWARD = (255, 215, 0)
COLOR_AGENT = (50, 150, 255)

def run_visual_demo():
    # 1. Initialisation
    env = GridWorld(width=10, height=10)
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # 2. Entraînement rapide (sans affichage pour la vitesse)
    print("⏳ Entraînement en cours...")
    agent.train(num_episodes=1000, verbose=False)
    print("✅ Entraînement terminé. Lancement de la démo Pygame.")

    # 3. Setup Pygame
    pygame.init()
    screen = pygame.display.set_mode((env.width * CELL_SIZE, env.height * CELL_SIZE))
    pygame.display.set_caption("RL Demo: Q-Learning on Complex GridWorld")
    clock = pygame.time.Clock()

    state = env.reset()
    done = False

    while True:
        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not done:
            # L'agent choisit la meilleure action (greedy)
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            time.sleep(0.1) # Ralentir pour l'œil humain

        # --- DESSIN ---
        screen.fill(COLOR_BG)

        # Dessiner les obstacles
        for obs in env.obstacles:
            pygame.draw.rect(screen, COLOR_WALL, (obs[0]*CELL_SIZE, obs[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Dessiner les pièges (statiques)
        for trap in env.traps:
            pygame.draw.circle(screen, COLOR_TRAP, (trap[0]*CELL_SIZE + 30, trap[1]*CELL_SIZE + 30), 15)

        # Dessiner les pièges mobiles
        for m_trap in env.moving_traps:
            pos = m_trap['pos']
            pygame.draw.rect(screen, COLOR_TRAP, (pos[0]*CELL_SIZE+10, pos[1]*CELL_SIZE+10, CELL_SIZE-20, CELL_SIZE-20), 3)

        # Dessiner les récompenses
        for r_pos in env.rewards_cells:
            pygame.draw.polygon(screen, COLOR_REWARD, [
                (r_pos[0]*CELL_SIZE + 30, r_pos[1]*CELL_SIZE + 10),
                (r_pos[0]*CELL_SIZE + 10, r_pos[1]*CELL_SIZE + 50),
                (r_pos[0]*CELL_SIZE + 50, r_pos[1]*CELL_SIZE + 50)
            ])

        # Goal
        pygame.draw.rect(screen, COLOR_GOAL, (env.goal[0]*CELL_SIZE, env.goal[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Agent
        pygame.draw.circle(screen, COLOR_AGENT, (state[0]*CELL_SIZE + 30, state[1]*CELL_SIZE + 30), 20)

        pygame.display.flip()
        clock.tick(10) # 10 FPS

        if done:
            print(f"Épisode terminé ! Reward total : {info['total_reward']}")
            time.sleep(2)
            state = env.reset()
            done = False

if __name__ == "__main__":
    run_visual_demo()