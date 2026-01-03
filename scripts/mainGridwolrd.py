import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from envs.gridworld import GridWorld

# --- Initialisation Pygame ---
pygame.init()
cell_size = 60  # Plus petit car grille plus grande
width_cells, height_cells = 10, 10  # <-- Grille 10x10
screen_width = width_cells * cell_size
screen_height = height_cells * cell_size + 50  # +50 pour les infos
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("GridWorld RL - 10x10")

font_small = pygame.font.Font(None, 20)
font_large = pygame.font.Font(None, 28)

# --- Créer l'environnement ---
env = GridWorld(width=width_cells, height=height_cells)
state = env.reset()

# --- Boucle principale ---
running = True
clock = pygame.time.Clock()
speed = 5  # FPS

while running:
    screen.fill((20, 20, 25))

    # Dessiner la grille
    for x in range(env.width):
        for y in range(env.height):
            color = (80, 80, 90)  # Cellule normale
            
            # Cellules visitées
            if (x, y) in env.visited:
                color = (60, 60, 80)
            
            # Types de cellules
            if (x, y) in env.obstacles:
                color = (40, 40, 40)  # Obstacle
            elif (x, y) in env.traps:
                color = (180, 30, 30)  # Piège statique
            elif (x, y) in env.rewards_cells:
                color = (255, 200, 50)  # Récompense
            elif (x, y) == env.goal:
                color = (50, 200, 50)  # Goal
            
            # Pièges mobiles
            for trap in env.moving_traps:
                if trap['pos'] == (x, y):
                    color = (220, 50, 220)  # Piège mobile (magenta)
            
            # Agent
            if (x, y) == state:
                color = (100, 255, 100)  # Agent
            
            pygame.draw.rect(screen, color, 
                           (x*cell_size + 2, y*cell_size + 2, 
                            cell_size-4, cell_size-4))
    
    # Afficher les infos en bas
    info_y = height_cells * cell_size + 10
    info1 = font_small.render(
        f"Steps: {env.step_count}/{env.max_steps} | Reward: {env.total_reward:.1f} | Explored: {len(env.visited)}/{env.width*env.height}", 
        True, (255, 255, 255))
    screen.blit(info1, (10, info_y))
    
    info2 = font_small.render(
        f"Collected: {width_cells//2 - len(env.rewards_cells)}/{width_cells//2}", 
        True, (255, 215, 0))
    screen.blit(info2, (10, info_y + 20))

    pygame.display.flip()
    clock.tick(speed)

    # Action aléatoire
    if not env.done:
        action = env.sample_action()
        state, reward, done, info = env.step(action)
    else:
        pygame.time.wait(1000)  # Pause avant reset
        state = env.reset()

    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Espace pour reset
                state = env.reset()
            elif event.key == pygame.K_UP:  # Augmenter vitesse
                speed = min(30, speed + 2)
            elif event.key == pygame.K_DOWN:  # Diminuer vitesse
                speed = max(1, speed - 2)

pygame.quit()