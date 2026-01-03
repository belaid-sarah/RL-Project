import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from envs.lineworld import LineWorld

# --- Initialisation Pygame ---
pygame.init()
length_cells = 30  # Longueur de la ligne
cell_size = 40  # Réduit car ligne plus longue
margin = 50
screen_width = length_cells * cell_size + 2 * margin
screen_height = 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("LineWorld RL - Complex")

font_small = pygame.font.Font(None, 20)
font_medium = pygame.font.Font(None, 24)
font_large = pygame.font.Font(None, 32)
font_tiny = pygame.font.Font(None, 16)

# --- Créer l'environnement ---
env = LineWorld(length=length_cells)
state_dict = env.reset()
state = state_dict['position']

# --- Boucle principale ---
running = True
clock = pygame.time.Clock()
speed = 3  # FPS

# Couleurs
BLACK = (20, 20, 25)
WHITE = (255, 255, 255)
AGENT_COLOR = (100, 255, 100)
GOAL_COLOR = (50, 200, 50)
OBSTACLE_COLOR = (60, 60, 60)
MOVING_OBS_COLOR = (200, 80, 80)
TRAP_COLOR = (180, 50, 50)
REWARD_COLOR = (255, 200, 50)
WIND_COLOR = (100, 150, 255)
PORTAL_COLOR = (200, 100, 255)
ICE_COLOR = (150, 220, 255)
KEY_COLOR = (255, 215, 0)
DOOR_COLOR = (139, 69, 19)
VISITED_COLOR = (60, 60, 80)
NORMAL_COLOR = (80, 80, 90)

while running:
    screen.fill(BLACK)
    
    # Position Y pour centrer la ligne
    line_y = screen_height // 2 - cell_size
    
    # Dessiner la ligne
    for i in range(env.length):
        x = margin + i * cell_size
        
        # Déterminer la couleur de base
        color = NORMAL_COLOR
        
        # Cellule visitée
        if i in env.visited:
            color = VISITED_COLOR
        
        # Types de cellules spéciales (par priorité)
        if i in env.obstacles:
            color = OBSTACLE_COLOR
        elif i in env.ice_zones:
            color = ICE_COLOR
        elif i in env.traps:
            color = TRAP_COLOR
        elif i in env.rewards:
            color = REWARD_COLOR
        elif i in env.wind_zones:
            color = WIND_COLOR
        elif i in env.portals:
            color = PORTAL_COLOR
        elif i in env.doors:
            color = DOOR_COLOR
        elif i in env.keys:
            color = KEY_COLOR
        elif i == env.goal:
            color = GOAL_COLOR
        
        # Agent
        if i == state:
            color = AGENT_COLOR
        
        # Dessiner la cellule
        pygame.draw.rect(screen, color, (x, line_y, cell_size-3, cell_size-3))
        
        # Dessiner obstacles mobiles (par-dessus)
        for obs in env.moving_obstacles:
            if int(obs['pos']) == i:
                pygame.draw.circle(screen, MOVING_OBS_COLOR, 
                                 (x + cell_size//2, line_y + cell_size//2), 
                                 cell_size//3)
        
        # Ajouter des symboles pour certaines cellules
        symbol = None
        if i in env.wind_zones and i != state:
            wind_dir = "◄" if env.wind_zones[i]['direction'] == -1 else "►"
            symbol = wind_dir
        elif i in env.portals and i != state:
            symbol = "⊕"
        elif i in env.doors and i != state:
            symbol = "▓"
        elif i in env.keys and i != state:
            symbol = "K"
        elif i in env.ice_zones and i != state:
            symbol = "❄"
        
        if symbol:
            text = font_medium.render(symbol, True, WHITE)
            text_rect = text.get_rect(center=(x + cell_size//2, line_y + cell_size//2))
            screen.blit(text, text_rect)
        
        # Numéro de position (en petit)
        if i % 5 == 0:  # Afficher seulement tous les 5
            pos_text = font_tiny.render(str(i), True, (120, 120, 120))
            screen.blit(pos_text, (x + 2, line_y - 15))
    
    # Légende en haut
    legend_y = 20
    legends = [
        ("Agent", AGENT_COLOR),
        ("Goal", GOAL_COLOR),
        ("Obstacle", OBSTACLE_COLOR),
        ("Moving", MOVING_OBS_COLOR),
        ("Trap", TRAP_COLOR),
        ("Reward", REWARD_COLOR),
        ("Ice", ICE_COLOR),
        ("Wind", WIND_COLOR),
        ("Portal", PORTAL_COLOR),
        ("Key", KEY_COLOR),
        ("Door", DOOR_COLOR)
    ]
    
    legend_x = margin
    for legend_text, legend_color in legends:
        pygame.draw.rect(screen, legend_color, (legend_x, legend_y, 12, 12))
        text = font_tiny.render(legend_text, True, WHITE)
        screen.blit(text, (legend_x + 16, legend_y - 2))
        legend_x += 90
    
    # Informations détaillées
    info_y = line_y + cell_size + 40
    
    # Ligne 1
    info1 = font_small.render(
        f"Position: {state}/{env.length-1} | Steps: {env.step_count}/{env.max_steps} | Energy: {env.energy}/100",
        True, WHITE)
    screen.blit(info1, (margin, info_y))
    
    # Ligne 2
    keys_status = f"{len(env.collected_keys)}/3"
    info2 = font_small.render(
        f"Total Reward: {env.total_reward:.2f} | Keys: {keys_status} | Explored: {len(env.visited)}/{env.length}",
        True, (255, 215, 0))
    screen.blit(info2, (margin, info_y + 25))
    
    # Ligne 3 - Keys collectées
    collected_text = f"Collected Keys: {', '.join(env.collected_keys) if env.collected_keys else 'None'}"
    info3 = font_small.render(collected_text, True, (100, 255, 100))
    screen.blit(info3, (margin, info_y + 50))
    
    # Ligne 4 - Rewards restantes
    rewards_left = len(env.rewards)
    info4 = font_small.render(
        f"Rewards left: {rewards_left} | Exploration: {len(env.visited)/env.length*100:.1f}%",
        True, (255, 150, 50))
    screen.blit(info4, (margin, info_y + 75))
    
    # Barre d'énergie visuelle
    energy_bar_x = margin
    energy_bar_y = info_y + 105
    energy_bar_width = 300
    energy_bar_height = 20
    
    # Fond de la barre
    pygame.draw.rect(screen, (50, 50, 50), 
                    (energy_bar_x, energy_bar_y, energy_bar_width, energy_bar_height))
    # Énergie actuelle
    energy_width = int((env.energy / 100) * energy_bar_width)
    energy_color = (0, 255, 0) if env.energy > 50 else (255, 165, 0) if env.energy > 20 else (255, 0, 0)
    pygame.draw.rect(screen, energy_color,
                    (energy_bar_x, energy_bar_y, energy_width, energy_bar_height))
    # Texte énergie
    energy_text = font_tiny.render(f"Energy: {env.energy}/100", True, WHITE)
    screen.blit(energy_text, (energy_bar_x + energy_bar_width + 10, energy_bar_y + 3))
    
    # Instructions
    instructions = font_tiny.render(
        "SPACE: Reset | UP/DOWN: Speed | ESC: Quit | Actions: 0=Left 1=Right 2=Stay 3=Jump 4=Sprint",
        True, (150, 150, 150))
    screen.blit(instructions, (margin, info_y + 135))

    pygame.display.flip()
    clock.tick(speed)

    # Action aléatoire
    if not env.done:
        action = env.sample_action()
        state_dict, reward, done, info = env.step(action)
        state = state_dict['position']
    else:
        # Afficher message de fin
        if len(env.collected_keys) == 3 and state == env.goal:
            end_msg = "SUCCESS! All keys collected and goal reached!"
            color = (0, 255, 0)
        else:
            end_msg = "FAILED! Timeout or out of energy"
            color = (255, 0, 0)
        
        end_text = font_medium.render(end_msg, True, color)
        end_rect = end_text.get_rect(center=(screen_width//2, screen_height - 30))
        screen.blit(end_text, end_rect)
        pygame.display.flip()
        
        pygame.time.wait(2000)  # Pause avant reset
        state_dict = env.reset()
        state = state_dict['position']

    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                state_dict = env.reset()
                state = state_dict['position']
            elif event.key == pygame.K_UP:
                speed = min(30, speed + 2)
            elif event.key == pygame.K_DOWN:
                speed = max(1, speed - 2)
            elif event.key == pygame.K_ESCAPE:
                running = False

pygame.quit()