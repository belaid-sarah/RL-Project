import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from envs.monty_hall_level2 import MontyHallLevel2

# --- Initialisation Pygame ---
pygame.init()
screen_width = 1100
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Monty Hall Level 2 - 5 Portes")

# Polices
font_large = pygame.font.Font(None, 52)
font_medium = pygame.font.Font(None, 32)
font_small = pygame.font.Font(None, 26)
font_tiny = pygame.font.Font(None, 20)

# Couleurs
BLACK = (20, 20, 25)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (100, 150, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
PURPLE = (200, 100, 255)

# --- Créer l'environnement ---
env = MontyHallLevel2()
state = env.reset()

# Variables de jeu
game_over = False
final_reward = 0
removed_doors = []

def draw_door(x, y, width, height, label, status="closed", show_content=False, is_winner=False):
    """Dessine une porte"""
    # Couleur selon le statut
    if status == "chosen":
        color = BLUE
        border_color = WHITE
        border_width = 5
    elif status == "removed":
        color = DARK_GRAY
        border_color = RED
        border_width = 2
    elif status == "winner" and show_content:
        color = GREEN
        border_color = GOLD
        border_width = 6
    elif status == "loser" and show_content:
        color = RED
        border_color = DARK_GRAY
        border_width = 3
    elif status == "available":
        color = GRAY
        border_color = WHITE
        border_width = 3
    else:
        color = (60, 60, 60)
        border_color = GRAY
        border_width = 2
    
    # Dessiner la porte
    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=8)
    pygame.draw.rect(screen, border_color, (x, y, width, height), border_width, border_radius=8)
    
    # Label de la porte
    label_text = font_medium.render(label, True, WHITE)
    label_rect = label_text.get_rect(center=(x + width//2, y + 30))
    screen.blit(label_text, label_rect)
    
    # Contenu (si révélé)
    if show_content:
        if is_winner:
            content = "🚗"
            content_text = "WIN"
            color_text = GOLD
        else:
            content = "🐐"
            content_text = "LOSE"
            color_text = RED
        
        emoji_text = font_large.render(content, True, WHITE)
        emoji_rect = emoji_text.get_rect(center=(x + width//2, y + height//2))
        screen.blit(emoji_text, emoji_rect)
        
        text = font_small.render(content_text, True, color_text)
        text_rect = text.get_rect(center=(x + width//2, y + height - 30))
        screen.blit(text, text_rect)
    elif status == "removed":
        # Croix pour porte retirée
        pygame.draw.line(screen, RED, (x + 15, y + 15), (x + width - 15, y + height - 15), 4)
        pygame.draw.line(screen, RED, (x + width - 15, y + 15), (x + 15, y + height - 15), 4)
        
        removed_text = font_small.render("OUT", True, RED)
        removed_rect = removed_text.get_rect(center=(x + width//2, y + height//2))
        screen.blit(removed_text, removed_rect)
    else:
        # Poignée
        pygame.draw.circle(screen, border_color, (x + width - 20, y + height//2), 8)

def draw_game():
    screen.fill(BLACK)
    
    # Titre
    title = font_large.render("MONTY HALL - LEVEL 2", True, GOLD)
    title_rect = title.get_rect(center=(screen_width//2, 35))
    screen.blit(title, title_rect)
    
    # Instructions selon l'étape
    if env.step_count == 0:
        instruction = f"Step 1/4: Choose your first door"
        color = WHITE
    elif env.step_count < 4:
        instruction = f"Step {env.step_count}/4: Choose again or keep your choice"
        color = BLUE
    elif env.step_count == 4 and not game_over:
        instruction = f"Step 4/4: FINAL CHOICE - Make your decision!"
        color = PURPLE
    else:
        if final_reward > 0.5:
            instruction = "🎉 YOU WON!"
            color = GREEN
        else:
            instruction = "😞 YOU LOST"
            color = RED
    
    inst_text = font_medium.render(instruction, True, color)
    inst_rect = inst_text.get_rect(center=(screen_width//2, 85))
    screen.blit(inst_text, inst_rect)
    
    # Info sur la porte choisie
    if env.chosen_door and not game_over:
        choice_text = font_small.render(f"Current choice: {env.chosen_door}", True, BLUE)
        screen.blit(choice_text, (screen_width//2 - 100, 120))
    
    # Dessiner les 5 portes
    door_width = 140
    door_height = 220
    spacing = 30
    total_width = 5 * door_width + 4 * spacing
    start_x = (screen_width - total_width) // 2
    door_y = 170
    
    for i, door_label in enumerate(env.doors):
        x = start_x + i * (door_width + spacing)
        
        # Déterminer le statut
        if game_over:
            # Révéler tout
            is_winner = door_label == env.winning_door
            if door_label in removed_doors:
                status = "removed"
                show_content = False
            elif is_winner:
                status = "winner"
                show_content = True
            else:
                status = "loser"
                show_content = True
        elif door_label in removed_doors:
            status = "removed"
            show_content = False
            is_winner = False
        elif door_label == env.chosen_door:
            status = "chosen"
            show_content = False
            is_winner = False
        elif door_label in env.available_doors:
            status = "available"
            show_content = False
            is_winner = False
        else:
            status = "closed"
            show_content = False
            is_winner = False
        
        draw_door(x, door_y, door_width, door_height, door_label, status, show_content, is_winner)
    
    # Afficher les portes disponibles cliquables
    if not game_over and env.available_doors:
        available_y = door_y + door_height + 20
        available_text = font_small.render(
            f"Available doors: {', '.join(env.available_doors)}", 
            True, GREEN
        )
        available_rect = available_text.get_rect(center=(screen_width//2, available_y))
        screen.blit(available_text, available_rect)
    
    # Historique des portes retirées
    if removed_doors:
        removed_y = door_y + door_height + 50
        removed_text = font_tiny.render(
            f"Removed: {', '.join(removed_doors)}", 
            True, RED
        )
        removed_rect = removed_text.get_rect(center=(screen_width//2, removed_y))
        screen.blit(removed_text, removed_rect)
    
    # Statistiques
    stats = env.get_stats()
    stats_y = screen_height - 100
    
    stats_text = font_small.render(
        f"Games: {stats['games_played']} | Wins: {stats['wins']} | Win Rate: {stats['win_rate']*100:.1f}%",
        True, WHITE
    )
    screen.blit(stats_text, (20, stats_y))
    
    reward_text = font_small.render(
        f"Total Reward: {stats['total_reward']:.2f} | Avg: {stats['avg_reward']:.2f}",
        True, GOLD
    )
    screen.blit(reward_text, (20, stats_y + 30))
    
    # Indicateur de progression
    progress_y = screen_height - 140
    progress_text = font_tiny.render(f"Progress: {env.step_count}/4 steps", True, BLUE)
    screen.blit(progress_text, (20, progress_y))
    
    # Barre de progression
    progress_bar_width = 200
    progress_bar_height = 15
    progress_x = 20
    progress_y_bar = progress_y + 25
    
    pygame.draw.rect(screen, DARK_GRAY, (progress_x, progress_y_bar, progress_bar_width, progress_bar_height))
    filled_width = int((env.step_count / 4) * progress_bar_width)
    pygame.draw.rect(screen, GREEN, (progress_x, progress_y_bar, filled_width, progress_bar_height))
    pygame.draw.rect(screen, WHITE, (progress_x, progress_y_bar, progress_bar_width, progress_bar_height), 2)
    
    # Bouton reset si game over
    if game_over:
        reset_rect = pygame.Rect(screen_width//2 - 100, screen_height - 60, 200, 45)
        pygame.draw.rect(screen, GOLD, reset_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, reset_rect, 3, border_radius=10)
        reset_text = font_medium.render("PLAY AGAIN", True, BLACK)
        reset_text_rect = reset_text.get_rect(center=reset_rect.center)
        screen.blit(reset_text, reset_text_rect)

# --- Boucle principale ---
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Cliquer sur une porte disponible
            if not game_over:
                door_width = 140
                door_height = 220
                spacing = 30
                total_width = 5 * door_width + 4 * spacing
                start_x = (screen_width - total_width) // 2
                door_y = 170
                
                for i, door_label in enumerate(env.doors):
                    if door_label in env.available_doors:
                        x = start_x + i * (door_width + spacing)
                        door_rect = pygame.Rect(x, door_y, door_width, door_height)
                        
                        if door_rect.collidepoint(mouse_pos):
                            # Trouver l'index dans available_doors
                            action = env.available_doors.index(door_label)
                            state, reward, done, info = env.step(action)
                            
                            if "removed" in info:
                                removed_doors.append(info["removed"])
                            
                            if done:
                                final_reward = reward
                                game_over = True
                            
                            break
            
            # Reset
            elif game_over:
                reset_rect = pygame.Rect(screen_width//2 - 100, screen_height - 60, 200, 45)
                if reset_rect.collidepoint(mouse_pos):
                    state = env.reset()
                    game_over = False
                    final_reward = 0
                    removed_doors = []
    
    draw_game()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()