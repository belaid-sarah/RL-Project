import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from envs.monty_hall_level1 import MontyHallLevel1

# --- Initialisation Pygame ---
pygame.init()
screen_width = 900
screen_height = 650
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Monty Hall - 3 Portes")

# Polices
font_large = pygame.font.Font(None, 56)
font_medium = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 28)
font_tiny = pygame.font.Font(None, 22)

# Couleurs
BLACK = (20, 20, 25)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (100, 150, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)

# --- Créer l'environnement ---
env = MontyHallLevel1()  # <-- CHANGÉ ICI : sans paramètres
state = env.reset()

# Variables de jeu
game_over = False
final_reward = 0

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
        border_width = 3
    elif status == "winner" and show_content:
        color = GREEN
        border_color = GOLD
        border_width = 6
    elif status == "loser" and show_content:
        color = RED
        border_color = DARK_GRAY
        border_width = 3
    else:
        color = GRAY
        border_color = WHITE
        border_width = 3
    
    # Dessiner la porte
    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=10)
    pygame.draw.rect(screen, border_color, (x, y, width, height), border_width, border_radius=10)
    
    # Label de la porte
    label_text = font_large.render(label, True, WHITE)
    label_rect = label_text.get_rect(center=(x + width//2, y + 40))
    screen.blit(label_text, label_rect)
    
    # Contenu (si révélé)
    if show_content:
        if is_winner:
            content = "🚗"
            content_text = "CAR!"
            color_text = GOLD
        else:
            content = "🐐"
            content_text = "GOAT"
            color_text = RED
        
        emoji_text = font_large.render(content, True, WHITE)
        emoji_rect = emoji_text.get_rect(center=(x + width//2, y + height//2))
        screen.blit(emoji_text, emoji_rect)
        
        text = font_medium.render(content_text, True, color_text)
        text_rect = text.get_rect(center=(x + width//2, y + height - 40))
        screen.blit(text, text_rect)
    elif status == "removed":
        # Croix pour porte retirée
        pygame.draw.line(screen, RED, (x + 20, y + 20), (x + width - 20, y + height - 20), 5)
        pygame.draw.line(screen, RED, (x + width - 20, y + 20), (x + 20, y + height - 20), 5)
        
        removed_text = font_small.render("REMOVED", True, RED)
        removed_rect = removed_text.get_rect(center=(x + width//2, y + height//2))
        screen.blit(removed_text, removed_rect)
    else:
        # Poignée
        pygame.draw.circle(screen, border_color, (x + width - 30, y + height//2), 12)

def draw_game():
    screen.fill(BLACK)
    
    # Titre
    title = font_large.render("MONTY HALL PROBLEM", True, GOLD)
    title_rect = title.get_rect(center=(screen_width//2, 40))
    screen.blit(title, title_rect)
    
    # Instructions selon l'étape
    if env.step_count == 0:
        instruction = "Choose a door (A, B, or C)"
        color = WHITE
    elif env.step_count == 1:
        instruction = f"Door {env.removed_door} is removed. Keep or Switch?"
        color = BLUE
    else:
        if final_reward > 0.5:
            instruction = "🎉 YOU WON THE CAR!"
            color = GREEN
        else:
            instruction = "😞 You got a goat..."
            color = RED
    
    inst_text = font_medium.render(instruction, True, color)
    inst_rect = inst_text.get_rect(center=(screen_width//2, 100))
    screen.blit(inst_text, inst_rect)
    
    # Dessiner les 3 portes
    door_width = 180
    door_height = 280
    spacing = 50
    total_width = 3 * door_width + 2 * spacing
    start_x = (screen_width - total_width) // 2
    door_y = 160
    
    doors_info = [
        ("A", 0),
        ("B", 1),
        ("C", 2)
    ]
    
    for label, idx in doors_info:
        x = start_x + idx * (door_width + spacing)
        
        # Déterminer le statut
        if game_over:
            # Révéler tout
            is_winner = label == env.winning_door
            if label == env.removed_door:
                status = "removed"
            elif is_winner:
                status = "winner"
            else:
                status = "loser"
            show_content = True
        elif label == env.removed_door:
            status = "removed"
            show_content = True
            is_winner = False
        elif label == env.chosen_door:
            status = "chosen"
            show_content = False
            is_winner = False
        else:
            status = "closed"
            show_content = False
            is_winner = False
        
        draw_door(x, door_y, door_width, door_height, label, status, show_content, is_winner)
    
    # Boutons pour l'étape 2
    if env.step_count == 1 and not game_over:
        button_y = door_y + door_height + 50
        
        # Bouton KEEP
        keep_rect = pygame.Rect(screen_width//2 - 240, button_y, 220, 70)
        pygame.draw.rect(screen, BLUE, keep_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, keep_rect, 4, border_radius=10)
        keep_text = font_medium.render("KEEP", True, WHITE)
        keep_sub = font_tiny.render(f"(Door {env.chosen_door})", True, WHITE)
        keep_text_rect = keep_text.get_rect(center=(keep_rect.centerx, keep_rect.centery - 10))
        keep_sub_rect = keep_sub.get_rect(center=(keep_rect.centerx, keep_rect.centery + 15))
        screen.blit(keep_text, keep_text_rect)
        screen.blit(keep_sub, keep_sub_rect)
        
        # Bouton SWITCH
        switch_rect = pygame.Rect(screen_width//2 + 20, button_y, 220, 70)
        pygame.draw.rect(screen, GREEN, switch_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, switch_rect, 4, border_radius=10)
        switch_text = font_medium.render("SWITCH", True, WHITE)
        switch_sub = font_tiny.render(f"(Door {env.remaining_door})", True, WHITE)
        switch_text_rect = switch_text.get_rect(center=(switch_rect.centerx, switch_rect.centery - 10))
        switch_sub_rect = switch_sub.get_rect(center=(switch_rect.centerx, switch_rect.centery + 15))
        screen.blit(switch_text, switch_text_rect)
        screen.blit(switch_sub, switch_sub_rect)
    
    # Statistiques
    stats = env.get_stats()
    stats_y = screen_height - 80
    
    stats_text = font_small.render(
        f"Games: {stats['games_played']} | Wins: {stats['wins']} | Win Rate: {stats['win_rate']*100:.1f}%",
        True, WHITE
    )
    screen.blit(stats_text, (20, stats_y))
    
    reward_text = font_small.render(
        f"Total Reward: {stats['total_reward']:.2f} | Switch Rate: {stats['switch_rate']*100:.1f}%",
        True, GOLD
    )
    screen.blit(reward_text, (20, stats_y + 30))
    
    # Bouton reset si game over
    if game_over:
        reset_rect = pygame.Rect(screen_width//2 - 100, screen_height - 50, 200, 40)
        pygame.draw.rect(screen, GOLD, reset_rect, border_radius=8)
        pygame.draw.rect(screen, WHITE, reset_rect, 3, border_radius=8)
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
            
            # Étape 1 : Cliquer sur une porte
            if env.step_count == 0 and not game_over:
                door_width = 180
                door_height = 280
                spacing = 50
                total_width = 3 * door_width + 2 * spacing
                start_x = (screen_width - total_width) // 2
                door_y = 160
                
                for idx in range(3):
                    x = start_x + idx * (door_width + spacing)
                    door_rect = pygame.Rect(x, door_y, door_width, door_height)
                    if door_rect.collidepoint(mouse_pos):
                        state, reward, done, info = env.step(idx)
                        break
            
            # Étape 2 : Cliquer sur KEEP ou SWITCH
            elif env.step_count == 1 and not game_over:
                button_y = 160 + 280 + 50
                keep_rect = pygame.Rect(screen_width//2 - 240, button_y, 220, 70)
                switch_rect = pygame.Rect(screen_width//2 + 20, button_y, 220, 70)
                
                if keep_rect.collidepoint(mouse_pos):
                    state, reward, done, info = env.step(0)  # Keep
                    final_reward = reward
                    game_over = True
                
                elif switch_rect.collidepoint(mouse_pos):
                    state, reward, done, info = env.step(1)  # Switch
                    final_reward = reward
                    game_over = True
            
            # Reset
            elif game_over:
                reset_rect = pygame.Rect(screen_width//2 - 100, screen_height - 50, 200, 40)
                if reset_rect.collidepoint(mouse_pos):
                    state = env.reset()
                    game_over = False
                    final_reward = 0
    
    draw_game()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()