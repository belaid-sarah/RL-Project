import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from envs.rps import TwoRoundRPS

# --- Initialisation Pygame ---
pygame.init()
screen_width = 900
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Two-Round Rock Paper Scissors")

# Polices
font_large = pygame.font.Font(None, 64)
font_medium = pygame.font.Font(None, 40)
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
DARK_GRAY = (60, 60, 60)

# Émojis/Symboles pour RPS
SYMBOLS = {
    0: "✊",  # Rock
    1: "✋",  # Paper
    2: "✌️"   # Scissors
}

NAMES = {
    0: "Rock",
    1: "Paper",
    2: "Scissors"
}

# --- Créer l'environnement ---
env = TwoRoundRPS()
state = env.reset()

# Variables de jeu
waiting_for_action = True
show_round1_result = False
show_round2_result = False
round1_info = None
round2_info = None

def draw_button(x, y, width, height, text, symbol, color, hover=False):
    """Dessine un bouton pour choisir une action"""
    border_color = WHITE if hover else GRAY
    border_width = 4 if hover else 2
    
    # Bouton
    pygame.draw.rect(screen, color, (x, y, width, height), border_radius=15)
    pygame.draw.rect(screen, border_color, (x, y, width, height), border_width, border_radius=15)
    
    # Symbole
    symbol_text = font_large.render(symbol, True, WHITE)
    symbol_rect = symbol_text.get_rect(center=(x + width//2, y + height//2 - 20))
    screen.blit(symbol_text, symbol_rect)
    
    # Nom
    name_text = font_small.render(text, True, WHITE)
    name_rect = name_text.get_rect(center=(x + width//2, y + height - 30))
    screen.blit(name_text, name_rect)

def draw_game():
    screen.fill(BLACK)
    
    # Titre
    title = font_large.render("ROCK PAPER SCISSORS", True, GOLD)
    title_rect = title.get_rect(center=(screen_width//2, 40))
    screen.blit(title, title_rect)
    
    subtitle = font_small.render("Two Rounds Challenge", True, WHITE)
    subtitle_rect = subtitle.get_rect(center=(screen_width//2, 90))
    screen.blit(subtitle, subtitle_rect)
    
    # Informations du round
    if env.round == 0:
        instruction = "ROUND 1: Choose your move!"
        color = WHITE
    elif env.round == 1 and not show_round1_result:
        instruction = "Round 1 in progress..."
        color = BLUE
    elif env.round == 1 and show_round1_result:
        instruction = "ROUND 2: Choose your move!"
        color = GREEN
    elif env.round == 2:
        instruction = "Game Over!"
        color = GOLD
    
    inst_text = font_medium.render(instruction, True, color)
    inst_rect = inst_text.get_rect(center=(screen_width//2, 140))
    screen.blit(inst_text, inst_rect)
    
    # Afficher résultat Round 1 si disponible
    if show_round1_result and round1_info:
        round1_y = 200
        
        round1_title = font_small.render("Round 1 Result:", True, WHITE)
        screen.blit(round1_title, (50, round1_y))
        
        you_text = font_small.render(
            f"You: {SYMBOLS[env.agent_round1_choice]} {NAMES[env.agent_round1_choice]}", 
            True, BLUE
        )
        screen.blit(you_text, (50, round1_y + 35))
        
        opp_text = font_small.render(
            f"Opponent: {SYMBOLS[env.opponent_round1_choice]} {NAMES[env.opponent_round1_choice]}", 
            True, RED
        )
        screen.blit(opp_text, (50, round1_y + 70))
        
        result = round1_info['result']
        result_color = GREEN if result == 'win' else RED if result == 'loss' else GRAY
        result_text = font_small.render(
            f"Result: {result.upper()} ({round1_info['round_reward']:+d})", 
            True, result_color
        )
        screen.blit(result_text, (50, round1_y + 105))
        
        # Hint pour Round 2
        if env.round == 1:
            hint = font_tiny.render(
                f"Hint: Opponent will play: {NAMES[env.agent_round1_choice]}", 
                True, GOLD
            )
            screen.blit(hint, (50, round1_y + 145))
    
    # Afficher résultat Round 2 si disponible
    if show_round2_result and round2_info:
        round2_y = 200 if not show_round1_result else 380
        
        # Résumé complet
        summary_title = font_medium.render("GAME SUMMARY", True, GOLD)
        summary_rect = summary_title.get_rect(center=(screen_width//2, round2_y))
        screen.blit(summary_title, summary_rect)
        
        # Round 1
        r1_text = font_small.render(
            f"Round 1: You {SYMBOLS[env.agent_round1_choice]} vs {SYMBOLS[env.opponent_round1_choice]} Opponent → {env.round1_reward:+d}",
            True, WHITE
        )
        screen.blit(r1_text, (100, round2_y + 50))
        
        # Round 2
        r2_text = font_small.render(
            f"Round 2: You {SYMBOLS[env.agent_round2_choice]} vs {SYMBOLS[env.opponent_round2_choice]} Opponent → {env.round2_reward:+d}",
            True, WHITE
        )
        screen.blit(r2_text, (100, round2_y + 85))
        
        # Total
        total_color = GREEN if env.total_reward > 0 else RED if env.total_reward < 0 else GRAY
        total_text = font_medium.render(
            f"TOTAL: {env.total_reward:+d}",
            True, total_color
        )
        total_rect = total_text.get_rect(center=(screen_width//2, round2_y + 140))
        screen.blit(total_text, total_rect)
        
        # Message final
        if env.total_reward > 0:
            final_msg = "🎉 YOU WIN!"
            final_color = GREEN
        elif env.total_reward < 0:
            final_msg = "😞 YOU LOSE!"
            final_color = RED
        else:
            final_msg = "🤝 DRAW!"
            final_color = GRAY
        
        final_text = font_large.render(final_msg, True, final_color)
        final_rect = final_text.get_rect(center=(screen_width//2, round2_y + 190))
        screen.blit(final_text, final_rect)
    
    # Boutons d'action (si en attente)
    if waiting_for_action and not show_round2_result:
        button_y = 500 if show_round1_result else 300
        button_width = 200
        button_height = 150
        spacing = 50
        total_width = 3 * button_width + 2 * spacing
        start_x = (screen_width - total_width) // 2
        
        mouse_pos = pygame.mouse.get_pos()
        
        # Rock
        rock_rect = pygame.Rect(start_x, button_y, button_width, button_height)
        hover_rock = rock_rect.collidepoint(mouse_pos)
        draw_button(start_x, button_y, button_width, button_height, 
                   "ROCK", "✊", GRAY, hover_rock)
        
        # Paper
        paper_rect = pygame.Rect(start_x + button_width + spacing, button_y, 
                                 button_width, button_height)
        hover_paper = paper_rect.collidepoint(mouse_pos)
        draw_button(start_x + button_width + spacing, button_y, button_width, button_height,
                   "PAPER", "✋", BLUE, hover_paper)
        
        # Scissors
        scissors_rect = pygame.Rect(start_x + 2 * (button_width + spacing), button_y,
                                    button_width, button_height)
        hover_scissors = scissors_rect.collidepoint(mouse_pos)
        draw_button(start_x + 2 * (button_width + spacing), button_y, button_width, button_height,
                   "SCISSORS", "✌️", RED, hover_scissors)
    
    # Bouton Play Again
    if show_round2_result:
        play_again_rect = pygame.Rect(screen_width//2 - 120, screen_height - 80, 240, 50)
        pygame.draw.rect(screen, GOLD, play_again_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, play_again_rect, 3, border_radius=10)
        
        play_text = font_medium.render("PLAY AGAIN", True, BLACK)
        play_rect = play_text.get_rect(center=play_again_rect.center)
        screen.blit(play_text, play_rect)
    
    # Statistiques
    stats = env.get_stats()
    stats_y = screen_height - 40
    stats_text = font_tiny.render(
        f"Games: {stats['games_played']} | Wins: {stats['total_wins']} | Win Rate: {stats['win_rate']*100:.1f}%",
        True, WHITE
    )
    screen.blit(stats_text, (20, stats_y))

# --- Boucle principale ---
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Actions pendant le jeu
            if waiting_for_action and not show_round2_result:
                button_y = 500 if show_round1_result else 300
                button_width = 200
                button_height = 150
                spacing = 50
                total_width = 3 * button_width + 2 * spacing
                start_x = (screen_width - total_width) // 2
                
                # Rock
                rock_rect = pygame.Rect(start_x, button_y, button_width, button_height)
                if rock_rect.collidepoint(mouse_pos):
                    state, reward, done, info = env.step(0)
                    
                    if env.round == 1:
                        round1_info = info
                        show_round1_result = True
                    elif env.round == 2:
                        round2_info = info
                        show_round2_result = True
                        waiting_for_action = False
                
                # Paper
                paper_rect = pygame.Rect(start_x + button_width + spacing, button_y,
                                        button_width, button_height)
                if paper_rect.collidepoint(mouse_pos):
                    state, reward, done, info = env.step(1)
                    
                    if env.round == 1:
                        round1_info = info
                        show_round1_result = True
                    elif env.round == 2:
                        round2_info = info
                        show_round2_result = True
                        waiting_for_action = False
                
                # Scissors
                scissors_rect = pygame.Rect(start_x + 2 * (button_width + spacing), button_y,
                                           button_width, button_height)
                if scissors_rect.collidepoint(mouse_pos):
                    state, reward, done, info = env.step(2)
                    
                    if env.round == 1:
                        round1_info = info
                        show_round1_result = True
                    elif env.round == 2:
                        round2_info = info
                        show_round2_result = True
                        waiting_for_action = False
            
            # Play Again
            if show_round2_result:
                play_again_rect = pygame.Rect(screen_width//2 - 120, screen_height - 80, 240, 50)
                if play_again_rect.collidepoint(mouse_pos):
                    state = env.reset()
                    waiting_for_action = True
                    show_round1_result = False
                    show_round2_result = False
                    round1_info = None
                    round2_info = None
    
    draw_game()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()