"""
GridWorld Simplifié - Inspiré de Sutton & Barto

Environnement simple avec rewards standards :
- -1 par step (coût de mouvement)
- +1 pour atteindre le goal
- Obstacles statiques seulement
- Peut être agrandi facilement
"""

from .utils import BaseEnv
import random

class GridWorldSimple(BaseEnv):
    """
    GridWorld simplifié style Sutton & Barto
    
    Rewards standards :
    - -1 par step (coût de mouvement)
    - +1 pour goal
    - 0 pour obstacles (reste sur place)
    """
    
    def __init__(self, width=5, height=5):
        super().__init__()
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width-1, height-1)
        self.state = self.start
        self.done = False
        
        # Obstacles statiques seulement (10-15% de la grille)
        self.obstacles = self._generate_obstacles()
        
        self.step_count = 0
        self.max_steps = width * height * 3  # Limite raisonnable
    
    def _generate_obstacles(self):
        """Génère des obstacles statiques (10-15% de la grille)"""
        obstacles = set()
        num_obstacles = int(self.width * self.height * 0.12)
        
        while len(obstacles) < num_obstacles:
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if pos != self.start and pos != self.goal:
                obstacles.add(pos)
        
        return obstacles
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.state = self.start
        self.done = False
        self.step_count = 0
        
        # Régénérer obstacles (optionnel : garder les mêmes)
        # self.obstacles = self._generate_obstacles()
        
        return self.state
    
    def step(self, action):
        """
        Exécute une action
        
        Actions: 0=haut, 1=bas, 2=gauche, 3=droite
        
        Rewards standards (Sutton & Barto) :
        - -1 par step
        - +1 pour goal
        - 0 si obstacle (reste sur place)
        """
        if self.done:
            return self.state, 0, True, {}
        
        self.step_count += 1
        x, y = self.state
        
        # Déplacement
        if action == 0:  # Haut
            y = max(0, y - 1)
        elif action == 1:  # Bas
            y = min(self.height - 1, y + 1)
        elif action == 2:  # Gauche
            x = max(0, x - 1)
        elif action == 3:  # Droite
            x = min(self.width - 1, x + 1)
        else:
            raise ValueError(f"Action invalide: {action}")
        
        new_state = (x, y)
        
        # Reward standard : -1 par step
        reward = -1.0
        
        # Vérifier obstacle
        if new_state in self.obstacles:
            # Reste sur place, reward = 0 (pas de coût)
            new_state = self.state
            reward = 0.0
        else:
            self.state = new_state
        
        # Goal : +1 (reward standard)
        if self.state == self.goal:
            reward = 1.0
            self.done = True
        
        # Timeout
        if self.step_count >= self.max_steps:
            self.done = True
        
        return self.state, reward, self.done, {
            'step_count': self.step_count
        }
    
    def sample_action(self):
        """Retourne une action aléatoire"""
        return random.randint(0, 3)
    
    def n_actions(self):
        """Retourne le nombre d'actions possibles"""
        return 4
    
    def get_all_states(self):
        """Retourne tous les états possibles"""
        return [(x, y) for x in range(self.width) for y in range(self.height)]
    
    def get_all_actions(self):
        """Retourne toutes les actions possibles"""
        return [0, 1, 2, 3]

