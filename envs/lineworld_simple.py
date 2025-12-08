"""
LineWorld Simplifié pour Policy Iteration

Version simplifiée qui utilise seulement la position comme état,
sans énergie, clés, obstacles mobiles, etc.
Parfait pour les algorithmes de Dynamic Programming.
"""

from .utils import BaseEnv
import random

class LineWorldSimple(BaseEnv):
    """
    LineWorld simplifié pour Policy Iteration
    
    - État : seulement la position (0 à length-1)
    - Actions : 0=gauche, 1=droite
    - Obstacles statiques seulement
    - Goal : atteindre la dernière position
    """
    
    def __init__(self, length=10):
        super().__init__()
        self.length = length
        self.start = 0
        self.goal = length - 1
        self.state = self.start
        self.done = False
        
        # Seulement obstacles statiques (simplifié)
        self.obstacles = self._generate_obstacles()
        self.traps = self._generate_traps()
        self.rewards = self._generate_rewards()
        
        self.step_count = 0
        self.max_steps = length * 2
        
    def _generate_obstacles(self):
        """Obstacles statiques (20% de la grille)"""
        obstacles = set()
        num_obstacles = int(self.length * 0.2)
        while len(obstacles) < num_obstacles:
            pos = random.randint(1, self.length-2)
            if pos != self.start and pos != self.goal:
                obstacles.add(pos)
        return obstacles
    
    def _generate_traps(self):
        """Pièges (forte pénalité)"""
        traps = set()
        num_traps = max(1, self.length // 5)
        while len(traps) < num_traps:
            pos = random.randint(1, self.length-2)
            if pos not in self.obstacles and pos != self.start and pos != self.goal:
                traps.add(pos)
        return traps
    
    def _generate_rewards(self):
        """Récompenses à collecter"""
        rewards = {}
        num_rewards = max(1, self.length // 4)
        for _ in range(num_rewards):
            pos = random.randint(1, self.length-2)
            if pos not in self.obstacles and pos not in self.traps:
                rewards[pos] = random.uniform(0.5, 1.5)
        return rewards
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.state = self.start
        self.done = False
        self.step_count = 0
        
        # Régénérer obstacles, pièges, récompenses
        self.obstacles = self._generate_obstacles()
        self.traps = self._generate_traps()
        self.rewards = self._generate_rewards()
        
        return self.state  # Retourne juste la position (int)
    
    def step(self, action):
        """
        Exécute une action
        
        Args:
            action: 0=gauche, 1=droite
        
        Returns:
            state (int): Position actuelle
            reward (float): Récompense
            done (bool): Épisode terminé
            info (dict): Informations supplémentaires
        """
        if self.done:
            return self.state, 0, True, {}
        
        self.step_count += 1
        old_state = self.state
        
        # Actions: 0=gauche, 1=droite
        if action == 0:  # Gauche
            self.state = max(0, self.state - 1)
        elif action == 1:  # Droite
            self.state = min(self.length - 1, self.state + 1)
        else:
            raise ValueError(f"Action invalide: {action}. Doit être 0 ou 1.")
        
        reward = -0.1  # Coût par step
        
        # Vérifier obstacles
        if self.state in self.obstacles:
            self.state = old_state  # Reste sur place
            reward = -0.5
        
        # Vérifier pièges
        if self.state in self.traps:
            reward += -2.0
        
        # Vérifier récompenses
        if self.state in self.rewards:
            reward += self.rewards[self.state]
            del self.rewards[self.state]
        
        # Goal
        if self.state == self.goal:
            reward = 10.0
            self.done = True
        
        # Timeout
        if self.step_count >= self.max_steps:
            self.done = True
            reward = -5.0
        
        return self.state, reward, self.done, {
            'step_count': self.step_count,
            'position': self.state
        }
    
    def sample_action(self):
        """Retourne une action aléatoire"""
        return random.randint(0, 1)
    
    def n_actions(self):
        """Retourne le nombre d'actions possibles"""
        return 2
    
    def get_all_states(self):
        """Retourne tous les états possibles (pour Policy Iteration)"""
        return list(range(self.length))
    
    def get_all_actions(self):
        """Retourne toutes les actions possibles"""
        return [0, 1]
    
    def get_transition_model(self):
        """
        Construit le modèle MDP complet : p[s, a, s', r]
        
        Returns:
            dict: {(s, a, s', r): probability}
        """
        model = {}
        states = self.get_all_states()
        actions = self.get_all_actions()
        
        # Pour chaque état et action, explorer toutes les transitions possibles
        for s in states:
            for a in actions:
                # Sauvegarder l'état actuel
                old_state = self.state
                old_obstacles = self.obstacles.copy()
                old_traps = self.traps.copy()
                old_rewards = self.rewards.copy()
                old_step_count = self.step_count
                
                # Simuler la transition
                self.state = s
                self.step_count = 0
                self.done = False
                
                next_state, reward, done, _ = self.step(a)
                
                # Arrondir la récompense pour créer des catégories
                reward_category = round(reward * 10) / 10  # Arrondir à 0.1 près
                
                # Stocker la transition (déterministe pour cette version simplifiée)
                key = (s, a, next_state, reward_category)
                model[key] = 1.0  # Probabilité 1.0 car déterministe
                
                # Restaurer l'état
                self.state = old_state
                self.obstacles = old_obstacles
                self.traps = old_traps
                self.rewards = old_rewards
                self.step_count = old_step_count
        
        return model

