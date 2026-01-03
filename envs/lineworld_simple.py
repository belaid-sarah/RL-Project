"""
LineWorld Simplifié Enrichi - Inspiré de Sutton & Barto

Environnement avec mécaniques supplémentaires :
- -1 par step (coût de mouvement)
- +1 pour atteindre le goal
- Obstacles statiques
- Récompenses intermédiaires (bonus)
- Pièges (pénalités)
- Zones de bonus
"""

from .utils import BaseEnv
import random

class LineWorldSimple(BaseEnv):
    """
    LineWorld simplifié enrichi style Sutton & Barto
    
    Rewards :
    - -1 par step (coût de mouvement)
    - +1 pour goal
    - 0 pour obstacles (reste sur place)
    - +0.5 pour récompenses intermédiaires
    - -2 pour pièges
    - +0.3 pour zones de bonus
    """
    
    def __init__(self, length=25):
        """
        Initialise l'environnement LineWorld
        
        Args:
            length: Longueur de la ligne (nombre de positions)
                   Recommandé: 15-30 pour un bon équilibre complexité/performance
                   Max recommandé: ~50 pour Dynamic Programming, ~100+ pour TD methods
        """
        super().__init__()
        self.length = length
        self.start = 0
        self.goal = length - 1
        self.state = self.start
        self.done = False
        
        # Pas d'obstacles pour garantir un chemin vers le goal
        # Les obstacles étaient source de blocages rendant le goal inaccessible
        self.obstacles = set()
        
        # Récompenses intermédiaires (positions avec bonus)
        self.rewards_cells = self._generate_rewards()
        
        # Pièges (positions avec pénalités)
        self.traps = self._generate_traps()
        
        # Zones de bonus (petites récompenses)
        self.bonus_zones = self._generate_bonus_zones()
        
        self.step_count = 0
        # Max steps adaptatif : plus de temps pour les environnements plus grands
        self.max_steps = length * 4  # Augmenté de 3 à 4 pour plus de flexibilité
        self.visited_rewards = set()  # Pour ne collecter qu'une fois
    
    def _generate_obstacles(self):
        """
        Génère des obstacles statiques
        
        NOTE: Actuellement désactivé (retourne set vide) pour garantir un chemin vers le goal.
        Les obstacles causaient des blocages rendant le goal inaccessible.
        """
        # Retourner un set vide = pas d'obstacles
        return set()
    
    def _generate_rewards(self):
        """Génère des récompenses intermédiaires (bonus de +0.5)"""
        rewards = {}
        num_rewards = max(1, self.length // 5)  # ~20% des positions
        
        for _ in range(num_rewards):
            pos = random.randint(1, self.length - 2)
            # Éviter obstacles, start, goal
            if (pos not in self.obstacles and 
                pos != self.start and pos != self.goal):
                rewards[pos] = 0.5
        
        return rewards
    
    def _generate_traps(self):
        """Génère des pièges (pénalités de -2)"""
        traps = set()
        num_traps = max(1, self.length // 6)  # ~15% des positions
        
        while len(traps) < num_traps:
            pos = random.randint(1, self.length - 2)
            # Éviter obstacles, rewards, start, goal
            if (pos not in self.obstacles and 
                pos not in self.rewards_cells and
                pos != self.start and pos != self.goal):
                traps.add(pos)
        
        return traps
    
    def _generate_bonus_zones(self):
        """Génère des zones de bonus (petites récompenses de +0.3)"""
        bonus = {}
        num_bonus = max(1, self.length // 4)  # ~25% des positions
        
        for _ in range(num_bonus):
            pos = random.randint(1, self.length - 2)
            # Éviter obstacles, rewards, traps, start, goal
            if (pos not in self.obstacles and 
                pos not in self.rewards_cells and
                pos not in self.traps and
                pos != self.start and pos != self.goal):
                bonus[pos] = 0.3
        
        return bonus
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.state = self.start
        self.done = False
        self.step_count = 0
        self.visited_rewards = set()
        
        # Régénérer obstacles et éléments (optionnel)
        # self.obstacles = self._generate_obstacles()
        # self.rewards_cells = self._generate_rewards()
        # self.traps = self._generate_traps()
        # self.bonus_zones = self._generate_bonus_zones()
        
        return self.state
    
    def step(self, action):
        """
        Exécute une action
        
        Actions: 0=gauche, 1=droite
        
        Rewards :
        - -1 par step (coût de mouvement)
        - +1 pour goal
        - 0 si obstacle (reste sur place)
        - +0.5 pour récompenses intermédiaires (une seule fois)
        - -2 pour pièges
        - +0.3 pour zones de bonus (une seule fois)
        """
        if self.done:
            return self.state, 0, True, {}
        
        self.step_count += 1
        old_state = self.state
        
        # Déplacement
        if action == 0:  # Gauche
            self.state = max(0, self.state - 1)
        elif action == 1:  # Droite
            self.state = min(self.length - 1, self.state + 1)
        else:
            raise ValueError(f"Action invalide: {action}")
        
        # Goal : reward finale positive (PRIORITÉ)
        if self.state == self.goal:
            reward = 10.0  # Reward finale positive standard
            self.done = True
            return self.state, reward, self.done, {
                'step_count': self.step_count,
                'goal_reached': True
            }
        
        # Reward de base : -1 par step
        reward = -1.0
        
        # Obstacles supprimés pour garantir un chemin vers le goal
        # (code conservé au cas où on voudrait réintroduire les obstacles plus tard)
        # if self.state in self.obstacles:
        #     self.state = old_state
        #     reward = -1.0
        #     return self.state, reward, self.done, {
        #         'step_count': self.step_count,
        #         'hit_obstacle': True,
        #         'goal_reached': False
        #     }
        
        # Vérifier piège (pénalité forte) - peut être combiné avec autres éléments
        if self.state in self.traps:
            reward += -2.0
        
        # Vérifier récompense intermédiaire (collectable une fois)
        if self.state in self.rewards_cells and self.state not in self.visited_rewards:
            reward += self.rewards_cells[self.state]
            self.visited_rewards.add(self.state)
        
        # Vérifier zone de bonus (collectable une fois)
        if self.state in self.bonus_zones and self.state not in self.visited_rewards:
            reward += self.bonus_zones[self.state]
            self.visited_rewards.add(self.state)
        
        # Timeout
        if self.step_count >= self.max_steps:
            self.done = True
        
        return self.state, reward, self.done, {
            'step_count': self.step_count,
            'in_trap': self.state in self.traps,
            'got_reward': self.state in self.rewards_cells and self.state in self.visited_rewards,
            'got_bonus': self.state in self.bonus_zones and self.state in self.visited_rewards,
            'goal_reached': self.state == self.goal  # Indicateur explicite pour le success
        }
    
    def sample_action(self):
        """Retourne une action aléatoire"""
        return random.randint(0, 1)
    
    def n_actions(self):
        """Retourne le nombre d'actions possibles"""
        return 2
    
    def get_all_states(self):
        """Retourne tous les états possibles"""
        return list(range(self.length))
    
    def get_all_actions(self):
        """Retourne toutes les actions possibles"""
        return [0, 1]
    
    def get_transition_model(self):
        """
        Construit le modèle MDP complet : p[s, a, s', r]
        Note: Les récompenses peuvent varier selon si elles ont été collectées
        """
        model = {}
        states = self.get_all_states()
        actions = self.get_all_actions()
        
        for s in states:
            for a in actions:
                # Sauvegarder
                old_state = self.state
                old_obstacles = self.obstacles.copy()
                old_rewards = self.rewards_cells.copy()
                old_traps = self.traps.copy()
                old_bonus = self.bonus_zones.copy()
                old_step_count = self.step_count
                old_visited = self.visited_rewards.copy()
                
                # Simuler (première visite = récompense disponible)
                self.state = s
                self.step_count = 0
                self.done = False
                self.visited_rewards = set()  # Reset pour simulation
                
                next_state, reward, done, _ = self.step(a)
                # Arrondir pour simplifier le modèle (optionnel)
                reward = round(reward * 2) / 2  # Arrondir à 0.5 près
                
                # Stocker
                key = (s, a, next_state, reward)
                model[key] = 1.0
                
                # Restaurer
                self.state = old_state
                self.obstacles = old_obstacles
                self.rewards_cells = old_rewards
                self.traps = old_traps
                self.bonus_zones = old_bonus
                self.step_count = old_step_count
                self.visited_rewards = old_visited
        
        return model
    
    def get_environment_info(self):
        """Retourne des informations sur l'environnement (pour visualisation)"""
        return {
            'length': self.length,
            'start': self.start,
            'goal': self.goal,
            'obstacles': sorted(list(self.obstacles)),
            'rewards': self.rewards_cells,
            'traps': sorted(list(self.traps)),
            'bonus_zones': self.bonus_zones
        }
