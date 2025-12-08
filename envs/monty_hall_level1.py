from .utils import BaseEnv
import random

class MontyHallLevel1(BaseEnv):
    def __init__(self):
        super().__init__()
        self.doors = ["A", "B", "C"]
        self.step_count = 0
        self.winning_door = None
        self.chosen_door = None
        self.remaining_door = None
        self.removed_door = None
        self.done = False
        
        # Statistiques
        self.total_reward = 0
        self.games_played = 0
        self.wins = 0
        self.switches = 0

    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.chosen_door = None
        self.remaining_door = None
        self.removed_door = None
        self.step_count = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """État enrichi pour l'agent"""
        return {
            "step": self.step_count,
            "chosen_door": self.chosen_door,
            "remaining_door": self.remaining_door,
            "removed_door": self.removed_door,
            "games_played": self.games_played,
            "win_rate": self.wins / max(1, self.games_played)
        }

    def step(self, action):
        self.step_count += 1
        reward = 0
        info = {}
        
        # Étape 1 : Choix initial (action = 0:"A", 1:"B", 2:"C")
        if self.step_count == 1:
            if action < 0 or action > 2:
                raise ValueError("Action doit être 0, 1 ou 2 pour A, B ou C")
            
            self.chosen_door = self.doors[action]
            
            # L'hôte retire une porte perdante
            available = [d for d in self.doors if d != self.chosen_door and d != self.winning_door]
            self.removed_door = random.choice(available)
            
            # La porte restante
            self.remaining_door = [d for d in self.doors if d not in [self.chosen_door, self.removed_door]][0]
            
            info["chosen_door"] = self.chosen_door
            info["removed_door"] = self.removed_door
            info["remaining_door"] = self.remaining_door
            
            self.done = False
        
        # Étape 2 : Décision finale (action = 0:garder, 1:changer)
        else:
            if action < 0 or action > 1:
                raise ValueError("Action doit être 0 (garder) ou 1 (changer)")
            
            if action == 0:  # Garder
                final_choice = self.chosen_door
                info["action"] = "kept"
            else:  # Changer
                final_choice = self.remaining_door
                info["action"] = "switched"
                self.switches += 1
            
            # Calculer la récompense
            if final_choice == self.winning_door:
                reward = 1.0
                self.wins += 1
                info["won"] = True
            else:
                reward = 0.0
                info["won"] = False
            
            info["final_choice"] = final_choice
            info["winning_door"] = self.winning_door
            
            self.total_reward += reward
            self.games_played += 1
            self.done = True
        
        return self._get_state(), reward, self.done, info

    def sample_action(self):
        if self.step_count == 0:
            return random.randint(0, 2)  # Choisir A, B ou C
        else:
            return random.randint(0, 1)  # Garder ou changer

    def n_actions(self):
        if self.step_count == 0:
            return 3  # 3 portes à l'étape 1
        else:
            return 2  # Garder ou changer à l'étape 2
    
    def get_stats(self):
        """Retourne les statistiques de performance"""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.games_played),
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.games_played),
            "switch_rate": self.switches / max(1, self.games_played)
        }