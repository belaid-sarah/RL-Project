from .utils import BaseEnv
import random

class MontyHallLevel2(BaseEnv):
    def __init__(self):
        super().__init__()
        self.num_doors = 5
        self.doors = [f"D{i}" for i in range(1, self.num_doors + 1)]
        self.step_count = 0
        self.winning_door = None
        self.chosen_door = None
        self.available_doors = None
        self.done = False
        
        # Statistiques
        self.total_reward = 0
        self.games_played = 0
        self.wins = 0
        self.switches = []  # Historique des changements
        
    def reset(self):
        self.winning_door = random.choice(self.doors)
        self.chosen_door = None
        self.available_doors = self.doors.copy()
        self.step_count = 0
        self.done = False
        self.switches = []
        return self._get_state()
    
    def _get_state(self):
        """État enrichi pour l'agent"""
        return {
            "step": self.step_count,
            "chosen_door": self.chosen_door,
            "available_doors": tuple(self.available_doors),
            "num_available": len(self.available_doors),
            "games_played": self.games_played,
            "win_rate": self.wins / max(1, self.games_played)
        }
    
    def step(self, action):
        self.step_count += 1
        reward = 0
        info = {}
        
        if self.step_count < 4:
            # Étapes 1, 2, 3 : sélectionner une porte
            previous_choice = self.chosen_door
            self.chosen_door = self.available_doors[action]
            
            # Enregistrer si l'agent a changé
            if previous_choice is not None:
                switched = (previous_choice != self.chosen_door)
                self.switches.append(switched)
                info["switched"] = switched
            
            # Retirer une porte non gagnante parmi les autres
            non_chosen_non_winning = [
                d for d in self.available_doors 
                if d != self.chosen_door and d != self.winning_door
            ]
            
            if non_chosen_non_winning:
                removed = random.choice(non_chosen_non_winning)
                self.available_doors.remove(removed)
                info["removed"] = removed
            
            info["chosen_door"] = self.chosen_door
            info["available_doors"] = self.available_doors.copy()
            self.done = False
            
        else:
            # Étape 4 (dernière étape) : choix final
            previous_choice = self.chosen_door
            self.chosen_door = self.available_doors[action]
            
            if previous_choice is not None:
                switched = (previous_choice != self.chosen_door)
                self.switches.append(switched)
                info["switched"] = switched
            
            # Calculer la récompense
            reward = 1.0 if self.chosen_door == self.winning_door else 0.0
            
            if reward > 0:
                self.wins += 1
            
            info["final_choice"] = self.chosen_door
            info["winning_door"] = self.winning_door
            info["won"] = reward == 1.0
            info["num_switches"] = sum(self.switches)
            
            self.total_reward += reward
            self.games_played += 1
            self.done = True
        
        return self._get_state(), reward, self.done, info
    
    def sample_action(self):
        return random.randint(0, self.n_actions() - 1)
    
    def n_actions(self):
        if self.available_doors is None:
            return self.num_doors
        return len(self.available_doors)
    
    def get_stats(self):
        """Retourne les statistiques de performance"""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "win_rate": self.wins / max(1, self.games_played),
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward / max(1, self.games_played)
        }