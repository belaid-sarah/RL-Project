from .utils import BaseEnv
import random

class TwoRoundRPS(BaseEnv):
    """
    Two-Round Rock Paper Scissors
    
    L'agent joue 2 rounds de Pierre-Papier-Ciseaux.
    Round 1: Adversaire joue aléatoirement
    Round 2: Adversaire joue le choix de l'agent au Round 1
    
    Actions:
        0 = Rock (Pierre)
        1 = Paper (Papier)
        2 = Scissors (Ciseaux)
    
    Rewards:
        +1 = Win
         0 = Draw
        -1 = Loss
    """
    
    def __init__(self):
        super().__init__()
        self.actions = ["Rock", "Paper", "Scissors"]
        self.round = 0  # 0 = pas commencé, 1 = round 1, 2 = round 2
        self.agent_round1_choice = None
        self.opponent_round1_choice = None
        self.agent_round2_choice = None
        self.opponent_round2_choice = None
        self.round1_reward = 0
        self.round2_reward = 0
        self.total_reward = 0
        self.done = False
        
        # Statistiques
        self.games_played = 0
        self.total_wins = 0
        self.round1_wins = 0
        self.round2_wins = 0
        
    def reset(self):
        """Réinitialise l'environnement"""
        self.round = 0
        self.agent_round1_choice = None
        self.opponent_round1_choice = None
        self.agent_round2_choice = None
        self.opponent_round2_choice = None
        self.round1_reward = 0
        self.round2_reward = 0
        self.total_reward = 0
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """Retourne l'état actuel pour l'agent"""
        return {
            'round': self.round,
            'agent_round1': self.agent_round1_choice,
            'opponent_round1': self.opponent_round1_choice,
            'round1_reward': self.round1_reward
        }
    
    def _get_winner(self, action1, action2):
        """
        Détermine le gagnant
        Returns: 1 si action1 gagne, -1 si action2 gagne, 0 si égalité
        """
        if action1 == action2:
            return 0  # Draw
        
        # Rock beats Scissors
        # Paper beats Rock
        # Scissors beats Paper
        wins = {
            (0, 2): 1,  # Rock beats Scissors
            (1, 0): 1,  # Paper beats Rock
            (2, 1): 1,  # Scissors beats Paper
            (2, 0): -1, # Scissors loses to Rock
            (0, 1): -1, # Rock loses to Paper
            (1, 2): -1  # Paper loses to Scissors
        }
        
        return wins.get((action1, action2), 0)
    
    def step(self, action):
        """
        Exécute une action
        
        Args:
            action (int): 0=Rock, 1=Paper, 2=Scissors
        
        Returns:
            state, reward, done, info
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")
        
        if action not in [0, 1, 2]:
            raise ValueError("Action must be 0 (Rock), 1 (Paper), or 2 (Scissors)")
        
        info = {}
        reward = 0
        
        # Round 1
        if self.round == 0:
            self.round = 1
            self.agent_round1_choice = action
            
            # Adversaire joue aléatoirement
            self.opponent_round1_choice = random.randint(0, 2)
            
            # Calculer le résultat du round 1
            result = self._get_winner(self.agent_round1_choice, self.opponent_round1_choice)
            self.round1_reward = result
            reward = result
            
            if result == 1:
                self.round1_wins += 1
            
            info['round'] = 1
            info['agent_choice'] = self.actions[self.agent_round1_choice]
            info['opponent_choice'] = self.actions[self.opponent_round1_choice]
            info['result'] = 'win' if result == 1 else 'loss' if result == -1 else 'draw'
            info['round_reward'] = result
            
            self.done = False
        
        # Round 2
        elif self.round == 1:
            self.round = 2
            self.agent_round2_choice = action
            
            # Adversaire joue le choix de l'agent au Round 1
            self.opponent_round2_choice = self.agent_round1_choice
            
            # Calculer le résultat du round 2
            result = self._get_winner(self.agent_round2_choice, self.opponent_round2_choice)
            self.round2_reward = result
            reward = result
            
            if result == 1:
                self.round2_wins += 1
            
            # Reward total de l'épisode
            self.total_reward = self.round1_reward + self.round2_reward
            
            # Statistiques
            self.games_played += 1
            if self.total_reward > 0:
                self.total_wins += 1
            
            info['round'] = 2
            info['agent_choice'] = self.actions[self.agent_round2_choice]
            info['opponent_choice'] = self.actions[self.opponent_round2_choice]
            info['result'] = 'win' if result == 1 else 'loss' if result == -1 else 'draw'
            info['round_reward'] = result
            info['total_reward'] = self.total_reward
            info['round1_reward'] = self.round1_reward
            info['round2_reward'] = self.round2_reward
            
            self.done = True
        
        return self._get_state(), reward, self.done, info
    
    def sample_action(self):
        """Retourne une action aléatoire"""
        return random.randint(0, 2)
    
    def n_actions(self):
        """Retourne le nombre d'actions possibles"""
        return 3
    
    def get_stats(self):
        """Retourne les statistiques de jeu"""
        return {
            'games_played': self.games_played,
            'total_wins': self.total_wins,
            'win_rate': self.total_wins / max(1, self.games_played),
            'round1_wins': self.round1_wins,
            'round1_win_rate': self.round1_wins / max(1, self.games_played),
            'round2_wins': self.round2_wins,
            'round2_win_rate': self.round2_wins / max(1, self.games_played)
        }
    
    def render(self):
        """Affichage texte de l'état"""
        print(f"\n{'='*50}")
        print(f"Two-Round Rock Paper Scissors")
        print(f"{'='*50}")
        
        if self.round == 0:
            print("Ready to start! Choose your action for Round 1")
            print("Actions: 0=Rock, 1=Paper, 2=Scissors")
        
        elif self.round == 1:
            print(f"\n📍 ROUND 1 COMPLETED")
            print(f"  You played: {self.actions[self.agent_round1_choice]}")
            print(f"  Opponent played: {self.actions[self.opponent_round1_choice]}")
            print(f"  Result: {'+1' if self.round1_reward == 1 else '0' if self.round1_reward == 0 else '-1'}")
            print(f"\nChoose your action for Round 2")
            print(f"Hint: Opponent will play '{self.actions[self.agent_round1_choice]}'")
        
        elif self.round == 2:
            print(f"\n📍 ROUND 1")
            print(f"  You: {self.actions[self.agent_round1_choice]} vs Opponent: {self.actions[self.opponent_round1_choice]}")
            print(f"  Reward: {self.round1_reward:+d}")
            
            print(f"\n📍 ROUND 2")
            print(f"  You: {self.actions[self.agent_round2_choice]} vs Opponent: {self.actions[self.opponent_round2_choice]}")
            print(f"  Reward: {self.round2_reward:+d}")
            
            print(f"\nTOTAL REWARD: {self.total_reward:+d}")
            
            if self.total_reward > 0:
                print("[WIN] You WIN the game!")
            elif self.total_reward < 0:
                print("[LOSE] You LOSE the game!")
            else:
                print("🤝 DRAW game!")
        
        print(f"{'='*50}\n")