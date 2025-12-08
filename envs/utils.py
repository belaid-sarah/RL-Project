import random

class BaseEnv:
    def __init__(self):
        self.done = False

    def reset(self):
        """Reset l'environnement et retourner l'état initial"""
        self.done = False
        return None

    def step(self, action):
        """
        Prend une action et retourne (next_state, reward, done, info)
        """
        raise NotImplementedError("step() must be implemented by subclass")

    def sample_action(self):
        """Retourne une action aléatoire possible"""
        raise NotImplementedError("sample_action() must be implemented by subclass")
    
    @property
    def action_space(self):
        """Retourne la liste des actions possibles"""
        if hasattr(self, 'n_actions'):
            return list(range(self.n_actions()))
        raise NotImplementedError("action_space or n_actions() must be implemented by subclass")