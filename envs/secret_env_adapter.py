"""
Adaptateur pour les environnements secrets (SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3)

Cet adaptateur permet d'utiliser les environnements secrets avec notre interface BaseEnv
et donc avec tous nos algorithmes de RL.
"""

from .utils import BaseEnv
from .secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
import numpy as np


class SecretEnvAdapter(BaseEnv):
    """
    Adaptateur qui convertit l'interface des environnements secrets
    vers notre interface BaseEnv standard.
    """
    
    def __init__(self, secret_env_class, env_name=None):
        """
        Args:
            secret_env_class: La classe de l'environnement secret (SecretEnv0, SecretEnv1, etc.)
            env_name: Nom de l'environnement (pour affichage)
        """
        super().__init__()
        try:
            self.secret_env = secret_env_class()
        except (FileNotFoundError, OSError) as e:
            # Si la DLL n'est pas trouvée, on lève une exception plus claire
            raise FileNotFoundError(
                f"Bibliothèque DLL/SO manquante pour {secret_env_class.__name__}. "
                f"Vérifiez que libs/secret_envs.dll (ou .so/.dylib selon votre OS) existe. "
                f"Erreur originale: {e}"
            )
        self.env_name = env_name or secret_env_class.__name__
        self._last_score = 0.0
        self._current_state = None
        self._step_count = 0
        self._max_steps = 10000  # Limite de sécurité
        
    def reset(self):
        """Réinitialise l'environnement et retourne l'état initial"""
        self.secret_env.reset()
        self._last_score = 0.0
        self._step_count = 0
        self._current_state = self.secret_env.state_id()
        self.done = False
        return self._current_state
    
    def step(self, action):
        """
        Exécute une action et retourne (next_state, reward, done, info)
        
        Args:
            action: L'action à exécuter (int)
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            return self._current_state, 0.0, True, {}
        
        # Vérifier si l'action est valide
        available_actions = self.secret_env.available_actions()
        if action not in available_actions:
            # Action interdite, retourner une pénalité
            return self._current_state, -10.0, False, {'invalid_action': True}
        
        # Sauvegarder le score avant le step
        score_before = self.secret_env.score()
        
        # Exécuter l'action
        self.secret_env.step(int(action))
        self._step_count += 1
        
        # Obtenir le nouveau score
        score_after = self.secret_env.score()
        
        # Calculer la récompense comme la différence de score
        reward = score_after - score_before
        
        # Vérifier si le jeu est terminé
        self.done = self.secret_env.is_game_over()
        
        # Obtenir le nouvel état
        self._current_state = self.secret_env.state_id()
        
        # Limite de sécurité pour éviter les boucles infinies
        if self._step_count >= self._max_steps:
            self.done = True
        
        info = {
            'step_count': self._step_count,
            'total_score': score_after,
            'game_over': self.done
        }
        
        return self._current_state, reward, self.done, info
    
    def sample_action(self):
        """Retourne une action aléatoire parmi les actions disponibles"""
        available_actions = self.secret_env.available_actions()
        if len(available_actions) == 0:
            return 0
        return int(np.random.choice(available_actions))
    
    @property
    def action_space(self):
        """Retourne la liste des actions possibles"""
        # Pour les environnements secrets, on retourne toutes les actions possibles
        # même si certaines peuvent être interdites dans certains états
        num_actions = self.secret_env.num_actions()
        return list(range(num_actions))
    
    def n_actions(self):
        """Retourne le nombre d'actions possibles"""
        return self.secret_env.num_actions()
    
    def get_available_actions(self):
        """Retourne les actions disponibles dans l'état actuel"""
        return self.secret_env.available_actions().tolist()
    
    def display(self):
        """Affiche l'état actuel de l'environnement"""
        self.secret_env.display()
    
    def get_state_id(self):
        """Retourne l'ID de l'état actuel"""
        return self.secret_env.state_id()
    
    def get_score(self):
        """Retourne le score total actuel"""
        return self.secret_env.score()
    
    def is_game_over(self):
        """Vérifie si le jeu est terminé"""
        return self.secret_env.is_game_over()
    
    # Méthodes MDP pour Policy Iteration et Value Iteration
    def get_transition_model(self):
        """
        Construit le modèle MDP complet pour Policy/Value Iteration
        
        Returns:
            dict: {(s, a, s_next, r): probabilité}
        """
        model = {}
        num_states = self.secret_env.num_states()
        num_actions = self.secret_env.num_actions()
        num_rewards = self.secret_env.num_rewards()
        
        # Parcourir toutes les transitions possibles
        for s in range(num_states):
            for a in range(num_actions):
                for s_next in range(num_states):
                    for r_idx in range(num_rewards):
                        prob = self.secret_env.p(s, a, s_next, r_idx)
                        if prob > 0:
                            reward = self.secret_env.reward(r_idx)
                            model[(s, a, s_next, reward)] = prob
        
        return model


# Fonctions de création pour chaque environnement secret
def create_secret_env_0():
    """Crée un adaptateur pour SecretEnv0"""
    return SecretEnvAdapter(SecretEnv0, "SecretEnv0")

def create_secret_env_1():
    """Crée un adaptateur pour SecretEnv1"""
    return SecretEnvAdapter(SecretEnv1, "SecretEnv1")

def create_secret_env_2():
    """Crée un adaptateur pour SecretEnv2"""
    return SecretEnvAdapter(SecretEnv2, "SecretEnv2")

def create_secret_env_3():
    """Crée un adaptateur pour SecretEnv3"""
    return SecretEnvAdapter(SecretEnv3, "SecretEnv3")

