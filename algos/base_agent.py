from abc import ABC, abstractmethod
import numpy as np
import json
import pickle
import time
from pathlib import Path

class BaseAgent(ABC):
    """
    Classe de base pour tous les algorithmes de Reinforcement Learning
    
    Tous les agents doivent hériter de cette classe et implémenter :
    - train() : Entraîner l'agent
    - select_action() : Choisir une action
    - update() : Mettre à jour l'agent (pour les méthodes online)
    """
    
    def __init__(self, env, name="BaseAgent", **kwargs):
        """
        Args:
            env: L'environnement sur lequel l'agent va apprendre
            name: Nom de l'algorithme (ex: "Q-Learning")
            **kwargs: Hyperparamètres spécifiques à l'algorithme
        """
        self.env = env
        self.name = name
        self.hyperparameters = kwargs
        
        # Métriques de performance (collectées pendant l'entraînement)
        self.episode_rewards = []      # Reward total par épisode
        self.episode_lengths = []      # Nombre de steps par épisode
        self.training_time = 0         # Temps d'entraînement total
        self.convergence_episode = None  # Épisode où l'algo a convergé
        
    @abstractmethod
    def train(self, num_episodes, verbose=True):
        """
        Entraîner l'agent pendant num_episodes épisodes
        
        Cette méthode DOIT être implémentée par chaque algorithme
        """
        pass
    
    @abstractmethod
    def select_action(self, state, training=False):
        """
        Sélectionner une action étant donné un état
        
        Args:
            state: L'état actuel
            training: Si True, utilise exploration (epsilon-greedy)
                     Si False, utilise exploitation pure (greedy)
        
        Returns:
            action: L'action choisie
        """
        pass
    
    def update(self, state, action, reward, next_state, done):
        """
        Mettre à jour l'agent (pour les méthodes online comme TD)
        
        Optionnel : certains algos (comme Monte Carlo) n'ont pas besoin
        de mise à jour à chaque step
        """
        pass
    
    def evaluate(self, num_episodes=100, max_steps=1000, verbose=False):
        """
        Évaluer les performances de l'agent
        
        Cette méthode exécute l'agent en mode greedy (pas d'exploration)
        pour mesurer ses vraies performances
        
        Args:
            num_episodes: Nombre d'épisodes de test
            max_steps: Nombre maximum de steps par épisode
            verbose: Si True, affiche les détails
        
        Returns:
            dict: Dictionnaire avec toutes les métriques
        """
        total_rewards = []
        total_steps = []
        success_count = 0
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            # Convertir état en format uniforme
            if isinstance(state, dict):
                state = state
            
            while not done and steps < max_steps:
                # Mode greedy (pas d'exploration)
                action = self.select_action(state, training=False)
                
                # Exécuter l'action
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            # Considérer comme succès si le goal est atteint
            # Vérifier dans l'état final ou dans info
            goal_reached = False
            if done:
                # Vérifier si l'état final est le goal
                if hasattr(self.env, 'goal'):
                    if hasattr(self.env, 'state'):
                        if self.env.state == self.env.goal:
                            goal_reached = True
                    # Alternative: vérifier dans info si disponible
                    if 'goal_reached' in info:
                        goal_reached = info['goal_reached']
            
            # Fallback: considérer comme succès si reward positif ET done=True
            # (pour les environnements qui n'ont pas de goal explicite)
            if not goal_reached and done and episode_reward > 0:
                goal_reached = True
            
            if goal_reached:
                success_count += 1
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"  Eval Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}")
        
        # Calculer les statistiques
        results = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps),
            'success_rate': success_count / num_episodes,
            'total_episodes': num_episodes
        }
        
        return results
    
    def save(self, path):
        """
        Sauvegarder l'agent (métriques + état)
        
        Args:
            path: Chemin du fichier de sauvegarde
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Préparer les données à sauvegarder
        save_data = {
            'name': self.name,
            'hyperparameters': self.hyperparameters,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': self.training_time,
            'convergence_episode': self.convergence_episode
        }
        
        # Sauvegarder en pickle (permet de sauvegarder n'importe quel objet Python)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[OK] Agent sauvegarde : {path}")
    
    def load(self, path):
        """
        Charger un agent sauvegardé
        
        Args:
            path: Chemin du fichier de sauvegarde
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.name = save_data['name']
        self.hyperparameters = save_data['hyperparameters']
        self.episode_rewards = save_data['episode_rewards']
        self.episode_lengths = save_data['episode_lengths']
        self.training_time = save_data['training_time']
        self.convergence_episode = save_data.get('convergence_episode')
        
        print(f"[OK] Agent charge : {path}")
    
    def _get_state_key(self, state):
        """
        Convertir un état en clé pour un dictionnaire
        
        Pourquoi ? Les états peuvent être des tuples, listes, dicts...
        On a besoin d'une représentation uniforme pour indexer nos structures
        
        Args:
            state: L'état à convertir
        
        Returns:
            Une clé hashable (qui peut être utilisée dans un dict)
        """
        if isinstance(state, dict):
            # Si état est un dict, on le convertit en tuple trié
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        else:
            return state
    
    def get_training_stats(self):
        """
        Retourne un résumé des statistiques d'entraînement
        
        Returns:
            dict: Statistiques d'entraînement
        """
        if not self.episode_rewards:
            return None
        
        # Calculer la moyenne sur les 100 derniers épisodes
        recent_rewards = self.episode_rewards[-100:]
        
        return {
            'total_episodes': len(self.episode_rewards),
            'final_mean_reward': np.mean(recent_rewards),
            'final_std_reward': np.std(recent_rewards),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards),
            'training_time': self.training_time,
            'convergence_episode': self.convergence_episode
        }
