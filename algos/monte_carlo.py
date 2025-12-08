from algos.base_agent import BaseAgent
import numpy as np
import random
import time

class MonteCarloES(BaseAgent):
    """
    Monte Carlo Exploring Starts (ES)
    
    Version de Monte Carlo où chaque paire (état, action) a une probabilité non-nulle
    d'être le point de départ d'un épisode.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Monte Carlo ES", gamma=gamma, epsilon=epsilon, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: (state, action) -> value
        self.q_table = {}
        # Returns: (state, action) -> list of returns
        self.returns = {}
        # Policy: state -> action (greedy)
        self.policy = {}
    
    def _get_state_key(self, state):
        """Convertit un état en clé hashable"""
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        else:
            return state
    
    def get_q(self, state, action):
        """Retourne Q(state, action)"""
        state_key = self._get_state_key(state)
        return self.q_table.get((state_key, action), 0.0)
    
    def set_q(self, state, action, value):
        """Met à jour Q(state, action)"""
        state_key = self._get_state_key(state)
        self.q_table[(state_key, action)] = value
    
    def select_action(self, state, training=False):
        """Sélectionne une action selon la politique"""
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        if training:
            # Epsilon-greedy
            if random.random() < self.epsilon:
                return random.choice(actions)
        
        # Greedy selon Q
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, num_episodes, verbose=True):
        """Entraîne l'agent avec Monte Carlo ES"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Exploring Starts: choisir un état et une action aléatoires pour démarrer
            # Pour simplifier, on génère un épisode normalement
            state = self.env.reset()
            actions = self.env.action_space
            
            # Exploring start: action aléatoire au début
            action = random.choice(actions)
            
            # Générer l'épisode
            episode_data = []
            done = False
            total_reward = 0
            
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                
                if not done:
                    # Action suivante selon la politique
                    state = next_state
                    action = self.select_action(state, training=True)
                else:
                    state = next_state
            
            # Calculer les returns et mettre à jour Q
            G = 0
            visited = set()
            
            for state, action, reward in reversed(episode_data):
                G = self.gamma * G + reward
                state_key = self._get_state_key(state)
                
                # First-visit: mettre à jour seulement la première occurrence
                if (state_key, action) not in visited:
                    visited.add((state_key, action))
                    
                    # Ajouter le return
                    if (state_key, action) not in self.returns:
                        self.returns[(state_key, action)] = []
                    self.returns[(state_key, action)].append(G)
                    
                    # Mettre à jour Q avec la moyenne
                    self.set_q(state, action, np.mean(self.returns[(state_key, action)]))
                    
                    # Mettre à jour la politique (greedy)
                    actions = self.env.action_space
                    q_values = [self.get_q(state, a) for a in actions]
                    best_action = max(actions, key=lambda a: self.get_q(state, a))
                    self.policy[state_key] = best_action
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(episode_data))
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")
        
        self.training_time = time.time() - start_time


class OnPolicyMonteCarlo(BaseAgent):
    """
    On-policy First Visit Monte Carlo Control
    
    Version de Monte Carlo où la politique d'exploration (epsilon-greedy)
    est la même que la politique cible.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="On-policy Monte Carlo", gamma=gamma, epsilon=epsilon, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = {}
        self.returns = {}
        self.policy = {}
    
    def _get_state_key(self, state):
        """Convertit un état en clé hashable"""
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        else:
            return state
    
    def get_q(self, state, action):
        """Retourne Q(state, action)"""
        state_key = self._get_state_key(state)
        return self.q_table.get((state_key, action), 0.0)
    
    def set_q(self, state, action, value):
        """Met à jour Q(state, action)"""
        state_key = self._get_state_key(state)
        self.q_table[(state_key, action)] = value
    
    def select_action(self, state, training=False):
        """Sélectionne une action selon epsilon-greedy"""
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        
        # Greedy selon Q
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, num_episodes, verbose=True):
        """Entraîne l'agent avec On-policy Monte Carlo"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Générer un épisode avec la politique epsilon-greedy
            state = self.env.reset()
            episode_data = []
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                state = next_state

            # Calculer les returns (first-visit)
            G = 0
            visited = set()
            
            for state, action, reward in reversed(episode_data):
                G = self.gamma * G + reward
                state_key = self._get_state_key(state)
                
                if (state_key, action) not in visited:
                    visited.add((state_key, action))
                    
                    if (state_key, action) not in self.returns:
                        self.returns[(state_key, action)] = []
                    self.returns[(state_key, action)].append(G)
                    
                    # Mettre à jour Q avec la moyenne
                    self.set_q(state, action, np.mean(self.returns[(state_key, action)]))
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(episode_data))
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")
        
        self.training_time = time.time() - start_time


class OffPolicyMonteCarlo(BaseAgent):
    """
    Off-policy Monte Carlo Control (Weighted Importance Sampling)
    
    Utilise une politique comportementale (b) pour explorer et une politique cible (π) pour apprendre.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Off-policy Monte Carlo", gamma=gamma, epsilon=epsilon, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = {}
        self.C = {}  # Cumulants pour weighted importance sampling
        self.policy = {}  # Politique cible (greedy)
    
    def _get_state_key(self, state):
        """Convertit un état en clé hashable"""
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        else:
            return state
    
    def get_q(self, state, action):
        """Retourne Q(state, action)"""
        state_key = self._get_state_key(state)
        return self.q_table.get((state_key, action), 0.0)
    
    def set_q(self, state, action, value):
        """Met à jour Q(state, action)"""
        state_key = self._get_state_key(state)
        self.q_table[(state_key, action)] = value
    
    def select_action(self, state, training=False):
        """Sélectionne une action selon la politique cible (greedy)"""
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        if training:
            # Politique comportementale: epsilon-greedy
            if random.random() < self.epsilon:
                return random.choice(actions)
        
        # Politique cible: greedy
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def _behavior_policy(self, state):
        """Politique comportementale: epsilon-greedy"""
        actions = self.env.action_space
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def train(self, num_episodes, verbose=True):
        """Entraîne l'agent avec Off-policy Monte Carlo"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Générer un épisode avec la politique comportementale
            state = self.env.reset()
            episode_data = []
            done = False
            total_reward = 0
            
            while not done:
                action = self._behavior_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                state = next_state
            
            # Calculer les returns et mettre à jour avec importance sampling
            G = 0
            W = 1  # Poids d'importance
            
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = self.gamma * G + reward
                state_key = self._get_state_key(state)
                
                # Mettre à jour C et Q avec weighted importance sampling
                if (state_key, action) not in self.C:
                    self.C[(state_key, action)] = 0
                
                self.C[(state_key, action)] += W
                
                # Mettre à jour Q
                current_q = self.get_q(state, action)
                new_q = current_q + (W / self.C[(state_key, action)]) * (G - current_q)
                self.set_q(state, action, new_q)
                
                # Mettre à jour la politique cible (greedy)
                actions = self.env.action_space
                best_action = max(actions, key=lambda a: self.get_q(state, a))
                self.policy[state_key] = best_action
                
                # Si l'action prise n'est pas la meilleure selon la politique cible, arrêter
                if action != best_action:
                    break
                
                # Mettre à jour W (importance sampling ratio)
                # W = W * (1 / (1 - epsilon + epsilon/|A|)) pour epsilon-greedy
                num_actions = len(actions)
                behavior_prob = self.epsilon / num_actions + (1 - self.epsilon) * (1 if action == best_action else 0)
                target_prob = 1.0  # Greedy policy
                W = W * (target_prob / behavior_prob) if behavior_prob > 0 else 0
                
                if W == 0:
                    break
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(episode_data))
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")
        
        self.training_time = time.time() - start_time
