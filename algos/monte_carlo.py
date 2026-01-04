"""
Monte Carlo Methods - Sutton & Barto Chapter 5

Algorithmes du livre :
- Monte Carlo ES (5.3)
- On-policy First Visit MC Control (5.4)
- Off-policy MC Control (5.6)
"""

from algos.base_agent import BaseAgent
import random
import time

class MonteCarloES(BaseAgent):
    """
    Monte Carlo Exploring Starts (Sutton & Barto 5.3)
    """
    
    def __init__(self, env, gamma=0.99, **kwargs):
        super().__init__(env, name="Monte Carlo ES", gamma=gamma, **kwargs)
        self.gamma = gamma
        
        # Q-table: Q(s, a)
        self.Q = {}
        # Returns: liste des returns pour chaque (s, a)
        self.returns = {}
        # Policy: π(s) = a
        self.policy = {}
    
    def _get_state_key(self, state):
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, tuple)):
            return tuple(state) if not (len(state) > 0 and isinstance(state[0], tuple)) else state
        else:
            return state
    
    def get_q(self, state, action):
        state_key = self._get_state_key(state)
        return self.Q.get((state_key, action), 0.0)
    
    def select_action(self, state, training=False):
        """Sélectionne action selon la politique (greedy)"""
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        if state_key in self.policy:
            return self.policy[state_key]
        return random.choice(actions)
    
    def train(self, num_episodes, verbose=True):
        """
        Monte Carlo ES (Sutton & Barto 5.3)
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Générer épisode avec exploring starts
            state = self.env.reset()
            actions = self.env.action_space
            action = random.choice(actions)  # Exploring start
            
            episode_data = []
            done = False
            total_reward = 0
            max_steps_per_episode = 1000  # Limite pour éviter les épisodes trop longs
            
            step_count = 0
            while not done and step_count < max_steps_per_episode:
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                step_count += 1
                
                if not done:
                    state = next_state
                    action = self.select_action(state, training=True)
                else:
                    state = next_state
            
            # Forcer done si on a atteint la limite
            if step_count >= max_steps_per_episode:
                done = True
            
            # Calculer returns (first-visit)
            G = 0
            visited = set()
            
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = self.gamma * G + reward
                state_key = self._get_state_key(state)
                
                if (state_key, action) not in visited:
                    visited.add((state_key, action))
                    
                    # Ajouter return
                    if (state_key, action) not in self.returns:
                        self.returns[(state_key, action)] = []
                    self.returns[(state_key, action)].append(G)
                    
                    # Mettre à jour Q avec moyenne
                    self.Q[(state_key, action)] = sum(self.returns[(state_key, action)]) / len(self.returns[(state_key, action)])
                    
                    # Mettre à jour policy (greedy)
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
    On-policy First Visit MC Control (Sutton & Barto 5.4)
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="On-policy Monte Carlo", gamma=gamma, epsilon=epsilon, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = {}
        self.returns = {}
        self.policy = {}
    
    def _get_state_key(self, state):
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, tuple)):
            return tuple(state) if not (len(state) > 0 and isinstance(state[0], tuple)) else state
        else:
            return state
    
    def get_q(self, state, action):
        state_key = self._get_state_key(state)
        return self.Q.get((state_key, action), 0.0)
    
    def select_action(self, state, training=False):
        """Epsilon-greedy"""
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
        """
        On-policy First Visit MC Control (Sutton & Barto 5.4)
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Générer épisode avec epsilon-greedy
            state = self.env.reset()
            episode_data = []
            done = False
            total_reward = 0
            max_steps_per_episode = 1000  # Limite pour éviter les épisodes trop longs
            
            step_count = 0
            while not done and step_count < max_steps_per_episode:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                state = next_state
                step_count += 1
            
            # Forcer done si on a atteint la limite
            if step_count >= max_steps_per_episode:
                done = True
            
            # Calculer returns (first-visit)
            G = 0
            visited = set()
            
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = self.gamma * G + reward
                state_key = self._get_state_key(state)
                
                if (state_key, action) not in visited:
                    visited.add((state_key, action))
                    
                    if (state_key, action) not in self.returns:
                        self.returns[(state_key, action)] = []
                    self.returns[(state_key, action)].append(G)
                    
                    # Mettre à jour Q avec moyenne
                    self.Q[(state_key, action)] = sum(self.returns[(state_key, action)]) / len(self.returns[(state_key, action)])
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(episode_data))
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")
        
        self.training_time = time.time() - start_time


class OffPolicyMonteCarlo(BaseAgent):
    """
    Off-policy MC Control avec Weighted Importance Sampling (Sutton & Barto 5.6)
    """
    
    def __init__(self, env, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Off-policy Monte Carlo", gamma=gamma, epsilon=epsilon, **kwargs)
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = {}
        self.C = {}  # Cumulants pour weighted importance sampling
        self.policy = {}  # Politique cible (greedy)
    
    def _get_state_key(self, state):
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, tuple)):
            return tuple(state) if not (len(state) > 0 and isinstance(state[0], tuple)) else state
        else:
            return state
    
    def get_q(self, state, action):
        state_key = self._get_state_key(state)
        return self.Q.get((state_key, action), 0.0)
    
    def select_action(self, state, training=False):
        """Politique cible (greedy)"""
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
        """
        Off-policy MC Control (Sutton & Barto 5.6)
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Générer épisode avec politique comportementale
            state = self.env.reset()
            episode_data = []
            done = False
            total_reward = 0
            max_steps_per_episode = 1000  # Limite pour éviter les épisodes trop longs
            
            step_count = 0
            while not done and step_count < max_steps_per_episode:
                action = self._behavior_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_data.append((state, action, reward))
                total_reward += reward
                state = next_state
                step_count += 1
            
            # Forcer done si on a atteint la limite
            if step_count >= max_steps_per_episode:
                done = True
            
            # Calculer returns avec importance sampling
            G = 0
            W = 1.0
            
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = self.gamma * G + reward
                state_key = self._get_state_key(state)
                actions = self.env.action_space
                
                # Mettre à jour C et Q (weighted importance sampling)
                if (state_key, action) not in self.C:
                    self.C[(state_key, action)] = 0.0
                
                self.C[(state_key, action)] += W
                
                # Mettre à jour Q
                current_q = self.get_q(state, action)
                new_q = current_q + (W / self.C[(state_key, action)]) * (G - current_q)
                self.Q[(state_key, action)] = new_q
                
                # Mettre à jour policy cible (greedy)
                best_action = max(actions, key=lambda a: self.get_q(state, a))
                self.policy[state_key] = best_action
                
                # Si action != best_action, arrêter (importance sampling)
                if action != best_action:
                    break
                
                # Mettre à jour W
                num_actions = len(actions)
                behavior_prob = self.epsilon / num_actions + (1 - self.epsilon) * (1.0 if action == best_action else 0.0)
                target_prob = 1.0  # Greedy
                W = W * (target_prob / behavior_prob) if behavior_prob > 0 else 0.0
                
                if W == 0:
                    break
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(episode_data))
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}")
        
        self.training_time = time.time() - start_time
