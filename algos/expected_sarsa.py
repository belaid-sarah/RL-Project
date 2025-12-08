from algos.base_agent import BaseAgent
import numpy as np
import random
import time

class ExpectedSARSAAgent(BaseAgent):
    """
    Expected SARSA
    
    Variante de SARSA qui utilise l'espérance de Q(s',a') sur toutes les actions
    possibles selon la politique, plutôt que la valeur de l'action suivante.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Expected SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: (state, action) -> value
        self.q_table = {}
    
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
    
    def _get_expected_q(self, state):
        """Calcule l'espérance de Q(s',a') selon la politique epsilon-greedy"""
        actions = self.env.action_space
        num_actions = len(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        
        # Probabilité de chaque action selon epsilon-greedy
        # Action greedy: (1 - epsilon) + epsilon/num_actions
        # Autres actions: epsilon/num_actions
        expected_value = 0.0
        for action in actions:
            if action in best_actions:
                prob = (1 - self.epsilon) / len(best_actions) + self.epsilon / num_actions
            else:
                prob = self.epsilon / num_actions
            expected_value += prob * self.get_q(state, action)
        
        return expected_value
    
    def update(self, state, action, reward, next_state, done):
        """Met à jour Q avec la formule Expected SARSA"""
        state_key = self._get_state_key(state)
        
        # Espérance de Q(s',a') selon la politique
        expected_next_q = 0 if done else self._get_expected_q(next_state)
        
        # Formule Expected SARSA
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * expected_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.set_q(state, action, new_q)
    
    def train(self, num_episodes, verbose=True):
        """Entraîne l'agent avec Expected SARSA"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # Mettre à jour Q
                self.update(state, action, reward, next_state, done)
                state = next_state
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
        
        self.training_time = time.time() - start_time



