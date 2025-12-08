"""
Q-Learning - Sutton & Barto Chapter 6.5

Formule exacte du livre :
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
"""

from algos.base_agent import BaseAgent
import random
import time

class QLearningAgent(BaseAgent):
    """
    Q-Learning - Off-policy TD Control (Sutton & Barto 6.5)
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Q-Learning", alpha=alpha, gamma=gamma, epsilon=epsilon, **kwargs)
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur d'actualisation
        self.epsilon = epsilon  # Exploration
        
        # Q-table: Q(s, a)
        self.Q = {}
    
    def _get_state_key(self, state):
        """Convertit état en clé hashable"""
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, tuple)) and len(state) > 0:
            return tuple(state) if not isinstance(state[0], tuple) else state
        else:
            return state
    
    def get_q(self, state, action):
        """Retourne Q(state, action)"""
        state_key = self._get_state_key(state)
        return self.Q.get((state_key, action), 0.0)
    
    def set_q(self, state, action, value):
        """Met à jour Q(state, action)"""
        state_key = self._get_state_key(state)
        self.Q[(state_key, action)] = value
    
    def select_action(self, state, training=False):
        """
        Epsilon-greedy (Sutton & Barto 2.7)
        """
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploitation: argmax_a Q(s, a)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update (Sutton & Barto 6.5)
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # TD Target: R_{t+1} + γ max_a Q(S_{t+1}, a)
        if done:
            max_next_q = 0.0
        else:
            actions = self.env.action_space
            max_next_q = max([self.get_q(next_state, a) for a in actions])
        
        # Q-Learning update
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.set_q(state, action, new_q)
    
    def train(self, num_episodes, verbose=True, max_steps_per_episode=1000):
        """
        Entraîne avec Q-Learning
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # Choisir action
                action = self.select_action(state, training=True)
                
                # Exécuter action
                next_state, reward, done, _ = self.env.step(action)
                
                # Mettre à jour Q
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
        
        self.training_time = time.time() - start_time
