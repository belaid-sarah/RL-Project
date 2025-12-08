"""
Expected SARSA - Sutton & Barto Chapter 6.6

Formule exacte du livre :
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Σ_a π(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]
"""

from algos.base_agent import BaseAgent
import random
import time

class ExpectedSARSAAgent(BaseAgent):
    """
    Expected SARSA (Sutton & Barto 6.6)
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Expected SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: Q(s, a)
        self.Q = {}
    
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
    
    def set_q(self, state, action, value):
        state_key = self._get_state_key(state)
        self.Q[(state_key, action)] = value
    
    def select_action(self, state, training=False):
        """Epsilon-greedy"""
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def _get_expected_q(self, state):
        """
        Calcule Σ_a π(a|S_{t+1}) Q(S_{t+1}, a)
        où π est epsilon-greedy
        """
        actions = self.env.action_space
        num_actions = len(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        
        # Probabilité selon epsilon-greedy
        expected_value = 0.0
        for action in actions:
            if action in best_actions:
                prob = (1 - self.epsilon) / len(best_actions) + self.epsilon / num_actions
            else:
                prob = self.epsilon / num_actions
            expected_value += prob * self.get_q(state, action)
        
        return expected_value
    
    def update(self, state, action, reward, next_state, done):
        """
        Expected SARSA update (Sutton & Barto 6.6)
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Σ_a π(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]
        """
        state_key = self._get_state_key(state)
        
        # TD Target: R_{t+1} + γ Σ_a π(a|S_{t+1}) Q(S_{t+1}, a)
        if done:
            expected_next_q = 0.0
        else:
            expected_next_q = self._get_expected_q(next_state)
        
        # Expected SARSA update
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * expected_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.set_q(state, action, new_q)
    
    def train(self, num_episodes, verbose=True, max_steps_per_episode=1000):
        """Entraîne avec Expected SARSA"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
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
