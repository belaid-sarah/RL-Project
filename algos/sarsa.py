from algos.base_agent import BaseAgent
import numpy as np
import random
import time

class SARSAAgent(BaseAgent):
    """
    SARSA (State-Action-Reward-State-Action)
    
    Algorithme on-policy de TD Learning qui apprend Q(s,a) en utilisant
    la valeur de l'action suivante selon la politique d'exploration.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="SARSA", alpha=alpha, gamma=gamma, epsilon=epsilon, **kwargs)
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
    
    def update(self, state, action, reward, next_state, done):
        """Met à jour Q avec la formule SARSA"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Action suivante selon la politique (epsilon-greedy)
        next_action = self.select_action(next_state, training=True)
        next_q = 0 if done else self.get_q(next_state, next_action)
        
        # Formule SARSA: Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.set_q(state, action, new_q)
    
    def train(self, num_episodes, verbose=True, max_steps_per_episode=1000):
        """Entraîne l'agent avec SARSA"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.select_action(state, training=True)
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # Mettre à jour Q
                self.update(state, action, reward, next_state, done)
                
                # Action suivante
                if not done:
                    action = self.select_action(next_state, training=True)
                state = next_state
            
            if steps >= max_steps_per_episode:
                done = True
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
        
        self.training_time = time.time() - start_time
