import numpy as np
import random
from algos.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, **kwargs):
        super().__init__(env, name="Q-Learning", **kwargs)
        self.alpha = alpha      # Taux d'apprentissage
        self.gamma = gamma      # Facteur d'actualisation
        self.epsilon = epsilon  # Exploration
        
        # Initialisation de la Q-Table : dict[(state, action)] = value
        self.q_table = {}

    def get_q(self, state, action):
        state_key = self._get_state_key(state)
        return self.q_table.get((state_key, action), 0.0)

    def select_action(self, state, training=False):
        state_key = self._get_state_key(state)
        actions = self.env.action_space
        
        # Epsilon-greedy pour l'entraînement
        if training and random.random() < self.epsilon:
            return random.choice(actions)
        
        # Greedy : meilleure action connue
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        # Gestion des égalités (tie-breaking) aléatoire
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Meilleure valeur future possible (max a' Q(s', a'))
        next_max_q = 0 if done else max([self.get_q(next_state, a) for a in self.env.action_space])
        
        # Formule Q-Learning : Q(s,a) = Q(s,a) + alpha * (TD_target - Q(s,a))
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * next_max_q
        self.q_table[(state_key, action)] = current_q + self.alpha * (td_target - current_q)

    def train(self, num_episodes, verbose=True, max_steps_per_episode=1000):
        import time
        start_time = time.time()
        
        for ep in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
            
            # Si l'épisode n'a pas terminé, forcer done
            if steps >= max_steps_per_episode:
                done = True
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if verbose and (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
        
        self.training_time = time.time() - start_time