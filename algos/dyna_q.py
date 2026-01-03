"""
Dyna-Q - Sutton & Barto Chapter 8.2

Algorithme exact du livre :
1. Q-Learning avec interaction réelle
2. Modèle de l'environnement
3. n étapes de planning simulées
"""

from algos.base_agent import BaseAgent
import random
import time

class DynaQAgent(BaseAgent):
    """
    Dyna-Q (Sutton & Barto 8.2)
    Combine Q-Learning avec planning
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05, n_planning_steps=5, **kwargs):
        super().__init__(env, name="Dyna-Q", alpha=alpha, gamma=gamma, epsilon=epsilon, 
                        n_planning_steps=n_planning_steps, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon  # Exploration initiale
        self.epsilon_decay = epsilon_decay  # Décroissance de l'exploration
        self.epsilon_min = epsilon_min  # Exploration minimale
        self.n_planning_steps = n_planning_steps
        
        # Q-table: Q(s, a)
        self.Q = {}
        
        # Modèle: (s, a) -> (s', r)
        self.model = {}
        
        # Liste des paires (s, a) visitées (pour planning)
        self.visited_sa = []
    
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
    
    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning update (Sutton & Barto 6.5)
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Q-Learning update
        if done:
            max_next_q = 0.0
        else:
            actions = self.env.action_space
            max_next_q = max([self.get_q(next_state, a) for a in actions])
        
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.set_q(state, action, new_q)
        
        # Mettre à jour le modèle
        self.model[(state_key, action)] = (next_state, reward, done)
        
        # Ajouter à visited_sa
        if (state_key, action) not in self.visited_sa:
            self.visited_sa.append((state_key, action))
    
    def _planning_step(self):
        """
        Planning step (Sutton & Barto 8.2)
        Choisit (s, a) aléatoire et simule Q-Learning update
        """
        if not self.visited_sa:
            return
        
        # Choisir (s, a) aléatoire
        state_key, action = random.choice(self.visited_sa)
        
        # Récupérer transition du modèle
        if (state_key, action) in self.model:
            next_state, reward, done = self.model[(state_key, action)]
            
            # Q-Learning update simulé
            next_state_key = self._get_state_key(next_state)
            if done:
                max_next_q = 0.0
            else:
                actions = self.env.action_space
                max_next_q = max([self.get_q(next_state, a) for a in actions])
            
            current_q = self.get_q(state_key, action)
            td_target = reward + self.gamma * max_next_q
            new_q = current_q + self.alpha * (td_target - current_q)
            self.set_q(state_key, action, new_q)
    
    def train(self, num_episodes, verbose=True, max_steps_per_episode=1000):
        """
        Dyna-Q (Sutton & Barto 8.2)
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < max_steps_per_episode:
                # Action réelle
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # Q-Learning update avec interaction réelle
                self.update(state, action, reward, next_state, done)
                
                # Planning: n étapes simulées
                for _ in range(self.n_planning_steps):
                    self._planning_step()
                
                state = next_state
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Décroissance de l'exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps} | Epsilon: {self.epsilon:.3f}")
        
        self.training_time = time.time() - start_time
