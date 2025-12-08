from algos.base_agent import BaseAgent
from algos.dyna_q import DynaQAgent
import numpy as np
import random
import time

class DynaQPlusAgent(BaseAgent):
    """
    Dyna-Q+
    
    Extension de Dyna-Q qui gère les changements dans l'environnement.
    Ajoute une récompense bonus pour les transitions non visitées récemment,
    encourageant l'exploration continue.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5, 
                 kappa=1e-3, tau=1000, **kwargs):
        super().__init__(env, name="Dyna-Q+", alpha=alpha, gamma=gamma, epsilon=epsilon,
                        n_planning_steps=n_planning_steps, kappa=kappa, tau=tau, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        self.kappa = kappa  # Poids du bonus d'exploration
        self.tau = tau  # Temps de vie pour considérer une transition comme "ancienne"
        
        # Q-table: (state, action) -> value
        self.q_table = {}
        
        # Modèle de l'environnement: (state, action) -> (next_state, reward, time)
        self.model = {}
        
        # Temps de dernière visite: (state, action) -> time
        self.last_visit_time = {}
        
        # Liste des paires (state, action) visitées
        self.visited_sa = []
        
        # Compteur de temps global
        self.time_step = 0
    
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
    
    def _get_exploration_bonus(self, state_key, action):
        """Calcule le bonus d'exploration pour encourager la visite de transitions anciennes"""
        if (state_key, action) not in self.last_visit_time:
            return self.kappa * np.sqrt(self.time_step)
        
        time_since_visit = self.time_step - self.last_visit_time[(state_key, action)]
        if time_since_visit > self.tau:
            return self.kappa * np.sqrt(time_since_visit)
        return 0.0
    
    def update(self, state, action, reward, next_state, done):
        """Met à jour Q avec Q-Learning"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Q-Learning update
        next_max_q = 0 if done else max([self.get_q(next_state, a) for a in self.env.action_space])
        current_q = self.get_q(state, action)
        td_target = reward + self.gamma * next_max_q
        new_q = current_q + self.alpha * (td_target - current_q)
        self.set_q(state, action, new_q)
        
        # Mettre à jour le modèle
        self.model[(state_key, action)] = (next_state, reward, done, self.time_step)
        self.last_visit_time[(state_key, action)] = self.time_step
        
        # Ajouter à la liste des paires visitées
        if (state_key, action) not in self.visited_sa:
            self.visited_sa.append((state_key, action))
    
    def _planning_step(self):
        """Effectue une étape de planning avec bonus d'exploration"""
        if not self.visited_sa:
            return
        
        # Choisir une paire (state, action) aléatoire
        state_key, action = random.choice(self.visited_sa)
        
        # Récupérer la transition du modèle
        if (state_key, action) in self.model:
            next_state, reward, done, model_time = self.model[(state_key, action)]
            
            # Calculer le bonus d'exploration
            exploration_bonus = self._get_exploration_bonus(state_key, action)
            reward_with_bonus = reward + exploration_bonus
            
            # Mettre à jour Q avec cette transition simulée
            next_state_key = self._get_state_key(next_state)
            next_max_q = 0 if done else max([self.get_q(next_state, a) for a in self.env.action_space])
            current_q = self.get_q(state_key, action)
            td_target = reward_with_bonus + self.gamma * next_max_q
            new_q = current_q + self.alpha * (td_target - current_q)
            self.set_q(state_key, action, new_q)
    
    def train(self, num_episodes, verbose=True):
        """Entraîne l'agent avec Dyna-Q+"""
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                self.time_step += 1
                
                # Action réelle
                action = self.select_action(state, training=True)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # Mettre à jour Q et le modèle avec l'interaction réelle
                self.update(state, action, reward, next_state, done)
                
                # Planning: n étapes simulées
                for _ in range(self.n_planning_steps):
                    self._planning_step()
                
                state = next_state
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
        
        self.training_time = time.time() - start_time



