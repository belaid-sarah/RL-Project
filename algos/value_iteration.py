from algos.base_agent import BaseAgent
import numpy as np
import random
import time

class ValueIteration(BaseAgent):
    """
    Value Iteration - Dynamic Programming
    
    Algorithme qui calcule directement la fonction de valeur optimale
    en itérant sur l'équation de Bellman optimale.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5, **kwargs):
        super().__init__(env, name="Value Iteration", gamma=gamma, theta=theta, **kwargs)
        self.gamma = gamma
        self.theta = theta
        
        # Initialiser les structures de données
        self.V = {}  # Value function
        self.policy = {}  # Policy: state -> action
        
        # Obtenir tous les états possibles
        self._initialize_states()
    
    def _initialize_states(self):
        """Initialise les états"""
        if hasattr(self.env, 'length'):
            self.states = list(range(self.env.length))
            self.actions = list(range(self.env.n_actions()))
        elif hasattr(self.env, 'width') and hasattr(self.env, 'height'):
            self.states = [(x, y) for x in range(self.env.width) for y in range(self.env.height)]
            self.actions = list(range(self.env.n_actions()))
        else:
            self.states = []
            self.actions = list(range(self.env.n_actions()))
        
        # Initialiser V
        for state in self.states:
            state_key = self._get_state_key(state)
            self.V[state_key] = 0.0
    
    def _get_state_key(self, state):
        """Convertit un état en clé hashable"""
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        else:
            return state
    
    def train(self, num_episodes=1000, verbose=True):
        """Entraîne l'agent avec Value Iteration"""
        start_time = time.time()
        
        for iteration in range(num_episodes):
            delta = 0
            V_new = {}
            
            for state in self.states:
                state_key = self._get_state_key(state)
                v_old = self.V.get(state_key, 0.0)
                
                # Calculer les valeurs d'action
                action_values = {}
                for action in self.actions:
                    self.env.reset()
                    if hasattr(self.env, 'state'):
                        self.env.state = state
                    
                    next_state, reward, done, _ = self.env.step(action)
                    next_state_key = self._get_state_key(next_state)
                    
                    action_values[action] = reward + self.gamma * (0 if done else self.V.get(next_state_key, 0.0))
                
                # Mettre à jour V avec la valeur maximale
                v_new = max(action_values.values())
                V_new[state_key] = v_new
                
                # Extraire la politique greedy
                best_action = max(action_values, key=action_values.get)
                self.policy[state_key] = best_action
                
                delta = max(delta, abs(v_old - v_new))
            
            # Mettre à jour V
            self.V.update(V_new)
            
            # Évaluer la performance
            total_reward = self._evaluate_policy_performance()
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(1)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1}/{num_episodes} | Delta: {delta:.6f} | Avg Reward: {total_reward:.2f}")
            
            if delta < self.theta:
                self.convergence_episode = iteration + 1
                if verbose:
                    print(f"✅ Value function converged at iteration {iteration+1}")
                break
        
        self.training_time = time.time() - start_time
    
    def _evaluate_policy_performance(self):
        """Évalue la performance de la politique actuelle"""
        total_reward = 0
        num_tests = 10
        
        for _ in range(num_tests):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                state_key = self._get_state_key(state)
                action = self.policy.get(state_key, random.choice(self.actions))
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_tests
    
    def select_action(self, state, training=False):
        """Sélectionne une action selon la politique optimale"""
        state_key = self._get_state_key(state)
        return self.policy.get(state_key, random.choice(self.actions))
