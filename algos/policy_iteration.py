from algos.base_agent import BaseAgent
import numpy as np
import random
import time

class PolicyIteration(BaseAgent):
    """
    Policy Iteration - Dynamic Programming
    
    Algorithme qui alterne entre :
    1. Policy Evaluation : évalue la valeur de la politique actuelle
    2. Policy Improvement : améliore la politique en étant greedy par rapport à V
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5, **kwargs):
        super().__init__(env, name="Policy Iteration", gamma=gamma, theta=theta, **kwargs)
        self.gamma = gamma
        self.theta = theta
        
        # Initialiser les structures de données
        self.V = {}  # Value function
        self.policy = {}  # Policy: state -> action
        
        # Obtenir tous les états possibles (nécessite un modèle de l'environnement)
        self._initialize_states()
    
    def _initialize_states(self):
        """Initialise les états et la politique"""
        # Pour les environnements simples (LineWorld, GridWorld)
        if hasattr(self.env, 'length'):
            # LineWorld
            self.states = list(range(self.env.length))
            self.actions = list(range(self.env.n_actions()))
        elif hasattr(self.env, 'width') and hasattr(self.env, 'height'):
            # GridWorld
            self.states = [(x, y) for x in range(self.env.width) for y in range(self.env.height)]
            self.actions = list(range(self.env.n_actions()))
        else:
            # Pour les autres environnements, on découvre les états dynamiquement
            self.states = []
            self.actions = list(range(self.env.n_actions()))
            # On découvrira les états pendant l'entraînement
        
        # Initialiser V et policy
        for state in self.states:
            state_key = self._get_state_key(state)
            self.V[state_key] = 0.0
            self.policy[state_key] = random.choice(self.actions)
    
    def _get_state_key(self, state):
        """Convertit un état en clé hashable"""
        if isinstance(state, dict):
            return tuple(sorted(state.items()))
        elif isinstance(state, (list, np.ndarray)):
            return tuple(state)
        else:
            return state
    
    def evaluate_policy(self):
        """Évalue la politique actuelle jusqu'à convergence"""
        max_iterations = 1000  # Limite de sécurité
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            delta = 0
            V_new = {}
            
            for state in self.states:
                state_key = self._get_state_key(state)
                v_old = self.V.get(state_key, 0.0)
                
                # Obtenir l'action selon la politique
                if state_key not in self.policy:
                    self.policy[state_key] = random.choice(self.actions)
                action = self.policy[state_key]
                
                # Calculer la valeur attendue
                v_new = 0.0
                
                try:
                    # Simuler la transition
                    self.env.reset()
                    if hasattr(self.env, 'state'):
                        self.env.state = state
                    elif isinstance(state, tuple) and len(state) == 2:
                        # Pour GridWorld: state est (x, y)
                        self.env.state = state
                    
                    next_state, reward, done, _ = self.env.step(action)
                    next_state_key = self._get_state_key(next_state)
                    v_new = reward + self.gamma * (0 if done else self.V.get(next_state_key, 0.0))
                except Exception as e:
                    # Si erreur, utiliser valeur par défaut
                    v_new = v_old
                
                V_new[state_key] = v_new
                delta = max(delta, abs(v_old - v_new))
            
            # Mettre à jour V
            self.V.update(V_new)
            
            if delta < self.theta:
                break
    
    def improve_policy(self):
        """Améliore la politique en étant greedy par rapport à V"""
        policy_stable = True
        
        for state in self.states:
            state_key = self._get_state_key(state)
            old_action = self.policy.get(state_key, random.choice(self.actions))
            
            # Calculer les valeurs d'action
            action_values = {}
            for action in self.actions:
                try:
                    self.env.reset()
                    if hasattr(self.env, 'state'):
                        self.env.state = state
                    elif isinstance(state, tuple) and len(state) == 2:
                        # Pour GridWorld: state est (x, y)
                        self.env.state = state
                    
                    next_state, reward, done, _ = self.env.step(action)
                    next_state_key = self._get_state_key(next_state)
                    
                    action_values[action] = reward + self.gamma * (0 if done else self.V.get(next_state_key, 0.0))
                except Exception:
                    # Si erreur, utiliser valeur par défaut
                    action_values[action] = -1000
            
            # Choisir l'action greedy
            if action_values:
                best_action = max(action_values, key=action_values.get)
                self.policy[state_key] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def train(self, num_episodes=100, verbose=True):
        """Entraîne l'agent avec Policy Iteration"""
        start_time = time.time()
        
        for iteration in range(num_episodes):
            # Policy Evaluation
            self.evaluate_policy()
            
            # Policy Improvement
            policy_stable = self.improve_policy()
            
            # Calculer la performance de la politique
            total_reward = self._evaluate_policy_performance()
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(1)  # Une itération
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{num_episodes} | Policy stable: {policy_stable} | Avg Reward: {total_reward:.2f}")
            
            if policy_stable:
                self.convergence_episode = iteration + 1
                if verbose:
                    print(f"✅ Policy converged at iteration {iteration+1}")
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
        """Sélectionne une action selon la politique"""
        state_key = self._get_state_key(state)
        return self.policy.get(state_key, random.choice(self.actions))
