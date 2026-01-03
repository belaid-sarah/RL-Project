"""
Value Iteration - Sutton & Barto Chapter 4.4

Algorithme exact du livre :
V(s) ← max_a Σ_s' Σ_r p(s',r|s,a) [r + γV(s')]
"""

from algos.base_agent import BaseAgent
import random
import time

class ValueIteration(BaseAgent):
    """
    Value Iteration - Dynamic Programming (Sutton & Barto 4.4)
    
    Nécessite un modèle MDP complet : p(s', r | s, a)
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5, **kwargs):
        super().__init__(env, name="Value Iteration", gamma=gamma, theta=theta, **kwargs)
        self.gamma = gamma
        self.theta = theta
        
        # Value function V(s)
        self.V = {}
        # Policy π(s) (dérivée de V)
        self.policy = {}
        
        # Modèle MDP : p(s', r | s, a) -> probabilité
        self.model = {}
        
        # Initialiser
        self._initialize()
    
    def _initialize(self):
        """Initialise les états, actions et V"""
        # Obtenir le modèle MDP
        if hasattr(self.env, 'get_transition_model'):
            self.model = self.env.get_transition_model()
            # Extraire les états et actions du modèle
            self.states = set()
            self.actions = set()
            for (s, a, s_next, r) in self.model.keys():
                self.states.add(s)
                self.states.add(s_next)
                self.actions.add(a)
            self.states = list(self.states)
            self.actions = list(self.actions)
        elif hasattr(self.env, 'secret_env') and hasattr(self.env.secret_env, 'num_states'):
            # Environnement secret avec méthodes MDP
            self._build_model_from_mdp()
        else:
            # Construire le modèle en explorant
            self._build_model()
        
        # Initialiser V
        for s in self.states:
            self.V[s] = 0.0
    
    def _build_model(self):
        """
        Construit le modèle MDP en explorant l'environnement systématiquement.
        
        Même principe que Policy Iteration : on teste toutes les transitions (s, a)
        pour construire le modèle complet p(s', r | s, a).
        
        Voir _build_model() dans policy_iteration.py pour les détails.
        """
        # ÉTAPE 1 : Identifier tous les états et actions
        if hasattr(self.env, 'length'):
            # LineWorld
            self.states = list(range(self.env.length))
            self.actions = [0, 1]
        elif hasattr(self.env, 'width') and hasattr(self.env, 'height'):
            # GridWorld
            self.states = [(x, y) for x in range(self.env.width) 
                          for y in range(self.env.height)]
            self.actions = list(range(self.env.n_actions()))
        else:
            raise ValueError("Environnement non supporte pour Value Iteration")
        
        # ÉTAPE 2 : Tester toutes les transitions
        print(f"Construction du modèle MDP : {len(self.states)} états × {len(self.actions)} actions = {len(self.states) * len(self.actions)} transitions à tester...")
        
        for s in self.states:
            for a in self.actions:
                try:
                    # Sauvegarder l'état actuel
                    old_state = getattr(self.env, 'state', None)
                    old_done = getattr(self.env, 'done', False)
                    
                    # Forcer l'état à s et exécuter l'action a
                    self.env.reset()
                    if hasattr(self.env, 'state'):
                        self.env.state = s
                    if hasattr(self.env, 'done'):
                        self.env.done = False
                    
                    # Tester la transition
                    s_next, r, done, _ = self.env.step(a)
                    r = round(r, 1)
                    
                    # Enregistrer dans le modèle
                    key = (s, a, s_next, r)
                    self.model[key] = 1.0  # Déterministe
                    
                    # Restaurer l'état
                    if old_state is not None:
                        self.env.state = old_state
                    if hasattr(self.env, 'done'):
                        self.env.done = old_done
                except:
                    pass
        
        print(f"[OK] Modele MDP construit : {len(self.model)} transitions enregistrees")
    
    def _build_model_from_mdp(self):
        """
        Construit le modèle MDP à partir des méthodes MDP de l'environnement secret.
        
        Les environnements secrets fournissent :
        - num_states() : nombre d'états
        - num_actions() : nombre d'actions
        - num_rewards() : nombre de rewards possibles
        - p(s, a, s_p, r_index) : probabilité de transition
        - reward(r_index) : valeur du reward
        """
        secret_env = self.env.secret_env
        num_states = secret_env.num_states()
        num_actions = secret_env.num_actions()
        num_rewards = secret_env.num_rewards()
        
        self.states = list(range(num_states))
        self.actions = list(range(num_actions))
        
        print(f"Construction du modèle MDP depuis méthodes MDP : {num_states} états × {num_actions} actions...")
        
        for s in self.states:
            for a in self.actions:
                for s_next in range(num_states):
                    for r_idx in range(num_rewards):
                        prob = secret_env.p(s, a, s_next, r_idx)
                        if prob > 0:
                            reward = secret_env.reward(r_idx)
                            key = (s, a, s_next, reward)
                            self.model[key] = prob
        
        print(f"[OK] Modele MDP construit : {len(self.model)} transitions enregistrees")
    
    def train(self, num_episodes=1000, verbose=True):
        """
        Value Iteration (Sutton & Barto 4.4)
        V(s) ← max_a Σ_s' Σ_r p(s',r|s,a) [r + γV(s')]
        """
        start_time = time.time()
        
        for iteration in range(num_episodes):
            delta = 0
            V_new = {}
            
            for s in self.states:
                v_old = self.V[s]
                
                # V(s) = max_a Σ_s' Σ_r p(s',r|s,a) [r + γV(s')]
                action_values = {}
                for a in self.actions:
                    q_value = 0.0
                    for (s_m, a_m, s_next, r), prob in self.model.items():
                        if s_m == s and a_m == a:
                            q_value += prob * (r + self.gamma * self.V.get(s_next, 0.0))
                    action_values[a] = q_value
                
                v_new = max(action_values.values()) if action_values else 0.0
                V_new[s] = v_new
                
                # Extraire la politique greedy
                if action_values:
                    best_action = max(action_values, key=action_values.get)
                    self.policy[s] = best_action
                
                delta = max(delta, abs(v_old - v_new))
            
            self.V.update(V_new)
            
            # Évaluer performance
            reward = self._test_policy()
            self.episode_rewards.append(reward)
            self.episode_lengths.append(1)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1} | Delta: {delta:.6f} | Reward: {reward:.2f}")
            
            if delta < self.theta:
                self.convergence_episode = iteration + 1
                if verbose:
                    print(f"Value function converged at iteration {iteration+1}")
                break
        
        self.training_time = time.time() - start_time
    
    def _test_policy(self):
        """Teste la politique actuelle"""
        total = 0
        for _ in range(10):
            s = self.env.reset()
            done = False
            reward_sum = 0
            steps = 0
            
            while not done and steps < 100:
                action = self.policy.get(s, random.choice(self.actions))
                s, r, done, _ = self.env.step(action)
                reward_sum += r
                steps += 1
            
            total += reward_sum
        return total / 10
    
    def select_action(self, state, training=False):
        """Sélectionne action selon la politique optimale"""
        return self.policy.get(state, random.choice(self.actions))
