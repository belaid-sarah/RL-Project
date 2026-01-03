"""
Policy Iteration - Sutton & Barto Chapter 4.3

Algorithme exact du livre :
1. Policy Evaluation: V(s) = Σ_s' Σ_r p(s',r|s,π(s)) [r + γV(s')]
2. Policy Improvement: π'(s) = argmax_a Σ_s' Σ_r p(s',r|s,a) [r + γV(s')]
"""

from algos.base_agent import BaseAgent
import random
import time

class PolicyIteration(BaseAgent):
    """
    Policy Iteration - Dynamic Programming (Sutton & Barto 4.3)
    
    Nécessite un modèle MDP complet : p(s', r | s, a)
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-5, **kwargs):
        super().__init__(env, name="Policy Iteration", gamma=gamma, theta=theta, **kwargs)
        self.gamma = gamma
        self.theta = theta
        
        # Value function V(s)
        self.V = {}
        # Policy π(s)
        self.policy = {}
        
        # Modèle MDP : p(s', r | s, a) -> probabilité
        self.model = {}
        
        # Initialiser
        self._initialize()
    
    def _initialize(self):
        """Initialise les états, actions, V et policy"""
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
        
        # Initialiser V et policy
        for s in self.states:
            self.V[s] = 0.0
            self.policy[s] = random.choice(self.actions)
    
    def _build_model(self):
        """
        Construit le modèle MDP en explorant l'environnement systématiquement.
        
        Le modèle MDP est : p(s', r | s, a) = probabilité d'aller en s' avec reward r
        depuis l'état s avec l'action a.
        
        Pour Policy/Value Iteration, on a besoin de TOUTES les transitions possibles.
        Si l'environnement ne fournit pas le modèle directement, on le construit en :
        1. Identifiant tous les états possibles
        2. Identifiant toutes les actions possibles
        3. Testant chaque transition (s, a) → (s', r)
        4. Enregistrant le résultat dans self.model
        
        Exemple pour LineWorldSimple (length=5) :
        - États : [0, 1, 2, 3, 4]
        - Actions : [0 (gauche), 1 (droite)]
        - On teste : (0,0), (0,1), (1,0), (1,1), ..., (4,0), (4,1)
        - Résultat : model[(s, a, s_next, r)] = 1.0 pour chaque transition
        """
        # ÉTAPE 1 : Identifier tous les états possibles
        if hasattr(self.env, 'length'):
            # LineWorld : états = positions de 0 à length-1
            self.states = list(range(self.env.length))
            self.actions = [0, 1]  # gauche, droite
        elif hasattr(self.env, 'width') and hasattr(self.env, 'height'):
            # GridWorld : états = toutes les positions (x, y)
            self.states = [(x, y) for x in range(self.env.width) 
                          for y in range(self.env.height)]
            self.actions = list(range(self.env.n_actions()))
        else:
            raise ValueError("Environnement non supporte pour Policy Iteration")
        
        # ÉTAPE 2 : Construire le modèle en testant toutes les transitions
        print(f"Construction du modèle MDP : {len(self.states)} états × {len(self.actions)} actions = {len(self.states) * len(self.actions)} transitions à tester...")
        
        for s in self.states:
            for a in self.actions:
                # ÉTAPE 3 : Tester la transition (s, a) → (s', r)
                try:
                    # Sauvegarder l'état actuel de l'environnement
                    old_state = getattr(self.env, 'state', None)
                    old_done = getattr(self.env, 'done', False)
                    
                    # Réinitialiser et forcer l'état à s
                    self.env.reset()
                    if hasattr(self.env, 'state'):
                        self.env.state = s
                    if hasattr(self.env, 'done'):
                        self.env.done = False
                    
                    # ÉTAPE 4 : Exécuter l'action a depuis l'état s
                    s_next, r, done, info = self.env.step(a)
                    
                    # Arrondir le reward pour regrouper les valeurs similaires
                    r = round(r, 1)
                    
                    # ÉTAPE 5 : Enregistrer la transition dans le modèle
                    # Clé : (état_source, action, état_destination, reward)
                    # Valeur : probabilité (1.0 = déterministe)
                    key = (s, a, s_next, r)
                    self.model[key] = 1.0  # Environnement déterministe
                    
                    # Restaurer l'état de l'environnement si possible
                    if old_state is not None:
                        self.env.state = old_state
                    if hasattr(self.env, 'done'):
                        self.env.done = old_done
                except Exception as e:
                    # Si une transition échoue, on l'ignore
                    # (peut arriver pour des états invalides)
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
    
    def _get_p(self, s, a, s_next, r):
        """Retourne p(s', r | s, a)"""
        return self.model.get((s, a, s_next, r), 0.0)
    
    def evaluate_policy(self):
        """
        Policy Evaluation (Sutton & Barto 4.3)
        V(s) = Σ_s' Σ_r p(s',r|s,π(s)) [r + γV(s')]
        """
        while True:
            delta = 0
            V_new = {}
            
            for s in self.states:
                v_old = self.V[s]
                a = self.policy[s]
                
                # V(s) = Σ_s' Σ_r p(s',r|s,π(s)) [r + γV(s')]
                v_new = 0.0
                for (s_m, a_m, s_next, r), prob in self.model.items():
                    if s_m == s and a_m == a:
                        v_new += prob * (r + self.gamma * self.V.get(s_next, 0.0))
                
                V_new[s] = v_new
                delta = max(delta, abs(v_old - v_new))
            
            self.V.update(V_new)
            
            if delta < self.theta:
                break
    
    def improve_policy(self):
        """
        Policy Improvement (Sutton & Barto 4.3)
        π'(s) = argmax_a Σ_s' Σ_r p(s',r|s,a) [r + γV(s')]
        """
        policy_stable = True
        
        for s in self.states:
            old_action = self.policy[s]
            
            # Calculer Q(s,a) pour chaque action
            action_values = {}
            for a in self.actions:
                q_value = 0.0
                for (s_m, a_m, s_next, r), prob in self.model.items():
                    if s_m == s and a_m == a:
                        q_value += prob * (r + self.gamma * self.V.get(s_next, 0.0))
                action_values[a] = q_value
            
            # Choisir l'action greedy
            best_action = max(action_values, key=action_values.get)
            self.policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def train(self, num_episodes=100, verbose=True):
        """
        Policy Iteration (Sutton & Barto 4.3)
        Répète Policy Evaluation puis Policy Improvement jusqu'à convergence
        """
        start_time = time.time()
        
        for iteration in range(num_episodes):
            # Policy Evaluation
            self.evaluate_policy()
            
            # Policy Improvement
            policy_stable = self.improve_policy()
            
            # Évaluer performance
            reward = self._test_policy()
            self.episode_rewards.append(reward)
            self.episode_lengths.append(1)
            
            if verbose and (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration+1} | Stable: {policy_stable} | Reward: {reward:.2f}")
            
            if policy_stable:
                self.convergence_episode = iteration + 1
                if verbose:
                    print(f"Policy converged at iteration {iteration+1}")
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
        """Sélectionne action selon la politique"""
        return self.policy.get(state, random.choice(self.actions))
    
    def save(self, path):
        """
        Sauvegarde l'agent avec V(s) et policy complètes
        """
        from pathlib import Path
        import pickle
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'name': self.name,
            'hyperparameters': self.hyperparameters,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_time': self.training_time,
            'convergence_episode': self.convergence_episode,
            # Sauvegarder V(s) et policy
            'V': self.V,
            'policy': self.policy,
            'states': self.states,
            'actions': self.actions,
            'gamma': self.gamma,
            'theta': self.theta
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[OK] Agent sauvegarde avec V(s) et policy : {path}")
    
    def load(self, path):
        """
        Charge un agent avec V(s) et policy complètes
        """
        import pickle
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.name = save_data['name']
        self.hyperparameters = save_data['hyperparameters']
        self.episode_rewards = save_data['episode_rewards']
        self.episode_lengths = save_data['episode_lengths']
        self.training_time = save_data['training_time']
        self.convergence_episode = save_data.get('convergence_episode')
        
        # Charger V(s) et policy
        if 'V' in save_data:
            self.V = save_data['V']
        if 'policy' in save_data:
            self.policy = save_data['policy']
        if 'states' in save_data:
            self.states = save_data['states']
        if 'actions' in save_data:
            self.actions = save_data['actions']
        if 'gamma' in save_data:
            self.gamma = save_data['gamma']
        if 'theta' in save_data:
            self.theta = save_data['theta']
        
        print(f"[OK] Agent charge avec V(s) et policy : {path}")