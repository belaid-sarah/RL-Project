"""
Policy Iteration avec modèle MDP complet

Implémentation qui construit un modèle MDP complet (p[s, a, s', r])
comme dans l'exemple du notebook.
"""

from algos.base_agent import BaseAgent
import numpy as np
import random
import time

class PolicyIterationMDP(BaseAgent):
    """
    Policy Iteration avec modèle MDP complet
    
    Construit un modèle de transition probabiliste p[s, a, s', r]
    et utilise la formule de Bellman complète.
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6, **kwargs):
        super().__init__(env, name="Policy Iteration (MDP)", gamma=gamma, theta=theta, **kwargs)
        self.gamma = gamma
        self.theta = theta
        
        # Modèle MDP : p[s, a, s', r] -> probability
        self.model = {}
        
        # Structures de données
        self.V = {}  # Value function: state -> value
        self.policy = {}  # Policy: state -> action
        
        # Obtenir tous les états et actions
        self._initialize_states()
        
        # Index pour recherche rapide
        self.transition_index = {}
        
        # Construire le modèle MDP (avec échantillonnage multiple)
        self._build_mdp_model(num_samples=10)
    
    def _initialize_states(self):
        """Initialise les états et actions"""
        if hasattr(self.env, 'get_all_states'):
            self.states = self.env.get_all_states()
            self.actions = self.env.get_all_actions()
        elif hasattr(self.env, 'length'):
            # LineWorld simple
            self.states = list(range(self.env.length))
            self.actions = [0, 1]  # gauche, droite
        elif hasattr(self.env, 'width') and hasattr(self.env, 'height'):
            # GridWorld
            self.states = [(x, y) for x in range(self.env.width) for y in range(self.env.height)]
            self.actions = list(range(self.env.n_actions()))
        elif hasattr(self.env, 'n_actions'):
            # Pour TwoRoundRPS et autres, découvrir les états dynamiquement
            # On va construire le modèle en explorant
            self.states = []
            self.actions = list(range(self.env.n_actions()))
        else:
            raise ValueError("Environnement non supporté pour Policy Iteration MDP")
        
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
    
    def _build_mdp_model(self, num_samples=10):
        """
        Construit le modèle MDP complet p[s, a, s', r] avec améliorations
        
        Améliorations :
        - Échantillonnage multiple pour gérer les transitions probabilistes
        - Normalisation correcte des probabilités
        - Indexation pour recherche rapide
        - Gestion des états terminaux
        """
        print("🔨 Construction du modèle MDP (amélioré)...")
        
        if hasattr(self.env, 'get_transition_model'):
            # L'environnement fournit directement le modèle
            self.model = self.env.get_transition_model()
            # Découvrir les états depuis le modèle
            discovered_states = set()
            for (s, a, s_next, r), prob in self.model.items():
                discovered_states.add(s)
                discovered_states.add(s_next)
            if not self.states:
                self.states = list(discovered_states)
        else:
            # Construire le modèle en explorant avec échantillonnage multiple
            self.model = {}
            transition_counts = {}  # Compteur pour normalisation
            discovered_states = set()
            
            # Si on n'a pas d'états, les découvrir en explorant
            if not self.states:
                # Explorer pour découvrir les états
                for _ in range(100):  # Explorer 100 épisodes
                    state = self.env.reset()
                    discovered_states.add(self._get_state_key(state))
                    done = False
                    steps = 0
                    while not done and steps < 50:
                        action = random.choice(self.actions)
                        state, reward, done, _ = self.env.step(action)
                        discovered_states.add(self._get_state_key(state))
                        steps += 1
                
                self.states = list(discovered_states)
                # Initialiser V et policy pour les nouveaux états
                for state_key in self.states:
                    if state_key not in self.V:
                        self.V[state_key] = 0.0
                    if state_key not in self.policy:
                        self.policy[state_key] = random.choice(self.actions)
            
            # Pour chaque état et action, échantillonner plusieurs fois
            # pour gérer les transitions probabilistes
            for s in self.states:
                for a in self.actions:
                    # Sauvegarder l'état de l'environnement
                    old_state = getattr(self.env, 'state', None)
                    old_done = getattr(self.env, 'done', False)
                    old_obstacles = getattr(self.env, 'obstacles', None)
                    old_traps = getattr(self.env, 'traps', None)
                    old_rewards = getattr(self.env, 'rewards', None)
                    
                    # Échantillonner plusieurs fois pour estimer les probabilités
                    transitions = {}  # {(s', r): count}
                    
                    for sample in range(num_samples):
                        try:
                            # Réinitialiser l'environnement à l'état s
                            self.env.reset()
                            if hasattr(self.env, 'state'):
                                self.env.state = s
                            elif isinstance(s, tuple) and len(s) == 2:
                                # Pour GridWorld
                                self.env.state = s
                            
                            # Restaurer les obstacles, pièges, récompenses si nécessaire
                            if old_obstacles is not None:
                                self.env.obstacles = old_obstacles.copy()
                            if old_traps is not None:
                                self.env.traps = old_traps.copy()
                            if old_rewards is not None:
                                self.env.rewards = old_rewards.copy()
                            
                            next_state, reward, done, _ = self.env.step(a)
                            
                            # Catégoriser la récompense (plus précis)
                            reward_category = round(reward * 20) / 20  # Arrondir à 0.05 près
                            
                            # Stocker la transition
                            s_key = self._get_state_key(s)
                            s_next_key = self._get_state_key(next_state)
                            transition_key = (s_next_key, reward_category)
                            
                            if transition_key not in transitions:
                                transitions[transition_key] = 0
                            transitions[transition_key] += 1
                            
                        except Exception as e:
                            # Si erreur, ignorer cet échantillon
                            pass
                    
                    # Normaliser les probabilités
                    s_key = self._get_state_key(s)
                    total_count = sum(transitions.values())
                    
                    if total_count > 0:
                        for (s_next_key, reward_category), count in transitions.items():
                            prob = count / total_count
                            key = (s_key, a, s_next_key, reward_category)
                            self.model[key] = prob
                            
                            # Compter pour vérification
                            transition_counts[(s_key, a)] = transition_counts.get((s_key, a), 0) + count
                    
                    # Restaurer l'état
                    if old_state is not None:
                        self.env.state = old_state
                    self.env.done = old_done
                    if old_obstacles is not None:
                        self.env.obstacles = old_obstacles
                    if old_traps is not None:
                        self.env.traps = old_traps
                    if old_rewards is not None:
                        self.env.rewards = old_rewards
            
            # Vérifier que les probabilités sont normalisées (somme = 1.0 pour chaque (s, a))
            self._normalize_probabilities()
        
        # Créer un index pour recherche rapide
        self._build_transition_index()
        
        print(f"✅ Modèle MDP construit: {len(self.model)} transitions, {len(self.states)} états")
    
    def _normalize_probabilities(self):
        """Normalise les probabilités pour que Σ_s',r p[s, a, s', r] = 1.0"""
        # Grouper par (s, a)
        grouped = {}
        for (s, a, s_next, r), prob in self.model.items():
            key = (s, a)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(((s, a, s_next, r), prob))
        
        # Normaliser chaque groupe
        for (s, a), transitions in grouped.items():
            total_prob = sum(prob for _, prob in transitions)
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
                # Normaliser
                for (key, prob) in transitions:
                    self.model[key] = prob / total_prob
    
    def _build_transition_index(self):
        """Construit un index pour recherche rapide des transitions"""
        self.transition_index = {}  # {(s, a): [(s', r, prob), ...]}
        
        for (s, a, s_next, r), prob in self.model.items():
            key = (s, a)
            if key not in self.transition_index:
                self.transition_index[key] = []
            self.transition_index[key].append((s_next, r, prob))
    
    def evaluate_policy(self):
        """
        Évalue la politique actuelle avec le modèle MDP complet (amélioré)
        
        Utilise l'index pour recherche rapide et gère correctement les états terminaux
        """
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            delta = 0
            V_new = {}
            
            for s in self.states:
                s_key = self._get_state_key(s)
                v_old = self.V.get(s_key, 0.0)
                
                # Obtenir l'action selon la politique
                if s_key not in self.policy:
                    self.policy[s_key] = random.choice(self.actions)
                action = self.policy[s_key]
                
                # Calculer V[s] avec la formule de Bellman complète (optimisée)
                # V[s] = Σ_s' Σ_r p[s, a, s', r] * (r + γ * V[s'])
                total = 0.0
                
                # Utiliser l'index pour recherche rapide
                key = (s_key, action)
                if key in self.transition_index:
                    for s_next_key, r, prob in self.transition_index[key]:
                        # Gérer les états terminaux (V[s'] = 0 si terminal)
                        next_value = 0.0 if self._is_terminal(s_next_key) else self.V.get(s_next_key, 0.0)
                        total += prob * (r + self.gamma * next_value)
                else:
                    # Si aucune transition trouvée, chercher dans le modèle complet
                    for (s_m, a_m, s_next, r), prob in self.model.items():
                        if s_m == s_key and a_m == action:
                            s_next_key = s_next
                            next_value = 0.0 if self._is_terminal(s_next_key) else self.V.get(s_next_key, 0.0)
                            total += prob * (r + self.gamma * next_value)
                
                # Si aucune transition trouvée, utiliser valeur par défaut
                if total == 0.0 and (s_key, action) not in self.transition_index:
                    total = v_old
                
                V_new[s_key] = total
                delta = max(delta, abs(v_old - V_new[s_key]))
            
            # Mettre à jour V
            self.V.update(V_new)
            
            if delta < self.theta:
                break
        
        return iteration
    
    def _is_terminal(self, state_key):
        """Vérifie si un état est terminal"""
        # Pour LineWorldSimple, le goal est length-1
        if isinstance(state_key, int):
            if hasattr(self.env, 'goal'):
                return state_key == self.env.goal
        # Pour GridWorld, vérifier si c'est le goal
        elif isinstance(state_key, tuple) and len(state_key) == 2:
            if hasattr(self.env, 'goal'):
                return state_key == self.env.goal
        return False
    
    def improve_policy(self):
        """
        Améliore la politique en étant greedy par rapport à V (amélioré)
        
        Utilise l'index pour recherche rapide et gère les égalités
        """
        policy_stable = True
        
        for s in self.states:
            s_key = self._get_state_key(s)
            old_action = self.policy.get(s_key, random.choice(self.actions))
            
            # Calculer les valeurs d'action pour toutes les actions
            action_values = {}
            
            for a in self.actions:
                total = 0.0
                
                # Calculer Q(s, a) = Σ_s' Σ_r p[s, a, s', r] * (r + γ * V[s'])
                # Utiliser l'index pour recherche rapide
                key = (s_key, a)
                if key in self.transition_index:
                    for s_next_key, r, prob in self.transition_index[key]:
                        next_value = 0.0 if self._is_terminal(s_next_key) else self.V.get(s_next_key, 0.0)
                        total += prob * (r + self.gamma * next_value)
                else:
                    # Fallback : chercher dans le modèle complet
                    for (s_m, a_m, s_next, r), prob in self.model.items():
                        if s_m == s_key and a_m == a:
                            s_next_key = s_next
                            next_value = 0.0 if self._is_terminal(s_next_key) else self.V.get(s_next_key, 0.0)
                            total += prob * (r + self.gamma * next_value)
                
                action_values[a] = total
            
            # Choisir l'action greedy (gestion des égalités)
            if action_values:
                max_value = max(action_values.values())
                best_actions = [a for a, v in action_values.items() if v == max_value]
                best_action = random.choice(best_actions)  # Tie-breaking aléatoire
                self.policy[s_key] = best_action
                
                if old_action != best_action:
                    policy_stable = False
            else:
                # Si aucune transition trouvée, garder l'action actuelle
                if s_key not in self.policy:
                    self.policy[s_key] = random.choice(self.actions)
        
        return policy_stable
    
    def train(self, num_episodes=100, verbose=True, max_steps_per_episode=None):
        """Entraîne l'agent avec Policy Iteration"""
        start_time = time.time()
        
        for iteration in range(num_episodes):
            # Policy Evaluation
            eval_iterations = self.evaluate_policy()
            
            # Policy Improvement
            policy_stable = self.improve_policy()
            
            # Calculer la performance
            total_reward = self._evaluate_policy_performance()
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(1)
            
            if verbose and (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration+1}/{num_episodes} | "
                      f"Eval iterations: {eval_iterations} | "
                      f"Policy stable: {policy_stable} | "
                      f"Avg Reward: {total_reward:.2f}")
            
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
            steps = 0
            
            while not done and steps < 100:
                state_key = self._get_state_key(state)
                action = self.policy.get(state_key, random.choice(self.actions))
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
        
        return total_reward / num_tests
    
    def select_action(self, state, training=False):
        """Sélectionne une action selon la politique"""
        state_key = self._get_state_key(state)
        return self.policy.get(state_key, random.choice(self.actions))

