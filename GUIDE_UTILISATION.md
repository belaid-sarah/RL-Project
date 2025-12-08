# Guide d'Utilisation - Projet Reinforcement Learning

Ce guide explique comment utiliser tous les algorithmes d'apprentissage par renforcement sur tous les environnements.

## 📋 Table des matières

1. [Structure du projet](#structure-du-projet)
2. [Algorithmes implémentés](#algorithmes-implémentés)
3. [Environnements disponibles](#environnements-disponibles)
4. [Utilisation de base](#utilisation-de-base)
5. [Tests complets](#tests-complets)
6. [Exemples d'utilisation](#exemples-dutilisation)
7. [Interprétation des résultats](#interprétation-des-résultats)

## 📁 Structure du projet

```
RL-PROJECT/
├── algos/                    # Algorithmes d'apprentissage
│   ├── base_agent.py        # Classe de base pour tous les agents
│   ├── policy_iteration.py  # Policy Iteration (DP)
│   ├── value_iteration.py   # Value Iteration (DP)
│   ├── monte_carlo.py       # Monte Carlo (ES, On-policy, Off-policy)
│   ├── sarsa.py             # SARSA (TD Learning)
│   ├── q_learning.py        # Q-Learning (TD Learning)
│   ├── expected_sarsa.py    # Expected SARSA (TD Learning)
│   ├── dyna_q.py            # Dyna-Q (Planning)
│   └── dyna_q_plus.py       # Dyna-Q+ (Planning)
├── envs/                     # Environnements
│   ├── lineworld.py         # Line World
│   ├── gridworld.py         # Grid World
│   ├── rps.py               # Two-Round Rock Paper Scissors
│   ├── monty_hall_level1.py # Monty Hall (3 portes)
│   └── monty_hall_level2.py # Monty Hall (5 portes)
└── test_all_algos_envs.py   # Script de test complet
```

## 🤖 Algorithmes implémentés

### Dynamic Programming

#### 1. Policy Iteration
- **Fichier**: `algos/policy_iteration.py`
- **Classe**: `PolicyIteration`
- **Description**: Alterne entre évaluation de politique et amélioration jusqu'à convergence
- **Hyperparamètres**:
  - `gamma` (0.99): Facteur d'actualisation
  - `theta` (1e-5): Seuil de convergence

#### 2. Value Iteration
- **Fichier**: `algos/value_iteration.py`
- **Classe**: `ValueIteration`
- **Description**: Calcule directement la fonction de valeur optimale
- **Hyperparamètres**:
  - `gamma` (0.99): Facteur d'actualisation
  - `theta` (1e-5): Seuil de convergence

### Monte Carlo Methods

#### 3. Monte Carlo ES (Exploring Starts)
- **Fichier**: `algos/monte_carlo.py`
- **Classe**: `MonteCarloES`
- **Description**: Monte Carlo avec exploring starts
- **Hyperparamètres**:
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration

#### 4. On-policy First Visit Monte Carlo
- **Fichier**: `algos/monte_carlo.py`
- **Classe**: `OnPolicyMonteCarlo`
- **Description**: Monte Carlo on-policy avec first-visit
- **Hyperparamètres**:
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration

#### 5. Off-policy Monte Carlo
- **Fichier**: `algos/monte_carlo.py`
- **Classe**: `OffPolicyMonteCarlo`
- **Description**: Monte Carlo off-policy avec importance sampling
- **Hyperparamètres**:
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration

### Temporal Difference Learning

#### 6. SARSA
- **Fichier**: `algos/sarsa.py`
- **Classe**: `SARSAAgent`
- **Description**: Algorithme on-policy de TD Learning
- **Hyperparamètres**:
  - `alpha` (0.1): Taux d'apprentissage
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration

#### 7. Q-Learning
- **Fichier**: `algos/q_learning.py`
- **Classe**: `QLearningAgent`
- **Description**: Algorithme off-policy de TD Learning
- **Hyperparamètres**:
  - `alpha` (0.1): Taux d'apprentissage
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration

#### 8. Expected SARSA
- **Fichier**: `algos/expected_sarsa.py`
- **Classe**: `ExpectedSARSAAgent`
- **Description**: Variante de SARSA utilisant l'espérance
- **Hyperparamètres**:
  - `alpha` (0.1): Taux d'apprentissage
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration

### Planning

#### 9. Dyna-Q
- **Fichier**: `algos/dyna_q.py`
- **Classe**: `DynaQAgent`
- **Description**: Q-Learning + modèle de l'environnement pour planning
- **Hyperparamètres**:
  - `alpha` (0.1): Taux d'apprentissage
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration
  - `n_planning_steps` (5): Nombre d'étapes de planning

#### 10. Dyna-Q+
- **Fichier**: `algos/dyna_q_plus.py`
- **Classe**: `DynaQPlusAgent`
- **Description**: Dyna-Q avec bonus d'exploration pour environnements changeants
- **Hyperparamètres**:
  - `alpha` (0.1): Taux d'apprentissage
  - `gamma` (0.99): Facteur d'actualisation
  - `epsilon` (0.1): Taux d'exploration
  - `n_planning_steps` (5): Nombre d'étapes de planning
  - `kappa` (1e-3): Poids du bonus d'exploration
  - `tau` (1000): Temps de vie pour transitions "anciennes"

## 🌍 Environnements disponibles

### 1. Line World
- **Fichier**: `envs/lineworld.py`
- **Classe**: `LineWorld`
- **Description**: Environnement 1D avec obstacles, pièges, récompenses
- **Actions**: 0=gauche, 1=droite, 2=rester, 3=sauter, 4=sprint
- **État**: Position + énergie + clés collectées

### 2. Grid World
- **Fichier**: `envs/gridworld.py`
- **Classe**: `GridWorld`
- **Description**: Environnement 2D avec obstacles, pièges mobiles, récompenses
- **Actions**: 0=haut, 1=bas, 2=gauche, 3=droite
- **État**: Position (x, y)

### 3. Two-Round Rock Paper Scissors
- **Fichier**: `envs/rps.py`
- **Classe**: `TwoRoundRPS`
- **Description**: 2 rounds de Pierre-Papier-Ciseaux
- **Actions**: 0=Rock, 1=Paper, 2=Scissors
- **État**: Round actuel + choix précédents

### 4. Monty Hall Level 1
- **Fichier**: `envs/monty_hall_level1.py`
- **Classe**: `MontyHallLevel1`
- **Description**: Problème de Monty Hall avec 3 portes
- **Actions**: Étape 1: 0-2 (choisir porte), Étape 2: 0=garder, 1=changer
- **État**: Étape + portes choisies/retirées

### 5. Monty Hall Level 2
- **Fichier**: `envs/monty_hall_level2.py`
- **Classe**: `MontyHallLevel2`
- **Description**: Problème de Monty Hall avec 5 portes (4 actions)
- **Actions**: Variable selon le nombre de portes disponibles
- **État**: Étape + portes disponibles

## 🚀 Utilisation de base

### Exemple 1: Q-Learning sur Line World

```python
from envs.lineworld import LineWorld
from algos.q_learning import QLearningAgent

# Créer l'environnement
env = LineWorld(length=10)

# Créer l'agent
agent = QLearningAgent(
    env,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1
)

# Entraîner
agent.train(num_episodes=1000, verbose=True)

# Évaluer
results = agent.evaluate(num_episodes=100)
print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Success rate: {results['success_rate']*100:.1f}%")

# Sauvegarder
agent.save("models/qlearning_lineworld.pkl")
```

### Exemple 2: Policy Iteration sur Grid World

```python
from envs.gridworld import GridWorld
from algos.policy_iteration import PolicyIteration

# Créer l'environnement
env = GridWorld(width=5, height=5)

# Créer l'agent
agent = PolicyIteration(
    env,
    gamma=0.99,
    theta=1e-5
)

# Entraîner
agent.train(num_episodes=100, verbose=True)

# Utiliser la politique apprise
state = env.reset()
action = agent.select_action(state, training=False)
```

### Exemple 3: Monte Carlo ES sur RPS

```python
from envs.rps import TwoRoundRPS
from algos.monte_carlo import MonteCarloES

# Créer l'environnement
env = TwoRoundRPS()

# Créer l'agent
agent = MonteCarloES(
    env,
    gamma=0.99,
    epsilon=0.1
)

# Entraîner
agent.train(num_episodes=500, verbose=True)

# Évaluer
results = agent.evaluate(num_episodes=100)
```

## 🧪 Tests complets

### Tester tous les algorithmes sur tous les environnements

```bash
# Tester tout
python test_all_algos_envs.py --all

# Tester avec sortie détaillée
python test_all_algos_envs.py --all --verbose

# Tester une combinaison spécifique
python test_all_algos_envs.py --algo Q-Learning --env LineWorld --episodes 1000

# Tester avec sortie détaillée
python test_all_algos_envs.py --algo Q-Learning --env LineWorld --episodes 1000 --verbose
```

### Résultats

Les résultats sont sauvegardés dans le dossier `results/`:
- Un fichier JSON par combinaison algorithme/environnement
- Un rapport complet avec tous les résultats

### Format des résultats

```json
{
  "algorithm": "Q-Learning",
  "environment": "LineWorld",
  "hyperparameters": {...},
  "training": {
    "num_episodes": 1000,
    "training_time": 12.34,
    "convergence_episode": 500,
    "final_mean_reward": 15.2,
    "best_reward": 20.0
  },
  "evaluation": {
    "mean_reward": 15.5,
    "std_reward": 2.3,
    "success_rate": 0.85,
    "mean_steps": 25.4
  }
}
```

## 📊 Exemples d'utilisation

### Comparer plusieurs algorithmes

```python
from envs.lineworld import LineWorld
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
from algos.monte_carlo import OnPolicyMonteCarlo

env = LineWorld(length=10)

algorithms = {
    'Q-Learning': QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1),
    'SARSA': SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1),
    'Monte Carlo': OnPolicyMonteCarlo(env, gamma=0.99, epsilon=0.1)
}

results = {}
for name, agent in algorithms.items():
    print(f"\nTraining {name}...")
    agent.train(num_episodes=1000, verbose=False)
    eval_results = agent.evaluate(num_episodes=100)
    results[name] = eval_results['mean_reward']
    print(f"{name}: {eval_results['mean_reward']:.2f}")

# Afficher le meilleur
best = max(results, key=results.get)
print(f"\nMeilleur algorithme: {best} ({results[best]:.2f})")
```

### Étude d'hyperparamètres

```python
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent
import numpy as np

env = GridWorld(width=5, height=5)

# Tester différentes valeurs d'alpha
alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
results = {}

for alpha in alphas:
    agent = QLearningAgent(env, alpha=alpha, gamma=0.99, epsilon=0.1)
    agent.train(num_episodes=1000, verbose=False)
    eval_results = agent.evaluate(num_episodes=100)
    results[alpha] = eval_results['mean_reward']
    print(f"Alpha {alpha}: {eval_results['mean_reward']:.2f}")

# Trouver le meilleur alpha
best_alpha = max(results, key=results.get)
print(f"\nMeilleur alpha: {best_alpha} ({results[best_alpha]:.2f})")
```

### Visualiser l'apprentissage

```python
import matplotlib.pyplot as plt
from envs.lineworld import LineWorld
from algos.q_learning import QLearningAgent

env = LineWorld(length=10)
agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# Entraîner
agent.train(num_episodes=1000, verbose=False)

# Visualiser les rewards par épisode
plt.figure(figsize=(10, 6))
plt.plot(agent.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Q-Learning on Line World - Training Progress')
plt.grid(True)
plt.show()

# Moyenne mobile sur 100 épisodes
window = 100
moving_avg = np.convolve(agent.episode_rewards, np.ones(window)/window, mode='valid')
plt.figure(figsize=(10, 6))
plt.plot(moving_avg)
plt.xlabel('Episode')
plt.ylabel('Average Reward (100 episodes)')
plt.title('Q-Learning - Moving Average')
plt.grid(True)
plt.show()
```

## 📈 Interprétation des résultats

### Métriques importantes

1. **Mean Reward**: Récompense moyenne par épisode
   - Plus élevé = meilleur
   - Comparez entre algorithmes

2. **Success Rate**: Taux de succès (épisodes avec reward > 0)
   - Plus élevé = meilleur
   - Indique la fiabilité de l'algorithme

3. **Convergence Episode**: Épisode où l'algorithme a convergé
   - Plus tôt = meilleur
   - Indique la vitesse d'apprentissage

4. **Training Time**: Temps d'entraînement
   - Comparez l'efficacité computationnelle

### Quand utiliser quel algorithme?

#### Dynamic Programming (Policy/Value Iteration)
- ✅ **Quand**: Modèle de l'environnement disponible, états discrets
- ❌ **Quand**: Grands espaces d'états, pas de modèle
- **Environnements**: Line World, Grid World (petits)

#### Monte Carlo
- ✅ **Quand**: Épisodes complets disponibles, pas besoin de modèle
- ❌ **Quand**: Épisodes très longs, besoin d'apprentissage rapide
- **Environnements**: Tous (surtout RPS, Monty Hall)

#### TD Learning (SARSA, Q-Learning, Expected SARSA)
- ✅ **Quand**: Apprentissage en ligne, pas de modèle nécessaire
- ❌ **Quand**: Besoin de convergence très rapide
- **Environnements**: Tous (surtout Line World, Grid World)

#### Planning (Dyna-Q, Dyna-Q+)
- ✅ **Quand**: Modèle peut être appris, besoin d'efficacité
- ❌ **Quand**: Environnement non-stationnaire (sauf Dyna-Q+)
- **Environnements**: Line World, Grid World

### Conseils pour le choix d'hyperparamètres

1. **Gamma (facteur d'actualisation)**
   - Proche de 1 (0.99): Privilégie les récompenses futures
   - Proche de 0 (0.5): Privilégie les récompenses immédiates
   - **Recommandation**: 0.9-0.99 pour la plupart des cas

2. **Alpha (taux d'apprentissage)**
   - Trop élevé (>0.5): Instabilité
   - Trop faible (<0.01): Apprentissage lent
   - **Recommandation**: 0.1 pour commencer, ajuster selon les résultats

3. **Epsilon (exploration)**
   - Trop élevé (>0.3): Trop d'exploration, peu d'exploitation
   - Trop faible (<0.01): Peu d'exploration
   - **Recommandation**: 0.1, peut être réduit progressivement

4. **N Planning Steps (Dyna-Q)**
   - Plus élevé: Plus de planning, mais plus coûteux
   - **Recommandation**: 5-10 pour commencer

## 🔧 Dépannage

### Erreurs communes

1. **"action_space not found"**
   - Solution: Vérifiez que l'environnement a `n_actions()` ou `action_space`

2. **"State key error"**
   - Solution: Les états doivent être hashables (tuples, pas listes)

3. **"Convergence lente"**
   - Solution: Ajustez les hyperparamètres (alpha, epsilon, gamma)

4. **"Mémoire insuffisante"**
   - Solution: Réduisez la taille de l'environnement ou le nombre d'épisodes

## 📝 Notes importantes

1. **Sauvegarde**: Utilisez `agent.save(path)` pour sauvegarder les agents entraînés
2. **Chargement**: Utilisez `agent.load(path)` pour charger un agent sauvegardé
3. **Évaluation**: Toujours évaluer en mode `training=False` pour des résultats fiables
4. **Reproductibilité**: Utilisez `random.seed()` pour des résultats reproductibles

## 🎯 Prochaines étapes

1. Tester tous les algorithmes sur tous les environnements
2. Comparer les performances
3. Étudier l'impact des hyperparamètres
4. Analyser les politiques apprises
5. Préparer le rapport et la soutenance

Bon apprentissage ! 🚀



