# 🧪 Comment Tester les Algorithmes - Guide Simple

## 📌 Méthode 1 : Test Rapide d'un Algorithme (Le Plus Simple)

### Exemple : Tester Q-Learning sur GridWorld

Créez un fichier `test_simple.py` :

```python
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent

# 1. Créer l'environnement
env = GridWorld(width=5, height=5)

# 2. Créer l'agent
agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# 3. Entraîner (500 épisodes)
print("Entraînement en cours...")
agent.train(num_episodes=500, verbose=True)

# 4. Évaluer les performances
print("\nÉvaluation...")
results = agent.evaluate(num_episodes=100)
print(f"Récompense moyenne: {results['mean_reward']:.2f}")
print(f"Taux de succès: {results['success_rate']*100:.1f}%")
```

**Exécutez :**
```bash
python test_simple.py
```

---

## 📌 Méthode 2 : Tester TOUS les Algorithmes sur UN Environnement

Créez un fichier `test_tous_algos.py` :

```python
from envs.gridworld import GridWorld
from algos.policy_iteration import PolicyIteration
from algos.value_iteration import ValueIteration
from algos.q_learning import QLearningAgent
from algos.sarsa import SARSAAgent
from algos.monte_carlo import MonteCarloES, OnPolicyMonteCarlo
from algos.expected_sarsa import ExpectedSARSAAgent
from algos.dyna_q import DynaQAgent

# Créer l'environnement
env = GridWorld(width=5, height=5)

# Liste de tous les algorithmes à tester
algorithmes = {
    'Q-Learning': QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1),
    'SARSA': SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1),
    'Expected SARSA': ExpectedSARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1),
    'Monte Carlo ES': MonteCarloES(env, gamma=0.99, epsilon=0.1),
    'On-Policy MC': OnPolicyMonteCarlo(env, gamma=0.99, epsilon=0.1),
    'Dyna-Q': DynaQAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5),
}

# Tester chaque algorithme
resultats = {}
for nom, agent in algorithmes.items():
    print(f"\n{'='*50}")
    print(f"Test de {nom}")
    print(f"{'='*50}")
    
    # Entraîner
    agent.train(num_episodes=500, verbose=False)
    
    # Évaluer
    eval_results = agent.evaluate(num_episodes=100)
    resultats[nom] = eval_results['mean_reward']
    
    print(f"✅ {nom}: Récompense moyenne = {eval_results['mean_reward']:.2f}")

# Afficher le classement
print(f"\n{'='*50}")
print("CLASSEMENT DES ALGORITHMES")
print(f"{'='*50}")
for i, (nom, score) in enumerate(sorted(resultats.items(), key=lambda x: x[1], reverse=True), 1):
    print(f"{i}. {nom:20s} : {score:7.2f}")
```

**Exécutez :**
```bash
python test_tous_algos.py
```

---

## 📌 Méthode 3 : Tester UN Algorithme sur TOUS les Environnements

Créez un fichier `test_tous_envs.py` :

```python
from envs.lineworld import LineWorld
from envs.gridworld import GridWorld
from envs.rps import TwoRoundRPS
from envs.monty_hall_level1 import MontyHallLevel1
from envs.monty_hall_level2 import MontyHallLevel2
from algos.q_learning import QLearningAgent

# Liste de tous les environnements
environnements = {
    'LineWorld': LineWorld(length=10),
    'GridWorld': GridWorld(width=5, height=5),
    'TwoRoundRPS': TwoRoundRPS(),
    'MontyHall Level 1': MontyHallLevel1(),
    'MontyHall Level 2': MontyHallLevel2(),
}

# Tester Q-Learning sur chaque environnement
resultats = {}
for nom_env, env in environnements.items():
    print(f"\n{'='*50}")
    print(f"Test sur {nom_env}")
    print(f"{'='*50}")
    
    # Créer l'agent
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # Entraîner
    agent.train(num_episodes=500, verbose=False)
    
    # Évaluer
    eval_results = agent.evaluate(num_episodes=100)
    resultats[nom_env] = eval_results['mean_reward']
    
    print(f"✅ {nom_env}: Récompense moyenne = {eval_results['mean_reward']:.2f}")

# Afficher les résultats
print(f"\n{'='*50}")
print("RÉSULTATS PAR ENVIRONNEMENT")
print(f"{'='*50}")
for env, score in resultats.items():
    print(f"{env:20s} : {score:7.2f}")
```

**Exécutez :**
```bash
python test_tous_envs.py
```

---

## 📌 Méthode 4 : Utiliser le Script de Test Complet (Recommandé pour le Projet)

Le script `test_all_algos_envs.py` teste **TOUS** les algorithmes sur **TOUS** les environnements automatiquement.

### Option A : Tester TOUT

```bash
python test_all_algos_envs.py --all
```

Cela va :
- Tester les 10 algorithmes sur les 5 environnements
- Sauvegarder les résultats dans `results/`
- Afficher un résumé

### Option B : Tester une Combinaison Spécifique

```bash
# Tester Q-Learning sur GridWorld avec 1000 épisodes
python test_all_algos_envs.py --algo Q-Learning --env GridWorld --episodes 1000

# Avec sortie détaillée
python test_all_algos_envs.py --algo Q-Learning --env GridWorld --episodes 1000 --verbose
```

### Option C : Voir les Options Disponibles

```bash
python test_all_algos_envs.py
```

Cela affiche :
- Les algorithmes disponibles
- Les environnements disponibles
- Comment utiliser le script

---

## 📌 Méthode 5 : Test Visuel avec Pygame (Comme votre mainGridwolrd.py)

Créez un fichier `test_visuel.py` pour voir l'agent apprendre en temps réel :

```python
import pygame
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent

# Initialisation Pygame
pygame.init()
cell_size = 60
width_cells, height_cells = 5, 5
screen_width = width_cells * cell_size
screen_height = height_cells * cell_size + 100
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("GridWorld - Q-Learning")

font = pygame.font.Font(None, 24)

# Créer l'environnement et l'agent
env = GridWorld(width=width_cells, height=height_cells)
agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# Entraîner l'agent (100 épisodes)
print("Entraînement...")
agent.train(num_episodes=100, verbose=False)
print("Entraînement terminé!")

# Maintenant visualiser la politique apprise
state = env.reset()
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                state = env.reset()
    
    # Dessiner la grille
    screen.fill((20, 20, 25))
    for x in range(env.width):
        for y in range(env.height):
            color = (80, 80, 90)
            if (x, y) in env.obstacles:
                color = (40, 40, 40)
            elif (x, y) == env.goal:
                color = (50, 200, 50)
            if (x, y) == state:
                color = (100, 255, 100)
            
            pygame.draw.rect(screen, color, 
                           (x*cell_size + 2, y*cell_size + 2, 
                            cell_size-4, cell_size-4))
    
    # Afficher les infos
    info_text = font.render(f"Position: {state} | Steps: {env.step_count}", True, (255, 255, 255))
    screen.blit(info_text, (10, height_cells * cell_size + 10))
    
    # Utiliser la politique apprise
    if not env.done:
        action = agent.select_action(state, training=False)  # Pas d'exploration
        state, reward, done, info = env.step(action)
        pygame.time.wait(200)  # Pause pour voir
    else:
        pygame.time.wait(1000)
        state = env.reset()
    
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
```

**Exécutez :**
```bash
python test_visuel.py
```

---

## 🎯 Résumé : Quelle Méthode Utiliser ?

| Situation | Méthode | Commande |
|-----------|---------|----------|
| **Test rapide d'un algo** | Méthode 1 | Créer `test_simple.py` |
| **Comparer tous les algos** | Méthode 2 | Créer `test_tous_algos.py` |
| **Tester un algo partout** | Méthode 3 | Créer `test_tous_envs.py` |
| **Test complet automatique** | Méthode 4 | `python test_all_algos_envs.py --all` |
| **Visualisation** | Méthode 5 | Créer `test_visuel.py` |

---

## 💡 Exemple Concret pour Votre Projet

Pour votre projet, je recommande de créer ces fichiers :

### 1. `test_rapide.py` - Pour tester rapidement
```python
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent

env = GridWorld(width=5, height=5)
agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
agent.train(num_episodes=500, verbose=True)
results = agent.evaluate(num_episodes=100)
print(f"Résultat: {results['mean_reward']:.2f}")
```

### 2. `test_complet.py` - Pour le rapport
```python
# Utilisez le script test_all_algos_envs.py
# python test_all_algos_envs.py --all
```

### 3. `test_hyperparametres.py` - Pour étudier les hyperparamètres
```python
from envs.gridworld import GridWorld
from algos.q_learning import QLearningAgent

env = GridWorld(width=5, height=5)

# Tester différentes valeurs d'alpha
for alpha in [0.01, 0.05, 0.1, 0.2]:
    agent = QLearningAgent(env, alpha=alpha, gamma=0.99, epsilon=0.1)
    agent.train(num_episodes=500, verbose=False)
    results = agent.evaluate(num_episodes=100)
    print(f"Alpha {alpha}: {results['mean_reward']:.2f}")
```

---

## ❓ Questions Fréquentes

**Q: Combien d'épisodes pour l'entraînement ?**
- R: Commencez avec 500-1000. Augmentez si l'agent n'a pas convergé.

**Q: Comment savoir si ça marche ?**
- R: Regardez la récompense moyenne. Elle devrait augmenter avec le temps.

**Q: Les résultats sont sauvegardés où ?**
- R: Dans le dossier `results/` (créé automatiquement).

**Q: Comment comparer deux algorithmes ?**
- R: Utilisez la Méthode 2 ou le script `test_all_algos_envs.py`.

---

**Besoin d'aide ?** Consultez `GUIDE_UTILISATION.md` pour plus de détails !

