# Scripts du Projet

Ce dossier contient tous les scripts utilitaires et d'interaction pour le projet.

## Scripts d'Interaction Humaine (Agent Humain)

Ces scripts permettent de jouer manuellement avec les environnements (requis par le projet) :

- **mainLineworld.py** : Interaction humaine avec LineWorld
- **mainGridwolrd.py** : Interaction humaine avec GridWorld  
- **main_rps_human.py** : Interaction humaine avec Two-Round Rock Paper Scissors
- **mainlevel2.py** : Interaction humaine avec Monty Hall Level 2
- **main_rl.py** : Script principal pour différents environnements

## Scripts de Visualisation

- **visualize_rl.py** : Visualisation interactive des environnements avec Pygame
- **replay_policy.py** : Replay pas à pas d'une politique apprise (pour la soutenance)

## Scripts d'Entraînement

- **entrainer_tous_agents.py** : Entraîne et sauvegarde tous les agents importants pour la soutenance

## Scripts d'Exemple

- **exemple_construction_modele.py** : Exemple de construction du modèle MDP pour Policy/Value Iteration

## Utilisation

### Interaction Humaine

```bash
# Jouer avec LineWorld
python scripts/mainLineworld.py

# Jouer avec GridWorld
python scripts/mainGridwolrd.py

# Jouer avec RPS
python scripts/main_rps_human.py
```

### Visualisation

```bash
# Visualiser un agent entraîné
python scripts/visualize_rl.py --env lineworld --algo qlearning --model models/qlearning_lineworld.pkl

# Replay d'une politique
python scripts/replay_policy.py --env lineworld --algo qlearning --model models/qlearning_lineworld.pkl
```

### Entraînement

```bash
# Entraîner tous les agents
python scripts/entrainer_tous_agents.py
```

