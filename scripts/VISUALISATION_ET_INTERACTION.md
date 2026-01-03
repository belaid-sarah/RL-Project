# Visualisation et Interaction - Guide Complet

Ce document explique comment utiliser tous les outils de visualisation et d'interaction du projet.

## ✅ Exigences du Projet Couvertes

Le projet répond à toutes les exigences de visualisation et d'interaction :

1. ✅ **Visualisation agréable** : Interface graphique Pygame pour chaque environnement
2. ✅ **Replay pas à pas** : Dérouler une stratégie apprise sans réentraîner (pour la soutenance)
3. ✅ **Agent humain** : Interaction manuelle sur chaque environnement

---

## 🎮 1. Interaction Humaine (Agent Humain)

Permet de jouer manuellement avec chaque environnement pour vérifier les règles.

### LineWorld
```bash
python scripts/mainLineworld.py
```
- **Contrôles** : Flèches gauche/droite ou A/D
- **R** : Reset
- **Q** : Quitter

### GridWorld
```bash
python scripts/mainGridwolrd.py
```
- **Contrôles** : Flèches directionnelles ou WASD
- **R** : Reset
- **Q** : Quitter

### Two-Round Rock Paper Scissors
```bash
python scripts/main_rps_human.py
```
- **Contrôles** : 0 (Pierre), 1 (Feuille), 2 (Ciseaux)
- **R** : Reset
- **Q** : Quitter

### Monty Hall Level 1
```bash
python scripts/main_rl.py
```
- **Contrôles** : Choix de porte (0, 1, 2) puis garder/changer
- **R** : Reset
- **Q** : Quitter

### Monty Hall Level 2
```bash
python scripts/mainlevel2.py
```
- **Contrôles** : Choix de portes successifs
- **R** : Reset
- **Q** : Quitter

---

## 📊 2. Visualisation des Algorithmes (Agent Entraîné)

Visualise un agent entraîné en action avec Pygame.

### Utilisation

```bash
python scripts/visualize_rl.py --env <environnement> --algo <algorithme> [--model <chemin>]
```

### Exemples

```bash
# LineWorld avec Q-Learning
python scripts/visualize_rl.py --env lineworld --algo qlearning --model models/qlearning_lineworld.pkl

# GridWorld avec SARSA
python scripts/visualize_rl.py --env gridworld --algo sarsa --model models/sarsa_gridworld.pkl

# Sans modèle (entraînement à la volée)
python scripts/visualize_rl.py --env lineworld --algo qlearning --episodes 1000
```

### Contrôles
- **ESPACE** : Pause/Reprendre
- **R** : Reset (nouvel épisode)
- **+/-** : Augmenter/Diminuer la vitesse
- **Q** : Quitter

### Environnements supportés
- `lineworld` : LineWorldSimple
- `gridworld` : GridWorldSimple

### Algorithmes supportés
- `qlearning` : Q-Learning
- `sarsa` : SARSA

---

## 🎯 3. Replay Pas à Pas (Pour la Soutenance)

Déroule une stratégie apprise pas à pas **sans réentraîner**. Parfait pour la démonstration lors de la soutenance.

### Utilisation

```bash
python scripts/replay_policy.py --env <environnement> --algo <algorithme> --model <chemin>
```

### Exemples

```bash
# Replay Q-Learning sur LineWorld
python scripts/replay_policy.py --env lineworld --algo qlearning --model models/qlearning_lineworld.pkl

# Replay Policy Iteration sur GridWorld
python scripts/replay_policy.py --env gridworld --algo policy_iteration --model models/policy_iteration_gridworld.pkl

# Replay Value Iteration sur LineWorld
python scripts/replay_policy.py --env lineworld --algo value_iteration --model models/value_iteration_lineworld.pkl
```

### Contrôles
- **Flèche droite (→)** : Étape suivante
- **ESPACE** : Auto-play / Pause (défile automatiquement)
- **R** : Reset (recommence l'épisode)
- **Q** : Quitter

### Caractéristiques
- ✅ Affiche la politique apprise
- ✅ Affiche les Q-values ou V-values
- ✅ Pas à pas manuel ou automatique
- ✅ Aucun réentraînement nécessaire
- ✅ Parfait pour la soutenance

### Environnements supportés
- `lineworld` : LineWorldSimple
- `gridworld` : GridWorldSimple

### Algorithmes supportés
- `qlearning` : Q-Learning
- `sarsa` : SARSA
- `policy_iteration` : Policy Iteration
- `value_iteration` : Value Iteration

---

## 📁 4. Préparer les Modèles pour la Soutenance

Avant la soutenance, entraînez et sauvegardez tous les agents :

```bash
python scripts/entrainer_tous_agents.py
```

Ce script va :
- Entraîner tous les algorithmes clés
- Les sauvegarder dans `models/`
- Être prêt pour la démonstration

Les modèles sauvegardés peuvent ensuite être utilisés avec `replay_policy.py` et `visualize_rl.py`.

---

## 🎨 Interface Graphique

Tous les outils utilisent Pygame avec :
- ✅ Visualisation claire et agréable
- ✅ Couleurs pour différencier les éléments (agent, goal, obstacles, etc.)
- ✅ Affichage des informations (reward, steps, Q-values, etc.)
- ✅ Contrôles intuitifs

---

## 📝 Résumé des Scripts

| Script | Fonction | Pour qui |
|--------|----------|----------|
| `main*.py` | Interaction humaine | Vérification des règles |
| `visualize_rl.py` | Visualisation agent entraîné | Démonstration en temps réel |
| `replay_policy.py` | Replay pas à pas | **Soutenance** |
| `entrainer_tous_agents.py` | Entraînement batch | Préparation |

---

## ✅ Checklist Soutenance

- [ ] Entraîner tous les agents : `python scripts/entrainer_tous_agents.py`
- [ ] Tester le replay : `python scripts/replay_policy.py --env lineworld --algo qlearning --model models/qlearning_lineworld.pkl`
- [ ] Tester l'interaction humaine : `python scripts/mainLineworld.py`
- [ ] Préparer les modèles pour chaque environnement/algorithme à démontrer

