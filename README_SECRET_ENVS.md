# Test des Algorithmes sur les Environnements Secrets

Ce document explique comment tester tous les algorithmes de RL sur les environnements secrets (SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3).

## Prérequis

### 1. Bibliothèques DLL/SO

Les environnements secrets nécessitent des bibliothèques natives qui doivent être placées dans le dossier `libs/` :

- **Windows**: `libs/secret_envs.dll`
- **Linux**: `libs/libsecret_envs.so`
- **macOS Intel**: `libs/libsecret_envs_intel_macos.dylib`
- **macOS Apple Silicon**: `libs/libsecret_envs.dylib`

Ces fichiers sont normalement fournis par l'enseignant. Si vous ne les avez pas, créez le dossier `libs/` et placez-y les fichiers appropriés.

### 2. Structure du projet

```
RL-PROJECT/
├── libs/                    # Dossier pour les DLL/SO (à créer si nécessaire)
│   ├── secret_envs.dll      # Windows
│   ├── libsecret_envs.so    # Linux
│   └── libsecret_envs.dylib # macOS
├── envs/
│   ├── secret_envs_wrapper.py  # Wrapper fourni par l'enseignant
│   └── secret_env_adapter.py   # Adaptateur créé pour notre interface
├── algos/                   # Tous nos algorithmes
└── test_secret_envs.py     # Script de test principal
```

## Utilisation

### Tester tous les algorithmes sur tous les environnements secrets

```bash
python test_secret_envs.py --all
```

Cette commande va :
1. Tester chaque algorithme sur chaque environnement secret
2. Sauvegarder les résultats dans `results/`
3. Afficher un résumé avec le meilleur algorithme pour chaque environnement

### Tester un algorithme spécifique sur un environnement spécifique

```bash
python test_secret_envs.py --algo Q-Learning --env SecretEnv0 --episodes 2000
```

### Options disponibles

- `--all`: Exécute tous les tests
- `--algo <nom>`: Nom de l'algorithme à tester
- `--env <nom>`: Nom de l'environnement à tester
- `--episodes <n>`: Nombre d'épisodes d'entraînement (défaut: 2000)
- `--verbose`: Affiche les détails pendant l'entraînement

### Algorithmes disponibles

- `PolicyIteration`
- `ValueIteration`
- `MonteCarloES`
- `OnPolicyMonteCarlo`
- `OffPolicyMonteCarlo`
- `SARSA`
- `Q-Learning`
- `ExpectedSARSA`
- `Dyna-Q`
- `Dyna-Q+`

### Environnements disponibles

- `SecretEnv0`
- `SecretEnv1`
- `SecretEnv2`
- `SecretEnv3`

## Résultats

Les résultats sont sauvegardés dans le dossier `results/` :

1. **Fichiers individuels**: Un fichier JSON par combinaison algorithme/environnement
   - Format: `{Algorithme}_{Environnement}_{Timestamp}.json`

2. **Rapport complet**: Un fichier JSON avec tous les résultats
   - Format: `secret_envs_complete_report_{Timestamp}.json`

### Structure des résultats

Chaque fichier JSON contient :
- `algorithm`: Nom de l'algorithme
- `environment`: Nom de l'environnement
- `hyperparameters`: Hyperparamètres utilisés
- `training`: Statistiques d'entraînement (temps, épisodes, convergence)
- `evaluation`: Statistiques d'évaluation (mean reward, success rate, steps)
- `success`: Si le test a réussi
- `error`: Message d'erreur si échec

## Identification du meilleur algorithme

Le script affiche automatiquement :
- Un résumé par environnement avec les algorithmes triés par performance
- Le meilleur algorithme pour chaque environnement (🏆)
- Les métriques clés : mean reward, success rate, mean steps, training time

## Exemple de sortie

```
================================================================
SUMMARY - SECRET ENVIRONMENTS
================================================================

SecretEnv0:
------------------------------------------------------------
🏆  1. Q-Learning          | Reward:  45.23 ±  2.15 | Success:  95.0% | Steps:   12.3 | Time:  15.23s
    2. SARSA              | Reward:  43.12 ±  2.45 | Success:  92.0% | Steps:   13.1 | Time:  14.87s
    3. Dyna-Q             | Reward:  42.89 ±  2.67 | Success:  91.0% | Steps:   13.5 | Time:  18.45s
...

================================================================
🏆 MEILLEURS ALGORITHMES PAR ENVIRONNEMENT
================================================================

SecretEnv0:
  🥇 Meilleur algorithme: Q-Learning
     - Mean Reward: 45.23 ± 2.15
     - Success Rate: 95.0%
     - Mean Steps: 12.3
     - Training Time: 15.23s
```

## Dépannage

### Erreur: "Could not find module 'libs/secret_envs.dll'"

**Solution**: Vérifiez que :
1. Le dossier `libs/` existe
2. Le fichier DLL/SO approprié est présent dans `libs/`
3. Le nom du fichier correspond à votre système d'exploitation

### Erreur: "Environnement non supporte pour Policy Iteration"

**Solution**: Les environnements secrets sont maintenant supportés grâce à l'adaptateur. Si cette erreur persiste, vérifiez que `secret_env_adapter.py` est bien présent dans `envs/`.

### Les tests sont très longs

**Solution**: Réduisez le nombre d'épisodes dans `CONFIG['num_episodes']` dans `test_secret_envs.py`.

## Notes techniques

### Adaptateur SecretEnvAdapter

L'adaptateur `SecretEnvAdapter` convertit l'interface des environnements secrets vers notre interface `BaseEnv` standard :

- `reset()` → retourne l'état initial
- `step(action)` → retourne `(next_state, reward, done, info)`
- `sample_action()` → retourne une action aléatoire
- `action_space` → liste des actions possibles

### Support des algorithmes basés sur modèle

Policy Iteration et Value Iteration utilisent les méthodes MDP des environnements secrets :
- `num_states()`: nombre d'états
- `num_actions()`: nombre d'actions
- `p(s, a, s_p, r_index)`: probabilité de transition
- `reward(r_index)`: valeur du reward

Ces méthodes permettent de construire le modèle MDP sans exploration.

