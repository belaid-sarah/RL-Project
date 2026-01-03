# Dossier de Tests

Ce dossier contient tous les scripts de test pour le projet Reinforcement Learning.

## Structure

```
tests/
├── standard_envs/          # Tests sur les environnements standards
│   ├── test_all_algos_envs.py      # Test tous les algorithmes sur tous les environnements standards
│   └── test_one_algo_all_envs.py   # Test un algorithme spécifique sur tous les environnements
│
├── secret_envs/            # Tests sur les environnements secrets
│   ├── test_secret_envs.py         # Test tous les algorithmes sur tous les environnements secrets
│   └── test_secret_adapter.py     # Test de l'adaptateur pour les environnements secrets
│
└── utils/                  # Utilitaires de test et d'analyse
    ├── compare_and_analyze.py     # Comparaison et analyse des résultats
    └── tester_agents_sauvegardes.py  # Test des agents sauvegardés
```

## Utilisation

### Tests sur environnements standards

```bash
# Tester tous les algorithmes sur tous les environnements standards
python tests/standard_envs/test_all_algos_envs.py --all

# Tester un algorithme spécifique
python tests/standard_envs/test_all_algos_envs.py --algo Q-Learning --env LineWorld
```

### Tests sur environnements secrets

```bash
# Tester tous les algorithmes sur tous les environnements secrets
python tests/secret_envs/test_secret_envs.py --all

# Tester un algorithme spécifique
python tests/secret_envs/test_secret_envs.py --algo Q-Learning --env SecretEnv0
```

### Utilitaires

```bash
# Analyser les résultats
python tests/utils/compare_and_analyze.py

# Tester les agents sauvegardés
python tests/utils/tester_agents_sauvegardes.py
```

## Notes

- Tous les scripts ajoutent automatiquement le répertoire parent au `sys.path` pour les imports
- Les résultats sont sauvegardés dans le dossier `results/` à la racine du projet
- Les modèles sauvegardés sont dans le dossier `models/` à la racine du projet

