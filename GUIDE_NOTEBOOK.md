# 📓 Guide : Utiliser le Notebook Jupyter

## 🚀 Démarrage Rapide

### 1. Installer Jupyter (si pas déjà installé)

```bash
pip install jupyter notebook
```

### 2. Lancer Jupyter Notebook

```bash
jupyter notebook
```

Cela ouvre votre navigateur avec l'interface Jupyter.

### 3. Ouvrir le Notebook

- Cliquez sur `rapport_projet_rl.ipynb` dans la liste des fichiers

---

## 📝 Structure du Notebook

Le notebook contient :

1. **Introduction** : Objectifs et structure du projet
2. **Méthodologie** : Algorithmes, environnements, hyperparamètres
3. **Résultats** : Analyses par environnement avec graphiques
4. **Analyse et Interprétation** : Comparaison des algorithmes
5. **Conclusion** : Résumé et recommandations

---

## 🎯 Comment Utiliser le Notebook

### Étape 1 : Exécuter les Tests (si pas encore fait)

Avant d'utiliser le notebook, vous devez avoir des résultats :

```bash
# Tester tous les algorithmes sur tous les environnements
python test_all_algos_envs.py --all
```

Cela génère des fichiers JSON dans `results/` que le notebook chargera.

### Étape 2 : Ouvrir le Notebook

```bash
jupyter notebook rapport_projet_rl.ipynb
```

### Étape 3 : Exécuter les Cellules

1. **Exécuter toutes les cellules** : `Cell` → `Run All`
2. **Exécuter cellule par cellule** : `Shift + Enter`
3. **Ajouter des cellules** : `Insert` → `Insert Cell Above/Below`

### Étape 4 : Personnaliser

- Compléter les sections "[À compléter]"
- Ajouter vos propres analyses
- Modifier les graphiques si nécessaire

---

## 📊 Fonctionnalités du Notebook

### Chargement Automatique des Résultats

Le notebook charge automatiquement tous les fichiers JSON de `results/` :

```python
# Cette cellule charge tous les résultats
results_dir = Path('results')
all_results = []
# ...
```

### Graphiques Automatiques

Le notebook génère automatiquement :
- Graphiques de reward moyen par algorithme
- Graphiques de taux de succès
- Tableaux comparatifs

### Analyse par Environnement

Sections dédiées pour :
- LineWorldSimple
- GridWorldSimple
- Two Round Rock Paper Scissors
- Monty Hall Level 1 & 2

---

## 📄 Exporter en PDF

### Méthode 1 : Via Jupyter (Recommandé)

```bash
# Installer nbconvert si nécessaire
pip install nbconvert

# Convertir en HTML puis en PDF
jupyter nbconvert --to pdf rapport_projet_rl.ipynb
```

### Méthode 2 : Via LaTeX (Meilleure qualité)

```bash
# Installer pandoc et LaTeX (MiKTeX sur Windows)
# Puis :
jupyter nbconvert --to pdf --template classic rapport_projet_rl.ipynb
```

### Méthode 3 : Via HTML puis Impression

```bash
# Convertir en HTML
jupyter nbconvert --to html rapport_projet_rl.ipynb

# Ouvrir le fichier HTML dans un navigateur
# Imprimer → Sauvegarder en PDF
```

---

## 🔧 Personnalisation

### Ajouter une Section

1. Cliquer sur une cellule
2. `Insert` → `Insert Cell Below`
3. Changer le type en `Markdown` (dans la barre d'outils)
4. Écrire votre texte

### Modifier les Graphiques

Dans les cellules Python, vous pouvez :
- Modifier les couleurs : `color='blue'`
- Changer la taille : `figsize=(12, 8)`
- Ajouter des légendes, titres, etc.

### Ajouter des Résultats Manuels

Si vous avez des résultats spécifiques à ajouter :

```python
# Ajouter manuellement
manual_result = {
    'algorithm': 'Q-Learning',
    'environment': 'LineWorldSimple',
    'evaluation': {
        'mean_reward': -5.2,
        'success_rate': 0.85
    }
}
all_results.append(manual_result)
```

---

## ✅ Checklist Avant Export

- [ ] Tous les tests exécutés (`test_all_algos_envs.py --all`)
- [ ] Toutes les cellules exécutées sans erreur
- [ ] Sections "[À compléter]" remplies
- [ ] Graphiques affichés correctement
- [ ] Résultats cohérents
- [ ] Date mise à jour
- [ ] Export PDF réussi

---

## 🐛 Problèmes Courants

### Erreur : "ModuleNotFoundError: No module named 'pandas'"

**Solution :**
```bash
pip install pandas matplotlib numpy seaborn
```

### Erreur : "Aucun résultat trouvé"

**Solution :** Exécutez d'abord les tests :
```bash
python test_all_algos_envs.py --all
```

### Les graphiques ne s'affichent pas

**Solution :** Ajoutez au début du notebook :
```python
%matplotlib inline
```

### Export PDF ne fonctionne pas

**Solution :** Utilisez l'export HTML puis imprimez en PDF :
```bash
jupyter nbconvert --to html rapport_projet_rl.ipynb
```

---

## 📚 Ressources

- [Documentation Jupyter](https://jupyter-notebook.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

---

**Le notebook est prêt ! Ouvrez-le avec `jupyter notebook rapport_projet_rl.ipynb`** 🚀

