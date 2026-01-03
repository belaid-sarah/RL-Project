"""
Exemple Visuel : Comment Policy/Value Iteration Construisent le Modèle MDP

Ce script montre étape par étape comment le modèle MDP est construit.
"""

from envs.lineworld_simple import LineWorldSimple

def exemple_construction_modele():
    """Montre comment le modèle est construit pour LineWorldSimple"""
    
    print("="*70)
    print("EXEMPLE : Construction du Modèle MDP pour LineWorldSimple")
    print("="*70)
    
    # Créer un environnement simple
    env = LineWorldSimple(length=5)  # Petit pour l'exemple
    print(f"\nEnvironnement : LineWorldSimple(length={env.length})")
    print(f"États possibles : {list(range(env.length))}")
    print(f"Actions possibles : [0 (gauche), 1 (droite)]")
    
    # Simuler la construction du modèle
    model = {}
    states = list(range(env.length))
    actions = [0, 1]
    
    print(f"\n{'='*70}")
    print("CONSTRUCTION DU MODÈLE : Test de toutes les transitions")
    print("="*70)
    
    for s in states:
        for a in actions:
            # Sauvegarder l'état actuel
            old_state = env.state
            
            # Forcer l'état à s
            env.reset()
            env.state = s
            
            # Exécuter l'action a
            s_next, r, done, info = env.step(a)
            r = round(r, 1)
            
            # Enregistrer dans le modèle
            key = (s, a, s_next, r)
            model[key] = 1.0
            
            # Afficher la transition
            action_name = "gauche" if a == 0 else "droite"
            print(f"  État {s:2d} + Action {a} ({action_name:6s}) → État {s_next:2d}, Reward {r:5.1f}")
            
            # Restaurer l'état
            env.state = old_state
    
    print(f"\n{'='*70}")
    print("MODÈLE CONSTRUIT")
    print("="*70)
    print(f"Nombre de transitions : {len(model)}")
    print(f"\nExemples de transitions :")
    
    # Afficher quelques exemples
    examples = [
        (0, 0, 0, -1.0),   # État 0, gauche → reste à 0
        (0, 1, 1, -1.0),   # État 0, droite → état 1
        (2, 1, 3, -1.0),   # État 2, droite → état 3
        (4, 1, 4, 1.0),    # État 4 (goal), droite → reste, reward +1
    ]
    
    for s, a, s_next, r in examples:
        if (s, a, s_next, r) in model:
            action_name = "gauche" if a == 0 else "droite"
            print(f"  p(s'={s_next}, r={r:5.1f} | s={s}, a={a} ({action_name})) = {model[(s, a, s_next, r)]}")
    
    print(f"\n{'='*70}")
    print("UTILISATION DU MODÈLE")
    print("="*70)
    print("""
Maintenant Policy Iteration peut utiliser ce modèle pour calculer :

V(s) = Σ_s' Σ_r p(s',r|s,π(s)) [r + γV(s')]

Par exemple, pour calculer V(2) avec π(2) = 1 (droite) :

V(2) = p(3, -1.0 | 2, 1) × [-1.0 + γ × V(3)]
     = 1.0 × [-1.0 + 0.99 × V(3)]
     = -1.0 + 0.99 × V(3)

Le modèle permet de connaître toutes les transitions possibles !
""")
    
    return model

if __name__ == "__main__":
    model = exemple_construction_modele()

