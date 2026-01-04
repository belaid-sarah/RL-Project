"""
Script de test simple pour vérifier que l'adaptateur des environnements secrets fonctionne
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from envs.secret_env_adapter import create_secret_env_0, create_secret_env_1, create_secret_env_2, create_secret_env_3
    
    print("Test de l'adaptateur SecretEnv0...")
    env0 = create_secret_env_0()
    print(f"  [OK] SecretEnv0 cree")
    print(f"  - Nombre d'états: {env0.secret_env.num_states()}")
    print(f"  - Nombre d'actions: {env0.secret_env.num_actions()}")
    print(f"  - Nombre de rewards: {env0.secret_env.num_rewards()}")
    
    # Test reset et step
    state = env0.reset()
    print(f"  - État initial: {state}")
    
    available = env0.get_available_actions()
    print(f"  - Actions disponibles: {available}")
    
    if len(available) > 0:
        action = available[0]
        next_state, reward, done, info = env0.step(action)
        print(f"  - Après action {action}: état={next_state}, reward={reward}, done={done}")
        print(f"  [OK] Test step() reussi")
    
    print("\nTest de l'adaptateur SecretEnv1...")
    env1 = create_secret_env_1()
    print(f"  [OK] SecretEnv1 cree")
    print(f"  - Nombre d'états: {env1.secret_env.num_states()}")
    print(f"  - Nombre d'actions: {env1.secret_env.num_actions()}")
    
    print("\nTest de l'adaptateur SecretEnv2...")
    env2 = create_secret_env_2()
    print(f"  [OK] SecretEnv2 cree")
    print(f"  - Nombre d'états: {env2.secret_env.num_states()}")
    print(f"  - Nombre d'actions: {env2.secret_env.num_actions()}")
    
    print("\nTest de l'adaptateur SecretEnv3...")
    env3 = create_secret_env_3()
    print(f"  [OK] SecretEnv3 cree")
    print(f"  - Nombre d'états: {env3.secret_env.num_states()}")
    print(f"  - Nombre d'actions: {env3.secret_env.num_actions()}")
    
    print("\n[OK] Tous les adaptateurs fonctionnent correctement!")
    
except ImportError as e:
    print(f"[ERROR] Erreur d'import: {e}")
    print("Vérifiez que secret_envs_wrapper.py est dans le dossier envs/")
except Exception as e:
    print(f"[ERROR] Erreur: {e}")
    import traceback
    traceback.print_exc()

