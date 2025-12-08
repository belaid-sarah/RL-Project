"""
Exemple rapide pour démarrer avec un algorithme d'apprentissage par renforcement

Ce script montre comment :
1. Créer un environnement
2. Créer un agent
3. Entraîner l'agent
4. Évaluer les performances
5. Utiliser la politique apprise
"""

from envs.lineworld import LineWorld
from algos.q_learning import QLearningAgent

def main():
    print("="*60)
    print("EXEMPLE RAPIDE - Q-Learning sur Line World")
    print("="*60)
    
    # 1. Créer l'environnement
    print("\n1. Création de l'environnement Line World...")
    env = LineWorld(length=10)
    print(f"   ✅ Environnement créé (longueur: {env.length})")
    
    # 2. Créer l'agent
    print("\n2. Création de l'agent Q-Learning...")
    agent = QLearningAgent(
        env,
        alpha=0.1,      # Taux d'apprentissage
        gamma=0.99,     # Facteur d'actualisation
        epsilon=0.1     # Exploration
    )
    print("   ✅ Agent créé")
    
    # 3. Entraîner l'agent
    print("\n3. Entraînement de l'agent...")
    print("   (Cela peut prendre quelques secondes...)")
    agent.train(num_episodes=500, verbose=True)
    print("   ✅ Entraînement terminé")
    
    # 4. Évaluer les performances
    print("\n4. Évaluation des performances...")
    eval_results = agent.evaluate(num_episodes=100, verbose=False)
    print(f"   ✅ Résultats d'évaluation:")
    print(f"      - Récompense moyenne: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"      - Taux de succès: {eval_results['success_rate']*100:.1f}%")
    print(f"      - Nombre moyen de steps: {eval_results['mean_steps']:.1f}")
    
    # 5. Tester la politique apprise
    print("\n5. Test de la politique apprise (5 épisodes)...")
    for i in range(5):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        print(f"\n   Épisode {i+1}:")
        while not done and steps < 50:
            # Utiliser la politique apprise (pas d'exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            steps += 1
            total_reward += reward
            
            # Afficher la position
            if isinstance(state, dict):
                pos = state.get('position', '?')
            else:
                pos = state
            print(f"      Step {steps}: Position={pos}, Action={action}, Reward={reward:.2f}")
            
            if done:
                print(f"      ✅ Épisode terminé! Reward total: {total_reward:.2f}")
                break
    
    # 6. Statistiques d'entraînement
    print("\n6. Statistiques d'entraînement:")
    stats = agent.get_training_stats()
    if stats:
        print(f"   - Épisodes d'entraînement: {stats['total_episodes']}")
        print(f"   - Récompense finale moyenne: {stats['final_mean_reward']:.2f}")
        print(f"   - Meilleure récompense: {stats['best_reward']:.2f}")
        print(f"   - Temps d'entraînement: {stats['training_time']:.2f}s")
        if stats['convergence_episode']:
            print(f"   - Convergence à l'épisode: {stats['convergence_episode']}")
    
    # 7. Sauvegarder l'agent
    print("\n7. Sauvegarde de l'agent...")
    agent.save("models/qlearning_lineworld_example.pkl")
    print("   ✅ Agent sauvegardé dans 'models/qlearning_lineworld_example.pkl'")
    
    print("\n" + "="*60)
    print("✅ Exemple terminé avec succès!")
    print("="*60)
    print("\nPour tester d'autres algorithmes, modifiez ce script ou")
    print("consultez le GUIDE_UTILISATION.md pour plus d'exemples.")

if __name__ == "__main__":
    main()



