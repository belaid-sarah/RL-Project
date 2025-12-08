"""
Version RAPIDE du test - pour tester rapidement sans attendre 20 minutes
"""

from envs.gridworld import GridWorld
from envs.rps import TwoRoundRPS
from envs.monty_hall_level1 import MontyHallLevel1
from envs.monty_hall_level2 import MontyHallLevel2
from algos.q_learning import QLearningAgent
import json
from pathlib import Path
from datetime import datetime

def test_quick():
    """Test rapide - saute LineWorld qui est trop lent"""
    
    environments = {
        'GridWorld': lambda: GridWorld(width=5, height=5),
        'TwoRoundRPS': lambda: TwoRoundRPS(),
        'MontyHallLevel1': lambda: MontyHallLevel1(),
        'MontyHallLevel2': lambda: MontyHallLevel2(),
    }
    
    episodes_by_env = {
        'GridWorld': 500,  # Réduit
        'TwoRoundRPS': 300,
        'MontyHallLevel1': 300,
        'MontyHallLevel2': 300,
    }
    
    max_steps_by_env = {
        'GridWorld': 100,
        'TwoRoundRPS': 10,
        'MontyHallLevel1': 10,
        'MontyHallLevel2': 20,
    }
    
    print(f"\n{'='*70}")
    print(f"TEST RAPIDE: Q-Learning (sans LineWorld)")
    print(f"{'='*70}\n")
    
    results = []
    hyperparams = {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1}
    
    for env_name, env_factory in environments.items():
        print(f"\n{'─'*70}")
        print(f"🌍 {env_name}")
        print(f"{'─'*70}")
        
        try:
            env = env_factory()
            agent = QLearningAgent(env, **hyperparams)
            
            num_eps = episodes_by_env.get(env_name, 300)
            max_steps = max_steps_by_env.get(env_name, 100)
            
            print(f"📚 Entraînement: {num_eps} épisodes...")
            agent.train(num_episodes=num_eps, verbose=False, max_steps_per_episode=max_steps)
            
            print(f"📊 Évaluation...")
            eval_results = agent.evaluate(num_episodes=50, verbose=False)
            
            print(f"✅ Reward: {eval_results['mean_reward']:.2f} | Success: {eval_results['success_rate']*100:.1f}%")
            
            results.append({
                'algorithm': 'Q-Learning',
                'environment': env_name,
                'hyperparameters': hyperparams,
                'num_episodes': num_eps,
                'training_time': agent.training_time,
                'evaluation': eval_results
            })
            
        except Exception as e:
            print(f"❌ Erreur: {str(e)}")
    
    # Résumé
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ")
    print(f"{'='*70}")
    for r in results:
        if 'evaluation' in r:
            print(f"{r['environment']:<20} | Reward: {r['evaluation']['mean_reward']:7.2f} | "
                  f"Success: {r['evaluation']['success_rate']*100:5.1f}% | "
                  f"Time: {r['training_time']:.2f}s")
    
    # Sauvegarder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    filename = results_dir / f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Sauvegardé: {filename}")

if __name__ == "__main__":
    test_quick()

