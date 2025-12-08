"""
Script pour comparer les algorithmes et analyser les résultats
Génère des rapports détaillés et des comparaisons
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_results(results_dir='results'):
    """Charge tous les résultats depuis le dossier results"""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"❌ Le dossier {results_dir} n'existe pas")
        return []
    
    all_results = []
    for file in results_dir.glob('*.json'):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
        except:
            pass
    
    return all_results

def analyze_results(results):
    """Analyse les résultats et génère des statistiques"""
    if not results:
        print("Aucun résultat trouvé")
        return
    
    print("\n" + "="*80)
    print("ANALYSE COMPLÈTE DES RÉSULTATS")
    print("="*80)
    
    # Grouper par algorithme
    by_algorithm = defaultdict(list)
    for result in results:
        algo_name = result.get('algorithm', 'Unknown')
        by_algorithm[algo_name].append(result)
    
    # Statistiques par algorithme
    print("\n📊 STATISTIQUES PAR ALGORITHME")
    print("-"*80)
    
    algo_stats = []
    for algo_name, algo_results in by_algorithm.items():
        rewards = [r['evaluation']['mean_reward'] for r in algo_results if 'evaluation' in r]
        success_rates = [r['evaluation']['success_rate']*100 for r in algo_results if 'evaluation' in r]
        times = [r['training_time'] for r in algo_results if 'training_time' in r]
        
        if rewards:
            algo_stats.append({
                'name': algo_name,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'best_reward': np.max(rewards),
                'mean_success': np.mean(success_rates),
                'mean_time': np.mean(times),
                'num_tests': len(rewards)
            })
    
    # Trier par performance
    algo_stats.sort(key=lambda x: x['mean_reward'], reverse=True)
    
    print(f"{'Algorithme':<30} | {'Reward Moy':>10} | {'Success %':>10} | {'Temps (s)':>10} | {'Tests':>6}")
    print("-"*80)
    
    for stat in algo_stats:
        print(f"{stat['name']:<30} | "
              f"{stat['mean_reward']:>10.2f} | "
              f"{stat['mean_success']:>9.1f}% | "
              f"{stat['mean_time']:>10.2f} | "
              f"{stat['num_tests']:>6}")
    
    # Meilleur algorithme
    if algo_stats:
        best = algo_stats[0]
        print(f"\n🏆 MEILLEUR ALGORITHME: {best['name']}")
        print(f"   Récompense moyenne: {best['mean_reward']:.2f} ± {best['std_reward']:.2f}")
        print(f"   Taux de succès: {best['mean_success']:.1f}%")
        print(f"   Temps moyen: {best['mean_time']:.2f}s")
        print(f"   Nombre de tests: {best['num_tests']}")
    
    # Grouper par environnement
    print("\n\n🌍 STATISTIQUES PAR ENVIRONNEMENT")
    print("-"*80)
    
    by_environment = defaultdict(list)
    for result in results:
        env_name = result.get('environment', 'Unknown')
        by_environment[env_name].append(result)
    
    for env_name, env_results in by_environment.items():
        print(f"\n{env_name}:")
        rewards = [r['evaluation']['mean_reward'] for r in env_results if 'evaluation' in r]
        if rewards:
            print(f"  Récompense moyenne: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"  Meilleure: {np.max(rewards):.2f}, Pire: {np.min(rewards):.2f}")
            print(f"  Nombre de tests: {len(rewards)}")
    
    # Analyse des hyperparamètres
    print("\n\n⚙️ ANALYSE DES HYPERPARAMÈTRES")
    print("-"*80)
    
    # Grouper par type d'algorithme
    qlearning_results = [r for r in results if 'Q-Learning' in r.get('algorithm', '')]
    if qlearning_results:
        print("\nQ-Learning - Impact de alpha:")
        alpha_groups = defaultdict(list)
        for r in qlearning_results:
            alpha = r.get('hyperparameters', {}).get('alpha', '?')
            if 'evaluation' in r:
                alpha_groups[alpha].append(r['evaluation']['mean_reward'])
        
        for alpha in sorted(alpha_groups.keys()):
            rewards = alpha_groups[alpha]
            print(f"  alpha={alpha}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f} (n={len(rewards)})")
    
    # Générer un rapport JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(results),
        'algorithms': algo_stats,
        'best_algorithm': algo_stats[0] if algo_stats else None
    }
    
    report_file = Path('results') / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path('results').mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Rapport détaillé sauvegardé: {report_file}")
    
    return report

def compare_hyperparameters(results):
    """Compare l'impact des hyperparamètres"""
    print("\n" + "="*80)
    print("COMPARAISON DES HYPERPARAMÈTRES")
    print("="*80)
    
    # Q-Learning: alpha
    ql_results = [r for r in results if 'Q-Learning' in r.get('algorithm', '')]
    if ql_results:
        print("\n📈 Q-Learning - Impact de alpha:")
        alpha_perf = defaultdict(list)
        for r in ql_results:
            alpha = r.get('hyperparameters', {}).get('alpha')
            if alpha and 'evaluation' in r:
                alpha_perf[alpha].append(r['evaluation']['mean_reward'])
        
        for alpha in sorted(alpha_perf.keys()):
            rewards = alpha_perf[alpha]
            print(f"  α={alpha:4.2f}: {np.mean(rewards):7.2f} ± {np.std(rewards):5.2f} (n={len(rewards)})")
    
    # Dyna-Q: n_planning_steps
    dyna_results = [r for r in results if 'Dyna-Q' in r.get('algorithm', '')]
    if dyna_results:
        print("\n📈 Dyna-Q - Impact de n_planning_steps:")
        n_perf = defaultdict(list)
        for r in dyna_results:
            n = r.get('hyperparameters', {}).get('n_planning_steps')
            if n and 'evaluation' in r:
                n_perf[n].append(r['evaluation']['mean_reward'])
        
        for n in sorted(n_perf.keys()):
            rewards = n_perf[n]
            print(f"  n={n:2d}: {np.mean(rewards):7.2f} ± {np.std(rewards):5.2f} (n={len(rewards)})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyser et comparer les résultats')
    parser.add_argument('--results-dir', type=str, default='results', help='Dossier contenant les résultats')
    
    args = parser.parse_args()
    
    # Charger les résultats
    results = load_results(args.results_dir)
    
    if results:
        print(f"✅ {len(results)} résultats chargés")
        analyze_results(results)
        compare_hyperparameters(results)
    else:
        print("❌ Aucun résultat trouvé. Exécutez d'abord des tests:")
        print("   python test_with_visualization.py --compare")
        print("   python test_all_algos_envs.py --all")

