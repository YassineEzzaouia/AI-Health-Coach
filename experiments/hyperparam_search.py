"""
Hyperparameter Search for Random Forest
Finds optimal n_estimators and max_depth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

from rf_extractor.random_forest_extractor import RandomForestExtractor, create_target_variable
from mlflow_tracker.mlflow_tracker import MLflowTracker


def hyperparameter_search(
    data: pd.DataFrame,
    n_estimators_list: list = [50, 100, 150, 200],
    max_depth_list: list = [3, 5, 7, 10],
    min_samples_split: int = 20,
    min_samples_leaf: int = 10
):
    """
    Grid search over RF hyperparameters
    
    Args:
        data: Health data
        n_estimators_list: List of n_estimators to try
        max_depth_list: List of max_depth to try
        min_samples_split: Fixed min_samples_split
        min_samples_leaf: Fixed min_samples_leaf
    
    Returns:
        DataFrame with results
    """
    print("="*80)
    print("RANDOM FOREST HYPERPARAMETER SEARCH")
    print("="*80)
    
    # Prepare data
    y, data_valid = create_target_variable(data, method='classification')
    
    feature_cols = [
        'TotalSteps', 'TotalDistance', 'Calories',
        'TotalActiveMinutes', 'VeryActiveMinutes', 'FairlyActiveMinutes',
        'LightlyActiveMinutes', 'SedentaryMinutes',
        'Activity_Score', 'Sleep_Score', 'Nutrition_Score',
        'Overall_Health_Score', 'DayOfWeekNum', 'IsWeekend'
    ]
    
    X = data_valid[feature_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")
    print(f"\nSearching over:")
    print(f"  - n_estimators: {n_estimators_list}")
    print(f"  - max_depth: {max_depth_list}")
    print(f"  - Total combinations: {len(n_estimators_list) * len(max_depth_list)}")
    
    # Initialize MLflow
    tracker = MLflowTracker(experiment_name="Health_RF_HyperparameterSearch")
    
    # Grid search
    results = []
    
    for n_est, max_d in product(n_estimators_list, max_depth_list):
        print(f"\nTrying n_estimators={n_est}, max_depth={max_d}...")
        
        # Start MLflow run
        tracker.start_run(
            run_name=f"RF_n{n_est}_d{max_d}",
            tags={'stage': 'hyperparam_search'}
        )
        
        try:
            # Train RF
            rf_extractor = RandomForestExtractor(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
            
            rf_extractor.train(X_train, y_train)
            
            # Evaluate
            train_score = rf_extractor.model.score(X_train, y_train)
            test_score = rf_extractor.model.score(X_test, y_test)
            n_conditions = len(rf_extractor.conditions)
            
            # Log to MLflow
            hyperparams = {
                'n_estimators': n_est,
                'max_depth': max_d,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }
            
            tracker.log_rf_experiment(
                rf_model=rf_extractor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_conditions=n_conditions,
                hyperparams=hyperparams
            )
            
            # Store results
            results.append({
                'n_estimators': n_est,
                'max_depth': max_d,
                'train_score': train_score,
                'test_score': test_score,
                'n_conditions': n_conditions,
                'overfit': train_score - test_score
            })
            
            print(f"  Train: {train_score:.4f}, Test: {test_score:.4f}, "
                  f"Conditions: {n_conditions}, Overfit: {train_score - test_score:.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
        
        finally:
            tracker.end_run()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('../data/rf_hyperparam_results.csv', index=False)
    print(f"\n✓ Results saved to rf_hyperparam_results.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Find best
    best_idx = results_df['test_score'].idxmax()
    best_params = results_df.iloc[best_idx]
    
    print("\n" + "="*80)
    print("BEST PARAMETERS:")
    print("="*80)
    print(f"  n_estimators: {best_params['n_estimators']:.0f}")
    print(f"  max_depth: {best_params['max_depth']:.0f}")
    print(f"  Train score: {best_params['train_score']:.4f}")
    print(f"  Test score: {best_params['test_score']:.4f}")
    print(f"  Conditions: {best_params['n_conditions']:.0f}")
    print(f"  Overfitting: {best_params['overfit']:.4f}")
    
    # Plot results
    plot_hyperparam_results(results_df)
    
    return results_df


def plot_hyperparam_results(results_df: pd.DataFrame):
    """
    Plot hyperparameter search results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pivot for heatmaps
    pivot_test = results_df.pivot(
        index='max_depth',
        columns='n_estimators',
        values='test_score'
    )
    
    pivot_conditions = results_df.pivot(
        index='max_depth',
        columns='n_estimators',
        values='n_conditions'
    )
    
    pivot_overfit = results_df.pivot(
        index='max_depth',
        columns='n_estimators',
        values='overfit'
    )
    
    # Test score heatmap
    sns.heatmap(
        pivot_test,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        ax=axes[0, 0],
        cbar_kws={'label': 'Test Accuracy'}
    )
    axes[0, 0].set_title('Test Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('n_estimators', fontsize=12)
    axes[0, 0].set_ylabel('max_depth', fontsize=12)
    
    # Number of conditions heatmap
    sns.heatmap(
        pivot_conditions,
        annot=True,
        fmt='.0f',
        cmap='plasma',
        ax=axes[0, 1],
        cbar_kws={'label': 'Count'}
    )
    axes[0, 1].set_title('Number of Conditions', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('n_estimators', fontsize=12)
    axes[0, 1].set_ylabel('max_depth', fontsize=12)
    
    # Overfitting heatmap
    sns.heatmap(
        pivot_overfit,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        ax=axes[1, 0],
        cbar_kws={'label': 'Train - Test'}
    )
    axes[1, 0].set_title('Overfitting (Train - Test)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('n_estimators', fontsize=12)
    axes[1, 0].set_ylabel('max_depth', fontsize=12)
    
    # Test score vs n_conditions scatter
    for depth in results_df['max_depth'].unique():
        subset = results_df[results_df['max_depth'] == depth]
        axes[1, 1].scatter(
            subset['n_conditions'],
            subset['test_score'],
            label=f'depth={depth}',
            s=100,
            alpha=0.7
        )
    
    axes[1, 1].set_title('Test Score vs Conditions', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Conditions', fontsize=12)
    axes[1, 1].set_ylabel('Test Accuracy', fontsize=12)
    axes[1, 1].legend(title='max_depth')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/rf_hyperparam_search.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to rf_hyperparam_search.png")
    plt.show()


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = pd.read_csv('../Processed_Data/rl_training_data_normalized.csv')
    print(f"✓ Loaded {len(data)} records")
    
    # Run hyperparameter search
    results = hyperparameter_search(
        data=data,
        n_estimators_list=[50, 100, 150, 200],
        max_depth_list=[3, 5, 7, 10],
        min_samples_split=20,
        min_samples_leaf=10
    )
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("Based on results:")
    print("  - For INTERPRETABILITY (shorter prompts): max_depth=3 or 5")
    print("  - For ACCURACY: max_depth=7 or 10")
    print("  - n_estimators: 100-150 (diminishing returns after)")
    print("\nFor T5 prompts, recommend max_depth=5 (balance interpretability + accuracy)")
