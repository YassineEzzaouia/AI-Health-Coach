"""
NEW ARCHITECTURE: RF -> Rule-Based Actions -> DQN Training Pipeline

Flow:
1. Random Forest extracts conditions
2. Load pre-generated rule-based health actions
3. DQN learns Q-values using rule-based actions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Import our modules
from rf_extractor.random_forest_extractor import RandomForestExtractor, create_target_variable
from dqn_agent.health_dqn_agent import HealthDQNAgent, ReplayBuffer
from mlflow_tracker.mlflow_tracker import MLflowTracker
from health_environment import HealthEnvironment  # Realistic health simulation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def reward_multi_component(current_state: pd.Series, next_state: pd.Series) -> float:
    """Multi-component reward function"""
    # Handle NaN values - default to 50 if missing
    curr_overall = current_state.get('Overall_Health_Score', 50) or 50
    next_overall = next_state.get('Overall_Health_Score', 50) or 50
    
    curr_activity = current_state.get('Activity_Score', 50) or 50
    next_activity = next_state.get('Activity_Score', 50) or 50
    
    curr_sleep = current_state.get('Sleep_Score', 50) or 50
    next_sleep = next_state.get('Sleep_Score', 50) or 50
    
    # Replace NaN with 0
    if pd.isna(curr_overall): curr_overall = 50
    if pd.isna(next_overall): next_overall = 50
    if pd.isna(curr_activity): curr_activity = 50
    if pd.isna(next_activity): next_activity = 50
    if pd.isna(curr_sleep): curr_sleep = 50
    if pd.isna(next_sleep): next_sleep = 50
    
    # Component improvements
    overall_delta = (next_overall - curr_overall) * 10.0
    activity_delta = (next_activity - curr_activity) * 5.0
    sleep_delta = (next_sleep - curr_sleep) * 5.0
    
    # Goal bonuses (normalized values 0-1)
    steps = next_state.get('TotalSteps', 0) or 0
    active_min = next_state.get('TotalActiveMinutes', 0) or 0
    if pd.isna(steps): steps = 0
    if pd.isna(active_min): active_min = 0
    
    steps_bonus = 50 if steps >= 0.5 else 0
    sleep_bonus = 30 if next_sleep >= 85 else 0
    active_bonus = 40 if active_min >= 0.3 else 0
    
    # Penalties
    sedentary = next_state.get('SedentaryMinutes', 0) or 0
    if pd.isna(sedentary): sedentary = 0
    sedentary_penalty = -30 if sedentary > 0.8 else 0
    under_sleep_penalty = -40 if next_sleep < 60 else 0
    
    reward = (
        overall_delta + activity_delta + sleep_delta +
        steps_bonus + sleep_bonus + active_bonus +
        sedentary_penalty + under_sleep_penalty
    )
    
    return float(reward)


def plot_dqn_performance(episode_rewards: List[float], episode_health_deltas: List[float], 
                         action_counts: Dict[str, int], output_dir: str = "../plots") -> Dict[str, str]:
    """Generate comprehensive DQN performance visualizations"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plots = {}
    
    # Plot 1: Episode Rewards Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    episodes = range(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
    
    # Add moving average
    if len(episode_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episode_rewards) + 1), moving_avg, 'r-', linewidth=2, label=f'{window}-Episode MA')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('DQN Training: Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = Path(output_dir) / 'episode_rewards.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots['episode_rewards'] = str(plot_path)
    
    # Plot 2: Health Score Improvements
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, episode_health_deltas, 'g-', alpha=0.6, label='Health Improvement')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    
    if len(episode_health_deltas) >= 10:
        window = 10
        moving_avg = np.convolve(episode_health_deltas, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episode_health_deltas) + 1), moving_avg, 'orange', linewidth=2, label=f'{window}-Episode MA')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Health Score Change', fontsize=12)
    ax.set_title('DQN Training: Health Score Improvements per Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = Path(output_dir) / 'health_improvements.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots['health_improvements'] = str(plot_path)
    
    # Plot 3: Action Distribution
    if action_counts:
        fig, ax = plt.subplots(figsize=(14, 8))
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        # Truncate long action names for display
        display_actions = [a[:40] + '...' if len(a) > 40 else a for a in actions]
        
        bars = ax.barh(display_actions, counts, color='steelblue', alpha=0.7)
        ax.set_xlabel('Selection Count', fontsize=12)
        ax.set_ylabel('Action', fontsize=12)
        ax.set_title('DQN Action Selection Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=9)
        
        plot_path = Path(output_dir) / 'action_distribution.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plots['action_distribution'] = str(plot_path)
    
    # Plot 4: Training Progress Summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rewards
    ax1.plot(episodes, episode_rewards, 'b-', alpha=0.6)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)
    
    # Health improvements
    ax2.plot(episodes, episode_health_deltas, 'g-', alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Health Score Change')
    ax2.set_title('Health Improvements')
    ax2.grid(True, alpha=0.3)
    
    # Cumulative rewards
    cumulative_rewards = np.cumsum(episode_rewards)
    ax3.plot(episodes, cumulative_rewards, 'purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Rewards')
    ax3.grid(True, alpha=0.3)
    
    # Statistics table
    ax4.axis('off')
    stats_text = f"""Training Statistics:
    
    Total Episodes: {len(episode_rewards)}
    
    Rewards:
      Mean: {np.mean(episode_rewards):.2f}
      Std: {np.std(episode_rewards):.2f}
      Max: {np.max(episode_rewards):.2f}
      Min: {np.min(episode_rewards):.2f}
    
    Health Improvements:
      Mean: {np.mean(episode_health_deltas):.2f}
      Std: {np.std(episode_health_deltas):.2f}
      Max: {np.max(episode_health_deltas):.2f}
      Min: {np.min(episode_health_deltas):.2f}
    
    Actions Used: {len(action_counts)}
    Total Actions Taken: {sum(action_counts.values())}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', va='center')
    
    plt.suptitle('DQN Training Summary Dashboard', fontsize=16, fontweight='bold')
    plot_path = Path(output_dir) / 'training_summary.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots['training_summary'] = str(plot_path)
    
    return plots


def plot_state_analysis(train_data: pd.DataFrame, output_dir: str = "../plots") -> Dict[str, str]:
    """Analyze and visualize health state distributions"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    plots = {}
    
    # Key health metrics to visualize
    health_metrics = ['Overall_Health_Score', 'Activity_Score', 'Sleep_Score', 
                      'TotalSteps', 'TotalActiveMinutes', 'Calories']
    available_metrics = [m for m in health_metrics if m in train_data.columns]
    
    if not available_metrics:
        return plots
    
    # Plot distributions
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, metric in enumerate(available_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        data = train_data[metric].dropna()
        ax.hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        ax.set_xlabel(metric.replace('_', ' '), fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{metric.replace("_", " ")} Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(available_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Health State Distributions', fontsize=16, fontweight='bold')
    plot_path = Path(output_dir) / 'state_distributions.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plots['state_distributions'] = str(plot_path)
    
    return plots


def train_random_forest(
    data: pd.DataFrame,
    hyperparams: Dict,
    tracker: MLflowTracker
) -> Tuple[RandomForestExtractor, Dict, pd.DataFrame]:
    """Train Random Forest and extract conditions"""
    print("\n" + "="*60)
    print("STAGE 1: RANDOM FOREST TRAINING")
    print("="*60)
    
    # Create target variable
    y, data_valid = create_target_variable(data, method='classification')
    
    # Select features - match actual CSV columns
    feature_cols = [
        'TotalSteps', 'TotalActiveMinutes', 'SedentaryMinutes', 'Calories',
        'Activity_Score', 'Sleep_Score', 'Nutrition_Score', 'Overall_Health_Score',
        'DayOfWeekNum', 'IsWeekend'
    ]
    
    X = data_valid[feature_cols]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test, {len(feature_cols)} features")
    
    # Train RF
    rf_extractor = RandomForestExtractor(
        n_estimators=hyperparams['n_estimators'],
        max_depth=hyperparams['max_depth'],
        min_samples_split=hyperparams['min_samples_split'],
        min_samples_leaf=hyperparams['min_samples_leaf']
    )
    
    rf_extractor.train(X_train, y_train)
    
    # Evaluate
    train_score = rf_extractor.model.score(X_train, y_train)
    test_score = rf_extractor.model.score(X_test, y_test)
    n_conditions = len(rf_extractor.conditions)
    
    print(f"[OK] RF trained: {train_score:.4f} train, {test_score:.4f} test, {n_conditions} conditions")
    
    # Log to MLflow
    tracker.log_rf_experiment(
        rf_model=rf_extractor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_conditions=n_conditions,
        hyperparams=hyperparams
    )
    
    metrics = {
        'rf_train_score': train_score,
        'rf_test_score': test_score,
        'n_conditions': n_conditions
    }
    
    return rf_extractor, metrics, data_valid[feature_cols]


def load_pregenerated_actions(
    max_conditions: int = 20
) -> Dict[int, Dict]:
    """
    Load pre-generated rule-based actions from JSON file
    
    Args:
        max_conditions: Maximum number of conditions to process
    
    Returns:
        Dictionary mapping condition_idx -> {'action': str, 'state': str}
    """
    print("\n" + "="*60)
    print("STAGE 2: LOADING PRE-GENERATED RULE-BASED ACTIONS")
    print("="*60)
    
    # Load pre-generated actions from JSON file - use absolute path from script location
    script_dir = Path(__file__).parent
    actions_file = script_dir / "../data/rulebased_condition_actions.json"
    actions_file = actions_file.resolve()  # Convert to absolute path
    
    if not actions_file.exists():
        print(f"[ERROR] Pre-generated actions file not found: {actions_file}")
        print("Please run regenerate_diverse_actions.py first!")
        raise FileNotFoundError(f"Actions file not found: {actions_file}")
    
    print(f"Loading actions from: {actions_file}")
    with open(actions_file, 'r') as f:
        loaded_actions = json.load(f)
    
    print(f"[OK] Loaded {len(loaded_actions)} pre-generated actions")
    
    # Convert to expected format
    condition_actions = {}
    for cond_idx_str, action_data in loaded_actions.items():
        cond_idx = int(cond_idx_str)
        condition_actions[cond_idx] = {
            'condition_text': action_data.get('condition', ''),
            'action': action_data.get('action', ''),
            'state': action_data.get('state', ''),
            'raw_response': action_data.get('raw_response', 'pre-generated')
        }
    
    # Print first 3 examples
    print("\nSample actions:")
    for i, (cond_idx, data) in enumerate(list(condition_actions.items())[:3], 1):
        print(f"\n{i}. Condition {cond_idx}:")
        print(f"   ACTION: {data['action'][:80]}...")
        print(f"   STATE: {data['state'][:70]}...")
    
    return condition_actions


def train_dqn_with_rule_actions(
    rf_extractor: RandomForestExtractor,
    train_data: pd.DataFrame,
    condition_actions: Dict,
    dqn_hyperparams: Dict,
    tracker: MLflowTracker,
    n_episodes: int = 50
) -> Tuple[HealthDQNAgent, Dict]:
    """
    Train DQN agent using rule-based actions
    
    Args:
        rf_extractor: Trained RF extractor
        train_data: Training data
        condition_actions: Rule-based actions for conditions
        dqn_hyperparams: DQN hyperparameters
        tracker: MLflow tracker
        n_episodes: Number of training episodes
    
    Returns:
        Trained DQN agent and metrics
    """
    print("\n" + "="*60)
    print("STAGE 3: DQN TRAINING WITH RULE-BASED ACTIONS")
    print("="*60)
    
    # Create DQN agent with rule-based actions
    state_dim = len(rf_extractor.conditions)
    
    agent = HealthDQNAgent(
        state_dim=state_dim,
        use_t5_actions=True,  # Dynamic action space (parameter name kept for compatibility)
        **dqn_hyperparams
    )
    
    print(f"\nAgent configuration:")
    print(f"  - State dimension: {state_dim} (RF conditions)")
    print(f"  - Action space: Dynamic (Rule-based)")
    print(f"  - Device: {agent.device}")
    
    # Pre-populate actions from rule-based generation
    print("\nPopulating action space from rule-based recommendations...")
    for cond_idx, cond_data in condition_actions.items():
        action_idx = agent.add_t5_action(cond_data['action'])  # Method name kept for compatibility
        if action_idx < 10:
            print(f"  Action {action_idx}: {cond_data['action'][:60]}...")
    
    print(f"[OK] Initial action space: {agent.action_dim} actions")
    
    # Training loop with realistic environment
    print(f"\nTraining for {n_episodes} episodes...")
    episode_rewards = []
    episode_health_improvements = []
    action_counts = {}
    
    for episode in range(n_episodes):
        # Sample random start state
        state_idx = np.random.randint(0, len(train_data))
        initial_state_data = train_data.iloc[state_idx]
        
        # Create realistic health environment
        env = HealthEnvironment(initial_state_data, action_description="")
        initial_health = initial_state_data['Overall_Health_Score']
        
        # Get initial state features
        state = rf_extractor.conditions_to_features(initial_state_data)
        
        episode_reward = 0
        episode_losses = []
        steps = 0
        done = False
        
        # Run episode with realistic environment
        while not done and steps < 30:
            # Select action (epsilon-greedy)
            action = agent.select_action(state)
            action_text = agent.idx_to_action.get(action, f"action_{action}")
            
            # Track action usage
            action_counts[action_text] = action_counts.get(action_text, 0) + 1
            
            # Take action in realistic environment
            new_state_dict, reward, done = env.step(action_text)
            
            # Convert state dict to pandas Series for RF features
            next_state_series = pd.Series(new_state_dict)
            next_state = rf_extractor.conditions_to_features(next_state_series)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
                tracker.log_dqn_training_step(
                    episode=episode,
                    step=agent.training_step,
                    loss=loss,
                    reward=reward,
                    epsilon=agent.get_epsilon()
                )
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        # Episode summary
        final_health = env.get_state()['Overall_Health_Score']
        health_improvement = final_health - initial_health
        
        episode_rewards.append(episode_reward)
        episode_health_improvements.append(health_improvement)
        agent.rewards.append(episode_reward)
        
        # Log episode
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        tracker.log_dqn_episode(
            episode=episode,
            total_reward=episode_reward,
            steps=steps,
            avg_loss=avg_loss,
            health_improvement=health_improvement
        )
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_health = np.mean(episode_health_improvements[-10:])
            print(f"Episode {episode+1}/{n_episodes} - "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Avg Health Δ: {avg_health:.2f}, "
                  f"Actions: {agent.action_dim}, "
                  f"ε: {agent.get_epsilon():.3f}")
    
    print(f"\n[OK] Training complete!")
    print(f"  Final avg reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Final action space size: {agent.action_dim}")
    print(f"  Unique actions used: {len(action_counts)}")
    
    # Generate and log visualizations
    print("\nGenerating performance visualizations...")
    plots_dir = Path("../plots")
    plots = plot_dqn_performance(episode_rewards, episode_health_improvements, action_counts, str(plots_dir))
    
    # Log plots to MLflow
    for plot_name, plot_path in plots.items():
        mlflow.log_artifact(plot_path, "performance_plots")
        print(f"  [OK] Logged {plot_name}")
    
    # Log episode metrics to MLflow
    for i, (reward, health_delta) in enumerate(zip(episode_rewards, episode_health_improvements)):
        mlflow.log_metric("episode_reward", reward, step=i)
        mlflow.log_metric("episode_health_delta", health_delta, step=i)
    
    # Create models directory and log model
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True, parents=True)
    tracker.log_dqn_model(agent, model_path="../models/dqn_rulebased_model")
    
    metrics = {
        'dqn_final_avg_reward': np.mean(episode_rewards[-10:]),
        'dqn_final_health_improvement': np.mean(episode_health_improvements[-10:]),
        'dqn_total_steps': agent.training_step,
        'dqn_final_action_count': agent.action_dim,
        'unique_actions_used': len(action_counts)
    }
    
    return agent, metrics


def test_full_pipeline(
    rf_extractor: RandomForestExtractor,
    dqn_agent: HealthDQNAgent,
    test_data: pd.DataFrame,
    tracker: MLflowTracker,
    n_samples: int = 10
):
    """Test the full RF -> DQN pipeline with rule-based actions"""
    print("\n" + "="*60)
    print("STAGE 4: FULL PIPELINE TEST")
    print("="*60)
    
    for i in range(min(n_samples, len(test_data))):
        user_state = test_data.iloc[i]
        
        # Get RF conditions
        condition_features = rf_extractor.conditions_to_features(user_state)
        condition_text = rf_extractor.conditions_to_text(user_state, top_k=3)
        
        # DQN selects action
        action_idx = dqn_agent.select_action(condition_features, epsilon=0.0)
        action_text = dqn_agent.get_action_name(action_idx)
        
        if i < 3:
            print(f"\n{'='*60}")
            print(f"TEST EXAMPLE {i+1}")
            print(f"{'='*60}")
            print(f"Conditions:\n{condition_text}")
            print(f"\nRecommended Action: {action_text}")


def run_full_pipeline_rulebased(
    data_path: str,
    rf_hyperparams: Dict,
    dqn_hyperparams: Dict,
    n_episodes: int = 50,
    max_conditions: int = 20
):
    """Run complete RF -> Rule-Based Actions -> DQN pipeline"""
    print("\n" + "="*80)
    print("RF -> RULE-BASED ACTIONS -> DQN HEALTH COACH PIPELINE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize MLflow
    tracker = MLflowTracker(experiment_name="Health_DQN_RuleBased_Actions")
    
    run_name = f"RF_n{rf_hyperparams['n_estimators']}_d{rf_hyperparams['max_depth']}_RuleBased_DQN"
    tracker.start_run(
        run_name=run_name,
        tags={'pipeline': 'rulebased_actions', 'version': '2.0', 'timestamp': datetime.now().isoformat()}
    )
    
    try:
        # Load data
        print(f"\nLoading data: {data_path}")
        data = pd.read_csv(data_path)
        print(f"[OK] Loaded {len(data)} records")
        
        # STAGE 1: Train Random Forest
        rf_extractor, rf_metrics, train_data = train_random_forest(
            data=data,
            hyperparams=rf_hyperparams,
            tracker=tracker
        )
        
        # Plot state distributions
        print("\nGenerating state distribution plots...")
        state_plots = plot_state_analysis(train_data, output_dir="../plots")
        for plot_name, plot_path in state_plots.items():
            mlflow.log_artifact(plot_path, "state_analysis")
            print(f"  [OK] Logged {plot_name}")
        
        # STAGE 2: Load pre-generated rule-based actions
        print("\nLoading pre-generated rule-based actions...")
        condition_actions = load_pregenerated_actions(
            max_conditions=max_conditions
        )
        
        # STAGE 3: Train DQN with rule-based actions
        dqn_agent, dqn_metrics = train_dqn_with_rule_actions(
            rf_extractor=rf_extractor,
            train_data=train_data,
            condition_actions=condition_actions,
            dqn_hyperparams=dqn_hyperparams,
            tracker=tracker,
            n_episodes=n_episodes
        )
        
        # STAGE 4: Test full pipeline
        test_full_pipeline(
            rf_extractor=rf_extractor,
            dqn_agent=dqn_agent,
            test_data=train_data[-20:],
            tracker=tracker,
            n_samples=10
        )
        
        # Log combined metrics
        combined_metrics = {**rf_metrics, **dqn_metrics}
        mlflow.log_metrics(combined_metrics)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nFinal Metrics:")
        for key, value in combined_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n[OK] Results logged to MLflow")
        print(f"  Run: mlflow ui")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        tracker.end_run()


if __name__ == "__main__":
    # Configuration - use absolute path from script location
    from pathlib import Path
    SCRIPT_DIR = Path(__file__).parent
    DATA_PATH = SCRIPT_DIR / "../../Processed_Data/rl_training_data_normalized.csv"
    DATA_PATH = str(DATA_PATH.resolve())
    
    RF_HYPERPARAMS = {
        'n_estimators': 50,  # Fewer trees for speed
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10
    }
    
    DQN_HYPERPARAMS = {
        'learning_rate': 0.0005,  # Lower learning rate for stability
        'gamma': 0.95,  # Higher discount factor for long-term rewards
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,  # Higher minimum exploration
        'epsilon_decay': 3000,  # Slower decay
        'buffer_size': 10000,  # Larger buffer
        'batch_size': 64,  # Larger batches for stability
        'target_update_freq': 100  # More frequent target updates
    }
    
    # Run pipeline
    run_full_pipeline_rulebased(
        data_path=DATA_PATH,
        rf_hyperparams=RF_HYPERPARAMS,
        dqn_hyperparams=DQN_HYPERPARAMS,
        n_episodes=100,  # Increased to 100 episodes for better learning
        max_conditions=20  # Generate rule-based actions for top 20 conditions
    )
    
    print("\n" + "="*80)
    print("NEW ARCHITECTURE COMPLETE!")
    print("RF conditions -> Rule-based actions -> DQN learns Q-values")
    print("\nTo view results:")
    print("  cd health_dqn_rf_t5")
    print("  mlflow ui")
    print("="*80)
