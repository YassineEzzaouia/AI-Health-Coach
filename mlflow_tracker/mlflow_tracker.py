"""
MLflow Experiment Tracker
Tracks all experiments: RF hyperparameters, DQN training, T5 generation
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
# import seaborn as sns  # Commented out to avoid import conflicts
from pathlib import Path


class MLflowTracker:
    """
    Centralized MLflow experiment tracking for RF-DQN-T5 pipeline
    """
    
    def __init__(
        self,
        experiment_name: str = "Health_DQN_RF_T5",
        tracking_uri: str = None
    ):
        """
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: MLflow tracking server URI (None = local)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Local tracking (./mlruns/)
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        self.experiment = mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
        print(f"[OK] MLflow experiment: {experiment_name}")
        print(f"  - Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"  - Experiment ID: {self.experiment.experiment_id}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name of the run
            tags: Dictionary of tags (e.g., {'stage': 'rf_training'})
        """
        mlflow.start_run(run_name=run_name)
        
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        print(f"[OK] Started MLflow run: {run_name or 'Unnamed'}")
        return mlflow.active_run()
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
    
    def log_rf_experiment(
        self,
        rf_model,
        X_train,
        y_train,
        X_test,
        y_test,
        n_conditions: int,
        hyperparams: Dict[str, Any]
    ):
        """
        Log Random Forest experiment
        
        Args:
            rf_model: Trained RandomForestExtractor
            X_train, y_train: Training data
            X_test, y_test: Test data
            n_conditions: Number of extracted conditions
            hyperparams: RF hyperparameters
        """
        # Log hyperparameters
        mlflow.log_params(hyperparams)
        
        # Log model performance
        train_score = rf_model.model.score(X_train, y_train)
        test_score = rf_model.model.score(X_test, y_test)
        
        mlflow.log_metrics({
            'rf_train_score': train_score,
            'rf_test_score': test_score,
            'n_conditions': n_conditions,
            'n_features': X_train.shape[1]
        })
        
        # Log feature importance
        importance_df = rf_model.get_feature_importance()
        importance_fig = self._plot_feature_importance(importance_df)
        mlflow.log_figure(importance_fig, "rf_feature_importance.png")
        plt.close()
        
        # Log model
        mlflow.sklearn.log_model(rf_model.model, "random_forest_model")
        
        # Log conditions as artifact
        conditions_path = "rf_conditions.json"
        rf_model.save_conditions(conditions_path)
        mlflow.log_artifact(conditions_path)
        
        print(f"[OK] Logged RF experiment")
        print(f"  - Train score: {train_score:.4f}")
        print(f"  - Test score: {test_score:.4f}")
        print(f"  - Conditions: {n_conditions}")
    
    def log_dqn_training_step(
        self,
        episode: int,
        step: int,
        loss: float,
        reward: float,
        epsilon: float,
        avg_reward: float = None
    ):
        """
        Log DQN training metrics for one step
        
        Args:
            episode: Current episode number
            step: Current step number
            loss: Training loss
            reward: Reward received
            epsilon: Current exploration rate
            avg_reward: Average reward over last N episodes
        """
        metrics = {
            'dqn_loss': loss,
            'dqn_reward': reward,
            'dqn_epsilon': epsilon
        }
        
        if avg_reward is not None:
            metrics['dqn_avg_reward'] = avg_reward
        
        mlflow.log_metrics(metrics, step=step)
    
    def log_dqn_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        avg_loss: float,
        health_improvement: float
    ):
        """
        Log DQN episode summary
        
        Args:
            episode: Episode number
            total_reward: Total reward in episode
            steps: Number of steps in episode
            avg_loss: Average loss in episode
            health_improvement: Health score improvement
        """
        mlflow.log_metrics({
            'episode_reward': total_reward,
            'episode_steps': steps,
            'episode_avg_loss': avg_loss,
            'episode_health_improvement': health_improvement
        }, step=episode)
    
    def log_dqn_model(self, dqn_agent, model_path: str = "dqn_model"):
        """
        Log trained DQN model
        
        Args:
            dqn_agent: Trained HealthDQNAgent
            model_path: Path to save model
        """
        # Save model
        dqn_agent.save(f"{model_path}.pth")
        
        # Log as artifact
        mlflow.log_artifact(f"{model_path}.pth")
        
        # Log training curves
        self._plot_dqn_curves(dqn_agent)
        
        print(f"[OK] Logged DQN model")
    
    def log_t5_generation(
        self,
        conditions: List[str],
        actions: List[str],
        recommendations: List[str],
        sample_size: int = 10
    ):
        """
        Log T5 generation examples
        
        Args:
            conditions: List of input conditions
            actions: List of recommended actions
            recommendations: List of generated recommendations
            sample_size: Number of samples to log
        """
        # Log sample recommendations as text artifact
        samples = []
        for i in range(min(sample_size, len(conditions))):
            samples.append(f"Example {i+1}:")
            samples.append(f"Conditions: {conditions[i][:200]}...")
            samples.append(f"Action: {actions[i]}")
            samples.append(f"Recommendation: {recommendations[i]}")
            samples.append("-" * 80)
        
        # Save to file and log
        with open("t5_samples.txt", "w") as f:
            f.write("\n".join(samples))
        
        mlflow.log_artifact("t5_samples.txt")
        
        # Log average recommendation length
        avg_length = np.mean([len(rec) for rec in recommendations])
        mlflow.log_metric("t5_avg_recommendation_length", avg_length)
        
        print(f"[OK] Logged T5 generation examples")
    
    def log_full_pipeline_evaluation(
        self,
        test_users: pd.DataFrame,
        rf_extractor,
        dqn_agent,
        t5_coach,
        metrics: Dict[str, float]
    ):
        """
        Log complete pipeline evaluation
        
        Args:
            test_users: Test user data
            rf_extractor: Trained RF extractor
            dqn_agent: Trained DQN agent
            t5_coach: T5 coach
            metrics: Evaluation metrics dictionary
        """
        # Log evaluation metrics
        mlflow.log_metrics(metrics)
        
        # Generate and log example recommendations
        n_samples = min(5, len(test_users))
        for i in range(n_samples):
            user_state = test_users.iloc[i]
            
            # Get conditions
            condition_features = rf_extractor.conditions_to_features(user_state)
            condition_text = rf_extractor.conditions_to_text(user_state, top_k=5)
            
            # Get action
            action = dqn_agent.select_action(condition_features, epsilon=0.0)  # Greedy
            action_name = dqn_agent.get_action_name(action)
            
            # Get recommendation (would need action_descriptions)
            # recommendation = t5_coach.generate_recommendation(condition_text, action_name, "...")
            
            # Log as artifact
            example = f"User {i+1}:\n"
            example += f"State: {user_state.to_dict()}\n"
            example += f"Conditions: {condition_text}\n"
            example += f"Action: {action_name}\n\n"
            
            with open(f"pipeline_example_{i+1}.txt", "w") as f:
                f.write(example)
            
            mlflow.log_artifact(f"pipeline_example_{i+1}.txt")
        
        print(f"[OK] Logged full pipeline evaluation")
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame) -> plt.Figure:
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        importance_df_top = importance_df.head(15)
        
        # Use matplotlib barplot instead of seaborn
        ax.barh(importance_df_top['feature'], importance_df_top['importance'])
        
        ax.set_title('Random Forest Feature Importance (Top 15)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def _plot_dqn_curves(self, dqn_agent):
        """Plot DQN training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        if len(dqn_agent.losses) > 0:
            axes[0].plot(dqn_agent.losses, alpha=0.6)
            axes[0].set_title('DQN Training Loss', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Training Step', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].grid(True, alpha=0.3)
        
        # Reward curve
        if len(dqn_agent.rewards) > 0:
            axes[1].plot(dqn_agent.rewards, alpha=0.6)
            # Moving average
            window = min(100, len(dqn_agent.rewards))
            if window > 1:
                moving_avg = pd.Series(dqn_agent.rewards).rolling(window=window).mean()
                axes[1].plot(moving_avg, color='red', linewidth=2, label=f'{window}-step MA')
                axes[1].legend()
            
            axes[1].set_title('DQN Episode Rewards', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Episode', fontsize=12)
            axes[1].set_ylabel('Total Reward', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        mlflow.log_figure(fig, "dqn_training_curves.png")
        plt.close()
    
    def get_best_run(self, metric: str = "rf_test_score", order: str = "DESC") -> Any:
        """
        Get best run based on metric
        
        Args:
            metric: Metric name
            order: 'DESC' for maximization, 'ASC' for minimization
        
        Returns:
            Best run object
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )
        
        if len(runs) > 0:
            best_run = runs[0]
            print(f"[OK] Best run: {best_run.info.run_id}")
            print(f"  - {metric}: {best_run.data.metrics.get(metric)}")
            return best_run
        else:
            print("No runs found")
            return None


if __name__ == "__main__":
    # Example usage
    print("Initializing MLflow Tracker...")
    
    tracker = MLflowTracker(experiment_name="Health_DQN_RF_T5_Test")
    
    # Example: Log RF experiment
    tracker.start_run(
        run_name="RF_n100_depth5",
        tags={'stage': 'rf_training', 'version': '1.0'}
    )
    
    # Simulate logging
    mlflow.log_params({
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 20
    })
    
    mlflow.log_metrics({
        'rf_train_score': 0.85,
        'rf_test_score': 0.78,
        'n_conditions': 127
    })
    
    tracker.end_run()
    
    print("\n[OK] Example run logged successfully")
    print("\nTo view results, run:")
    print("  mlflow ui")
    print("  Then open: http://localhost:5000")
