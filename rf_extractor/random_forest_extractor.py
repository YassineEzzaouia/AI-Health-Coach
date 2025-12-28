"""
Random Forest Condition Extractor
Trains RF on health data and extracts interpretable decision rules
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import _tree
from typing import List, Dict, Tuple
import json


class RandomForestExtractor:
    """
    Extracts decision tree conditions from Random Forest
    Converts conditions to:
    1. Numerical features for DQN
    2. Text descriptions for T5 prompts
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42,
    ):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (controls prompt length)
            min_samples_split: Min samples to split node (prevents overfitting)
            min_samples_leaf: Min samples in leaf (prevents overfitting)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
    
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
      
        self.feature_names = None
        self.conditions = []
        self.condition_features = []
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train Random Forest on health data
        
        Args:
            X: Feature matrix (e.g., TotalSteps, Sleep_Score, etc.)
            y: Target variable (e.g., health improvement class or score)
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        
        print(f"✓ Random Forest trained")
        print(f"  - Trees: {self.n_estimators}")
        print(f"  - Max depth: {self.max_depth}")
        print(f"  - Train score: {self.model.score(X, y):.4f}")
        
        # Extract conditions from all trees
        self._extract_all_conditions()
        
        return self
    
    def _extract_all_conditions(self):
        """
        Extract decision paths from all trees
        Each path = one condition (e.g., "steps < 5000 AND sleep_score < 70")
        """
        self.conditions = []
        
        print(f"\nExtracting conditions from {self.n_estimators} trees...")
        
        for tree_idx, tree in enumerate(self.model.estimators_):
            tree_obj = tree.tree_ if hasattr(tree, 'tree_') else tree
            tree_conditions = self._extract_tree_conditions(tree_obj, tree_idx)
            self.conditions.extend(tree_conditions)
            
            if (tree_idx + 1) % 10 == 0:
                print(f"  Processed {tree_idx + 1}/{self.n_estimators} trees...")
        
        # Remove duplicates while preserving order
        unique_conditions = []
        seen = set()
        for cond in self.conditions:
            cond_str = json.dumps(cond['rules'], sort_keys=True)
            if cond_str not in seen:
                seen.add(cond_str)
                unique_conditions.append(cond)
        
        self.conditions = unique_conditions
        
        print(f"[OK] Extracted {len(self.conditions)} unique conditions from {self.n_estimators} trees")
    
    def _extract_tree_conditions(self, tree, tree_idx: int) -> List[Dict]:
        """
        Extract all decision paths from a single tree
        
        Returns:
            List of conditions, each with:
            - tree_id: Which tree
            - path_id: Which path in tree
            - rules: List of (feature, operator, threshold)
            - leaf_value: Prediction at leaf
        """
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value
        
        conditions = []
        
        def recurse(node, path_rules):
            """Recursively traverse tree and collect paths"""
            if tree.feature[node] != _tree.TREE_UNDEFINED:  # Not a leaf
                name = self.feature_names[feature[node]]
                thresh = threshold[node]
                
                # Left child: feature <= threshold
                left_rules = path_rules + [(name, '<=', thresh)]
                recurse(tree.children_left[node], left_rules)
                
                # Right child: feature > threshold
                right_rules = path_rules + [(name, '>', thresh)]
                recurse(tree.children_right[node], right_rules)
            else:
                # Reached leaf: save this path as a condition
                if len(path_rules) > 0:  # Only if path has rules
                    conditions.append({
                        'tree_id': tree_idx,
                        'path_id': len(conditions),
                        'rules': path_rules,
                        'leaf_value': value[node].flatten().tolist(),
                        'n_samples': tree.n_node_samples[node]
                    })
        
        recurse(0, [])
        return conditions
    
    def conditions_to_features(self, state: pd.Series) -> np.ndarray:
        """
        Convert state to binary features based on conditions
        
        Args:
            state: Health state (e.g., TotalSteps=8000, Sleep_Score=75, ...)
        
        Returns:
            Binary feature vector [0, 1, 1, 0, ...] where each position
            indicates if that condition is satisfied
        """
        features = []
        
        for condition in self.conditions:
            # Check if ALL rules in this condition are satisfied
            satisfied = self._check_condition(condition['rules'], state)
            features.append(1.0 if satisfied else 0.0)
        
        return np.array(features)
    
    def _check_condition(self, rules: List[Tuple], state: pd.Series) -> bool:
        """
        Check if state satisfies all rules in condition
        
        Args:
            rules: [(feature, operator, threshold), ...]
            state: Current health state
        
        Returns:
            True if all rules satisfied
        """
        for feature, operator, threshold in rules:
            if feature not in state:
                return False
            
            value = state[feature]
            
            if operator == '<=':
                if not (value <= threshold):
                    return False
            elif operator == '>':
                if not (value > threshold):
                    return False
        
        return True
    
    def conditions_to_text(
        self, 
        state: pd.Series, 
        top_k: int = 5,
        include_all: bool = False
    ) -> str:
        """
        Convert satisfied conditions to natural language text for T5 prompt
        
        Args:
            state: Current health state
            top_k: Number of most important conditions to include
            include_all: If True, include all satisfied conditions (may be long)
        
        Returns:
            Text description like:
            "Current health conditions:
            - Your daily steps are low (less than 5,000 steps)
            - Your sleep quality is poor (score below 70)
            - You are moderately active (30-60 minutes of activity)
            ..."
        """
        satisfied_conditions = []
        
        for cond in self.conditions:
            if self._check_condition(cond['rules'], state):
                # Convert rules to text
                text = self._rules_to_text(cond['rules'])
                importance = cond['n_samples']  # More samples = more important
                satisfied_conditions.append((importance, text))
        
        # Sort by importance and take top_k
        satisfied_conditions.sort(reverse=True, key=lambda x: x[0])
        
        if not include_all:
            satisfied_conditions = satisfied_conditions[:top_k]
        
        # Format as bullet points
        if len(satisfied_conditions) == 0:
            return "Current health conditions: No specific patterns detected."
        
        text_lines = ["Current health conditions:"]
        for _, text in satisfied_conditions:
            text_lines.append(f"- {text}")
        
        return "\n".join(text_lines)
    
    def _rules_to_text(self, rules: List[Tuple]) -> str:
        """
        Convert rules to natural language
        
        Example:
            [('TotalSteps', '<=', 5000), ('Sleep_Score', '<=', 70)]
            ->
            "Your daily steps are low (<=5,000) AND your sleep quality is poor (<=70)"
        """
        text_parts = []
        
        for feature, operator, threshold in rules:
            # Create human-readable feature names
            feature_text = self._humanize_feature_name(feature)
            
            # Create human-readable comparisons
            if operator == '<=':
                comparison = f"{feature_text} is low (<={threshold:.0f})"
            else:  # '>'
                comparison = f"{feature_text} is high (>{threshold:.0f})"
            
            text_parts.append(comparison)
        
        # Join with AND
        return " AND ".join(text_parts)
    
    def _humanize_feature_name(self, feature: str) -> str:
        """
        Convert feature names to human-readable text
        """
        mapping = {
            'TotalSteps': 'your daily steps',
            'TotalDistance': 'your total distance',
            'Calories': 'your calorie burn',
            'TotalActiveMinutes': 'your active minutes',
            'VeryActiveMinutes': 'your intense activity time',
            'FairlyActiveMinutes': 'your moderate activity time',
            'LightlyActiveMinutes': 'your light activity time',
            'SedentaryMinutes': 'your sedentary time',
            'Activity_Score': 'your activity score',
            'Sleep_Score': 'your sleep quality',
            'Nutrition_Score': 'your nutrition score',
            'Overall_Health_Score': 'your overall health score',
            'DayOfWeekNum': 'the day of week',
            'IsWeekend': 'whether it is weekend'
        }
        
        return mapping.get(feature, feature.lower().replace('_', ' '))
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest
        """
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance
    
    def save_conditions(self, filepath: str):
        """
        Save extracted conditions to JSON file
        """
        # Convert numpy types to Python native types for JSON serialization
        conditions_serializable = []
        for cond in self.conditions:
            cond_copy = {}
            for key, value in cond.items():
                if key == 'rules':
                    # Convert each rule tuple
                    rules_serializable = []
                    for feature, operator, threshold in value:
                        rules_serializable.append([
                            str(feature),
                            str(operator),
                            float(threshold) if isinstance(threshold, (np.floating, np.integer)) else threshold
                        ])
                    cond_copy[key] = rules_serializable
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    cond_copy[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    cond_copy[key] = float(value)
                else:
                    cond_copy[key] = value
            conditions_serializable.append(cond_copy)
        
        with open(filepath, 'w') as f:
            json.dump({
                'n_conditions': len(self.conditions),
                'n_estimators': int(self.n_estimators),
                'max_depth': int(self.max_depth) if self.max_depth else None,
                'conditions': conditions_serializable
            }, f, indent=2)
        
        print(f"[OK] Conditions saved to {filepath}")
    
    def load_conditions(self, filepath: str):
        """
        Load conditions from JSON file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.conditions = data['conditions']
        self.n_estimators = data['n_estimators']
        self.max_depth = data['max_depth']
        
        print(f"✓ Loaded {len(self.conditions)} conditions from {filepath}")


def create_target_variable(df: pd.DataFrame, method: str = 'classification') -> pd.Series:
    """
    Create target variable for RF training
    
    Args:
        df: Health data with Overall_Health_Score
        method: 
            - 'classification': Classify improvement (0=worse, 1=maintain, 2=improve)
            - 'regression': Predict next day's health score
            - 'improvement': Binary improve/not improve
    
    Returns:
        Target variable y
    """
    if method == 'classification':
        # Calculate health improvement from previous day
        df_sorted = df.sort_values(['Id', 'ActivityDate'])
        df_sorted['Health_Delta'] = df_sorted.groupby('Id')['Overall_Health_Score'].diff()
        
        # Classify into 3 classes
        conditions = [
            df_sorted['Health_Delta'] < -2,  # Worse
            (df_sorted['Health_Delta'] >= -2) & (df_sorted['Health_Delta'] <= 2),  # Maintain
            df_sorted['Health_Delta'] > 2  # Improve
        ]
        choices = [0, 1, 2]
        y = np.select(conditions, choices, default=1)
        
        # Remove first day (no previous day to compare)
        valid_idx = df_sorted['Health_Delta'].notna()
        
        return pd.Series(y, index=df_sorted.index)[valid_idx], df_sorted[valid_idx]
    
    elif method == 'improvement':
        # Binary: health improved or not
        df_sorted = df.sort_values(['Id', 'ActivityDate'])
        df_sorted['Health_Delta'] = df_sorted.groupby('Id')['Overall_Health_Score'].diff()
        
        y = (df_sorted['Health_Delta'] > 0).astype(int)
        valid_idx = df_sorted['Health_Delta'].notna()
        
        return pd.Series(y, index=df_sorted.index)[valid_idx], df_sorted[valid_idx]
    
    else:  # regression
        # Predict next day's health score
        df_sorted = df.sort_values(['Id', 'ActivityDate'])
        df_sorted['Next_Health_Score'] = df_sorted.groupby('Id')['Overall_Health_Score'].shift(-1)
        
        valid_idx = df_sorted['Next_Health_Score'].notna()
        y = df_sorted['Next_Health_Score'][valid_idx]
        
        return y, df_sorted[valid_idx]


if __name__ == "__main__":
    # Example usage
    print("Loading health data...")
    df = pd.read_csv('../Processed_Data/rl_training_data_normalized.csv')
    
    # Create target: classify health improvement
    y, df_valid = create_target_variable(df, method='classification')
    
    # Select features for RF
    feature_cols = [
        'TotalSteps', 'TotalDistance', 'Calories',
        'TotalActiveMinutes', 'VeryActiveMinutes', 'FairlyActiveMinutes',
        'LightlyActiveMinutes', 'SedentaryMinutes',
        'Activity_Score', 'Sleep_Score', 'Nutrition_Score',
        'Overall_Health_Score', 'DayOfWeekNum', 'IsWeekend'
    ]
    
    X = df_valid[feature_cols]
    
    print(f"\nTraining Random Forest...")
    print(f"Samples: {len(X)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Train RF extractor
    rf_extractor = RandomForestExtractor(
        n_estimators=100,
        max_depth=5,
    )
    
    rf_extractor.train(X, y)
    
    # Test: Convert state to features
    example_state = X.iloc[0]
    condition_features = rf_extractor.conditions_to_features(example_state)
    
    print(f"\nExample state condition features:")
    print(f"Shape: {condition_features.shape}")
    print(f"Active conditions: {condition_features.sum():.0f}/{len(condition_features)}")
    
    # Test: Convert conditions to text
    text = rf_extractor.conditions_to_text(example_state, top_k=5)
    print(f"\nExample text for T5 prompt:")
    print(text)
    
    # Feature importance
    importance = rf_extractor.get_feature_importance()
    print(f"\nTop 5 important features:")
    print(importance.head())
    
    # Save conditions
    rf_extractor.save_conditions('../data/rf_conditions.json')
