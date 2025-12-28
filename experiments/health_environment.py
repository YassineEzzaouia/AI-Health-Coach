"""
Realistic Health Environment for DQN Training
Simulates how health metrics change based on actions taken
"""

import numpy as np
from typing import Dict, Tuple


class HealthEnvironment:
    """
    Simulates realistic health state transitions based on actions.
    Models how daily health metrics evolve with interventions.
    """
    
    def __init__(self, initial_state_data, action_description: str):
        """
        Initialize environment with a starting health state
        
        Args:
            initial_state_data: Pandas Series with health metrics
            action_description: Text description of the action taken
        """
        # Extract metrics with fallback for different column names, ensure valid floats
        def safe_float(value, default):
            try:
                val = float(value)
                return val if not np.isnan(val) and not np.isinf(val) else default
            except (ValueError, TypeError):
                return default
        
        self.state = {
            'Overall_Health_Score': safe_float(initial_state_data.get('Overall_Health_Score', 70), 70),
            'Activity_Score': safe_float(initial_state_data.get('Activity_Score', 60), 60),
            'Sleep_Score': safe_float(initial_state_data.get('Sleep_Score', 70), 70),
            'Nutrition_Score': safe_float(initial_state_data.get('Nutrition_Score', 65), 65),
            'TotalSteps': safe_float(initial_state_data.get('TotalSteps', 5000), 5000) / 15000.0,
            'Calories': safe_float(initial_state_data.get('Calories', 2000), 2000) / 4000.0,
            'TotalActiveMinutes': safe_float(initial_state_data.get('TotalActiveMinutes', 30), 30) / 240.0,
            'SedentaryMinutes': safe_float(initial_state_data.get('SedentaryMinutes', 600), 600) / 1440.0,
            'DayOfWeek': safe_float(initial_state_data.get('DayOfWeekNum', 0), 0)
        }
        self.action_description = action_description
        self.day = 0
    
    def simulate_action_effects(self, action_text: str) -> Dict[str, float]:
        """
        Simulate how an action affects health metrics over one day
        
        Returns:
            Dictionary of delta changes to health metrics
        """
        # Parse action keywords to determine effects
        action_lower = action_text.lower()
        
        # Initialize deltas (how much each metric changes)
        deltas = {
            'Activity_Score': 0.0,
            'Sleep_Score': 0.0,
            'Nutrition_Score': 0.0,
            'TotalSteps': 0.0,
            'Calories': 0.0,
            'TotalActiveMinutes': 0.0,
            'SedentaryMinutes': 0.0
        }
        
        # Cardio/Exercise actions (30-40 mins, 3-5x weekly)
        if any(word in action_lower for word in ['cardio', 'jogging', 'cycling', 'swimming', 'hiit', 'running']):
            deltas['Activity_Score'] += np.random.uniform(3, 6)
            deltas['TotalSteps'] += np.random.uniform(0.05, 0.12)
            deltas['Calories'] += np.random.uniform(0.04, 0.08)
            deltas['TotalActiveMinutes'] += np.random.uniform(0.08, 0.15)
            deltas['SedentaryMinutes'] -= np.random.uniform(0.03, 0.07)
            deltas['Sleep_Score'] += np.random.uniform(1, 3)  # Exercise improves sleep
        
        # Strength/Circuit training
        if any(word in action_lower for word in ['strength', 'circuit', 'weights', 'resistance']):
            deltas['Activity_Score'] += np.random.uniform(2, 5)
            deltas['TotalActiveMinutes'] += np.random.uniform(0.06, 0.12)
            deltas['Calories'] += np.random.uniform(0.03, 0.06)
            deltas['SedentaryMinutes'] -= np.random.uniform(0.02, 0.05)
        
        # Active hobbies (rock climbing, martial arts, sports)
        if any(word in action_lower for word in ['hobbies', 'climbing', 'martial', 'sports', 'team']):
            deltas['Activity_Score'] += np.random.uniform(2, 4)
            deltas['TotalSteps'] += np.random.uniform(0.04, 0.09)
            deltas['TotalActiveMinutes'] += np.random.uniform(0.05, 0.10)
            deltas['Sleep_Score'] += np.random.uniform(1, 2)
        
        # Daily movement (biking, walking, stairs)
        if any(word in action_lower for word in ['movement', 'bike', 'walk', 'commut', 'stairs', 'active errands']):
            deltas['Activity_Score'] += np.random.uniform(1, 3)
            deltas['TotalSteps'] += np.random.uniform(0.08, 0.15)
            deltas['SedentaryMinutes'] -= np.random.uniform(0.04, 0.08)
        
        # Sleep-focused actions
        if any(word in action_lower for word in ['sleep', 'bedtime', 'rest', 'screen time']):
            deltas['Sleep_Score'] += np.random.uniform(4, 8)
            deltas['Activity_Score'] += np.random.uniform(0.5, 1.5)  # Better sleep = more energy
        
        # Consistency/Habits actions
        if any(word in action_lower for word in ['consistency', 'schedule', 'routine', 'meal prep', 'habits']):
            deltas['Activity_Score'] += np.random.uniform(1, 2)
            deltas['Sleep_Score'] += np.random.uniform(1, 2)
            deltas['Nutrition_Score'] += np.random.uniform(2, 4)
        
        # Social/mentoring actions (moderate benefits)
        if any(word in action_lower for word in ['mentor', 'teach', 'share', 'communities']):
            deltas['Activity_Score'] += np.random.uniform(0.5, 1.5)
            deltas['Sleep_Score'] += np.random.uniform(0.5, 1)
        
        # Challenge/goals actions
        if any(word in action_lower for word in ['challenge', 'goals', 'learn new skills']):
            deltas['Activity_Score'] += np.random.uniform(2, 4)
            deltas['TotalActiveMinutes'] += np.random.uniform(0.05, 0.10)
        
        # Varied cardio
        if 'varied' in action_lower or 'variety' in action_lower:
            deltas['Activity_Score'] += np.random.uniform(2, 4)
            deltas['Calories'] += np.random.uniform(0.03, 0.06)
        
        # Add natural variance (not all days are perfect)
        for key in deltas:
            variance = np.random.uniform(-0.5, 0.5)
            deltas[key] += variance
        
        return deltas
    
    def step(self, action_text: str) -> Tuple[Dict[str, float], float, bool]:
        """
        Take one step in the environment
        
        Args:
            action_text: Description of action taken
        
        Returns:
            (new_state, reward, done)
        """
        # Simulate action effects
        deltas = self.simulate_action_effects(action_text)
        
        # Apply changes to state
        old_overall = self.state['Overall_Health_Score']
        old_activity = self.state['Activity_Score']
        old_sleep = self.state['Sleep_Score']
        
        for metric, delta in deltas.items():
            self.state[metric] += delta
            # Clip to valid ranges [0, 100] for scores, [0, 1] for normalized
            if 'Score' in metric:
                self.state[metric] = np.clip(self.state[metric], 0, 100)
            else:
                self.state[metric] = np.clip(self.state[metric], 0, 1)
        
        # Recalculate overall health as weighted average
        self.state['Overall_Health_Score'] = (
            0.35 * self.state['Activity_Score'] +
            0.30 * self.state['Sleep_Score'] +
            0.25 * self.state['Nutrition_Score'] +
            0.10 * (self.state['TotalSteps'] * 100)
        )
        self.state['Overall_Health_Score'] = np.clip(self.state['Overall_Health_Score'], 0, 100)
        
        # Calculate reward
        reward = self._calculate_reward(old_overall, old_activity, old_sleep)
        
        # Increment day
        self.day += 1
        self.state['DayOfWeek'] = (self.state['DayOfWeek'] + 1) % 7
        
        # Check if done (reached goal or max days)
        done = (self.state['Overall_Health_Score'] >= 90) or (self.day >= 30)
        
        return self.state.copy(), reward, done
    
    def _calculate_reward(self, old_overall: float, old_activity: float, old_sleep: float) -> float:
        """
        Calculate reward based on health improvements
        
        Rewards:
        - Large bonus for improving overall health
        - Bonus for maintaining good metrics
        - Penalty for declining health
        """
        # Ensure no NaN values
        old_overall = 0 if np.isnan(old_overall) else old_overall
        old_activity = 0 if np.isnan(old_activity) else old_activity
        old_sleep = 0 if np.isnan(old_sleep) else old_sleep
        
        current_overall = self.state.get('Overall_Health_Score', 70)
        current_activity = self.state.get('Activity_Score', 60)
        current_sleep = self.state.get('Sleep_Score', 70)
        current_steps = self.state.get('TotalSteps', 0.33)
        current_sedentary = self.state.get('SedentaryMinutes', 0.5)
        
        # Overall health change (most important)
        overall_delta = current_overall - old_overall
        reward = overall_delta * 15.0
        
        # Activity improvement
        activity_delta = current_activity - old_activity
        reward += activity_delta * 8.0
        
        # Sleep improvement
        sleep_delta = current_sleep - old_sleep
        reward += sleep_delta * 8.0
        
        # Bonuses for maintaining good levels
        if current_overall >= 85:
            reward += 20
        
        if current_activity >= 85:
            reward += 10
        
        if current_sleep >= 85:
            reward += 10
        
        if current_steps >= 0.6:
            reward += 8
        
        # Penalties for poor metrics
        if current_sedentary > 0.8:
            reward -= 15
        
        if current_sleep < 60:
            reward -= 20
        
        if current_activity < 50:
            reward -= 15
        
        # Ensure reward is not NaN
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        return float(reward)
    
    def get_state(self) -> Dict[str, float]:
        """Get current state"""
        return self.state.copy()
    
    def reset(self, initial_state_data):
        """Reset environment with new initial state"""
        self.__init__(initial_state_data, self.action_description)
        return self.get_state()
