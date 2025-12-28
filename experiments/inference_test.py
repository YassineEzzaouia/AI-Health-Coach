"""
Quick inference test with new T5 actions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json

from rf_extractor.random_forest_extractor import RandomForestExtractor
from dqn_agent.health_dqn_agent import HealthDQNAgent


def quick_inference_test():
    """Test inference with new actions"""
    
    print("\n" + "="*80)
    print("QUICK INFERENCE TEST WITH IMPROVED T5 ACTIONS")
    print("="*80 + "\n")
    
    # Paths
    SCRIPT_DIR = Path(__file__).parent
    MODEL_PATH = SCRIPT_DIR / "../models/dqn_rulebased_model.pth"
    RF_CONDITIONS_PATH = SCRIPT_DIR / "rf_conditions.json"
    ACTIONS_PATH = SCRIPT_DIR / "../data/rulebased_condition_actions.json"
    DATA_PATH = SCRIPT_DIR / "../../Processed_Data/rl_training_data_normalized.csv"
    
    # Load RF conditions
    print(f"Loading RF conditions...")
    rf_extractor = RandomForestExtractor()
    rf_extractor.load_conditions(str(RF_CONDITIONS_PATH))
    print(f"  [OK] {len(rf_extractor.conditions)} conditions\n")
    
    # Load new T5 actions
    print(f"Loading improved T5 actions...")
    with open(ACTIONS_PATH, 'r') as f:
        t5_actions = json.load(f)
    print(f"  [OK] {len(t5_actions)} actions\n")
    
    print("Sample actions:")
    for i, (idx, action_data) in enumerate(list(t5_actions.items())[:3], 1):
        print(f"\n{i}. Condition {idx}:")
        print(f"   ACTION: {action_data['action'][:80]}")
        print(f"   STATE: {action_data['state'][:80]}")
    
    # Load DQN model
    print(f"\n\nLoading DQN model...")
    state_dim = len(rf_extractor.conditions)
    dqn_agent = HealthDQNAgent(
        state_dim=state_dim,
        use_t5_actions=True,
        learning_rate=1e-4,
        gamma=0.99
    )
    
    # Load T5 actions into agent
    for cond_idx, cond_data in t5_actions.items():
        dqn_agent.add_t5_action(cond_data['action'])
    
    # Load weights
    dqn_agent.load(str(MODEL_PATH))
    dqn_agent.policy_net.eval()
    print(f"  [OK] Model loaded (Action space: {dqn_agent.action_dim})\n")
    
    # Load test data
    print(f"Loading test data...")
    test_data = pd.read_csv(DATA_PATH)
    print(f"  [OK] {len(test_data)} records\n")
    
    # Test on one sample
    print("="*80)
    print("TESTING ON SAMPLE HEALTH STATE")
    print("="*80 + "\n")
    
    sample_idx = np.random.choice(len(test_data))
    health_state = test_data.iloc[sample_idx]
    
    print(f"Sample #{sample_idx}")
    print(f"  Overall Health: {health_state['Overall_Health_Score']:.2f}")
    print(f"  Activity Score: {health_state['Activity_Score']:.2f}")
    print(f"  Sleep Score: {health_state['Sleep_Score']:.2f}")
    print(f"  Nutrition Score: {health_state['Nutrition_Score']:.2f}")
    print(f"  Total Steps: {health_state['TotalSteps']:.2f}")
    
    # Get state vector
    state_vector = rf_extractor.conditions_to_features(health_state)
    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(dqn_agent.device)
    
    # Get recommendations
    with torch.no_grad():
        q_values = dqn_agent.policy_net(state_tensor)
    
    # Top 3 actions
    top_k = min(3, dqn_agent.action_dim)
    top_q_values, top_indices = torch.topk(q_values, top_k, dim=1)
    
    print(f"\n--- DQN RECOMMENDATIONS ---\n")
    for rank, (idx, q_val) in enumerate(zip(top_indices[0].cpu().numpy(), top_q_values[0].cpu().numpy()), 1):
        action_text = dqn_agent.idx_to_action.get(int(idx), f"Action_{idx}")
        confidence = float(torch.softmax(q_values, dim=1)[0, idx].cpu().numpy())
        
        print(f"{rank}. {action_text}")
        print(f"   Q-Value: {q_val:.2f} | Confidence: {confidence*100:.1f}%\n")
    
    # Show satisfied conditions
    satisfied = [i for i, val in enumerate(state_vector) if val == 1]
    print(f"\n--- SATISFIED CONDITIONS: {len(satisfied)} ---")
    for i in satisfied[:5]:
        condition = rf_extractor.conditions[i]
        condition_text = rf_extractor._rules_to_text(condition['rules'])
        print(f"  • {condition_text[:100]}")
    
    print("\n" + "="*80)
    print("✓ INFERENCE TEST COMPLETE!")
    print("="*80)
    print("\nThe actions are now proper health recommendations!")
    print("To retrain DQN with these new actions, run:")
    print("  python experiments/train_pipeline_rulebased.py")


if __name__ == "__main__":
    quick_inference_test()
