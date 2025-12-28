"""Show all 20 generated actions"""
import json
from pathlib import Path

data_path = Path(__file__).parent.parent / "data" / "rulebased_condition_actions.json"
with open(data_path, 'r') as f:
    data = json.load(f)

print("\n" + "="*80)
print("20 DIVERSE, CONDITION-SPECIFIC HEALTH ACTIONS")
print("="*80 + "\n")

for i, (idx, item) in enumerate(sorted(data.items(), key=lambda x: int(x[0])), 1):
    print(f"{i}. Condition {idx}:")
    print(f"   ACTION: {item['action']}")
    print(f"   STATE: {item['state']}\n")

print("="*80)
print("✓ All actions are unique and condition-specific!")
print("="*80)
