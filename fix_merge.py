import json

# Load notebook
with open('FitBit_Complete_EDA.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the cell with the merge issue
for idx, cell in enumerate(notebook['cells']):
    if 'source' in cell and '# 1. Daily Health Data' in ''.join(cell.get('source', [])):
        source_lines = cell['source']
        new_lines = []
        
        i = 0
        while i < len(source_lines):
            line = source_lines[i]
            
            # Insert dtype conversion code after the comment
            if '# Merge transformations back to daily_with_sleep' in line:
                new_lines.append(line)
                new_lines.append('# Ensure ActivityDate is same type in both dataframes for merge\n')
                new_lines.append('daily_with_sleep_temp = daily_with_sleep.copy()\n')
                new_lines.append('daily_activity_temp = daily_activity.copy()\n')
                new_lines.append("daily_with_sleep_temp['ActivityDate'] = daily_with_sleep_temp['ActivityDate'].astype(str)\n")
                new_lines.append("daily_activity_temp['ActivityDate'] = daily_activity_temp['ActivityDate'].astype(str)\n")
                new_lines.append('\n')
                i += 1
            # Replace the merge line to use _temp dataframes
            elif 'daily_export_full = daily_with_sleep.merge(' in line:
                new_lines.append('daily_export_full = daily_with_sleep_temp.merge(\n')
                i += 1
            elif "daily_activity[['Id', 'ActivityDate'," in line:
                new_lines.append("    daily_activity_temp[['Id', 'ActivityDate', 'TotalSteps_BoxCox', 'VeryActiveMinutes_Sqrt', 'LightlyActiveMinutes_BoxCox']],\n")
                i += 1
            else:
                new_lines.append(line)
                i += 1
        
        notebook['cells'][idx]['source'] = new_lines
        break

# Save the fixed notebook
with open('FitBit_Complete_EDA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Fixed the merge error!")
print("- Added dtype conversion for ActivityDate columns")
print("- Both dataframes now converted to string before merge")
