import json

# Load notebook
with open('FitBit_Complete_EDA.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# List of emojis to remove (using actual characters)
emojis_to_remove = [
    '✅', '❌', '📊', '📈', '📉', '🔬', '🔍', '⚠️', '📁', '📅', 
    '⏰', '⏱️', '🏥', '🤖', '🧠', '💡', '📋', '🏃', '💤', '🎯', 
    '📌', '✓', '×', '➜', '🔄', '👥'
]

def clean_text(text):
    """Remove emojis from text"""
    for emoji in emojis_to_remove:
        text = text.replace(emoji, '')
    return text

# Clean all cells
cells_cleaned = 0
for cell in notebook['cells']:
    if 'source' in cell:
        original = cell['source']
        if isinstance(original, list):
            cleaned = []
            for line in original:
                # Skip decorative print lines
                if line.strip() in ['print("="*80)', 'print("-"*80)', 'print("="*80)\n', 'print("-"*80)\n']:
                    continue
                cleaned_line = clean_text(line)
                cleaned.append(cleaned_line)
            cell['source'] = cleaned
            if cleaned != original:
                cells_cleaned += 1
        elif isinstance(original, str):
            cleaned = clean_text(original)
            # Remove decorative print lines
            lines = cleaned.split('\n')
            lines = [l for l in lines if l.strip() not in ['print("="*80)', 'print("-"*80)']]
            cell['source'] = '\n'.join(lines)
            if cleaned != original:
                cells_cleaned += 1

# Save cleaned notebook
with open('FitBit_Complete_EDA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook cleaned successfully!")
print(f"- Cleaned {cells_cleaned} cells")
print(f"- Removed all emojis and decorative elements")
