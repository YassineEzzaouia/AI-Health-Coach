import json
import re

# Load notebook
with open('FitBit_Complete_EDA.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Comprehensive emoji pattern - matches all Unicode emojis
emoji_pattern = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "\u2600-\u26FF"          # Miscellaneous Symbols
    "\u2700-\u27BF"          # Dingbats
    "]+",
    flags=re.UNICODE
)

# Pattern for decorative lines (=== or ---)
decorative_pattern = re.compile(r'^\s*print\(["\'][-=]{3,}["\']\)\s*$', re.MULTILINE)

def clean_cell_source(source):
    """Remove emojis and decorative print statements from cell source"""
    if isinstance(source, list):
        cleaned = []
        for line in source:
            # Skip lines that are just print("===") or print("---")
            if re.match(r'^\s*print\(["\'][-=]{3,}["\']\)\s*$', line.strip()):
                continue
            # Remove emojis from the line
            cleaned_line = emoji_pattern.sub('', line)
            cleaned.append(cleaned_line)
        return cleaned
    elif isinstance(source, str):
        # Remove decorative print lines
        cleaned = re.sub(decorative_pattern, '', source)
        # Remove emojis
        cleaned = emoji_pattern.sub('', cleaned)
        return cleaned
    return source

# Clean all cells
for cell in notebook['cells']:
    if 'source' in cell:
        cell['source'] = clean_cell_source(cell['source'])

# Save cleaned notebook
with open('FitBit_Complete_EDA.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook cleaned successfully!")
print("- Removed all emojis")
print("- Removed decorative print statements (=== and ---)")
