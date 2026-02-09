import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'

# 1. Load data
# Note: Use the correct path for your environment
file_path = 'results/all_shapley.json' 
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_tokens_data = []

# 2. Parsing the JSON structure
for run_id, run_content in data.items():
    tokens_dict = run_content.get('tokens', {})
    for t_key, t_info in tokens_dict.items():
        all_tokens_data.append({
            'run': run_id,
            'token': t_info['decoded'],
            'position': t_info['position'],
            'shapley_value': t_info['shapley_value']
        })

# Create DataFrame
df = pd.DataFrame(all_tokens_data)

# 3. Calculate Means
# Get the top 20 tokens by mean value
token_means = df.groupby('token')['shapley_value'].mean().sort_values(ascending=False).head(20)

# Sort them ascending (from smallest to largest) for the requested plot order
token_means_sorted = token_means.sort_values(ascending=True)

# Mean by position for the second plot
pos_means = df.groupby('position')['shapley_value'].mean().reset_index()

tokens = token_means_sorted.index
values = token_means_sorted.values

# 4. Plotting
sns.set_theme(style="whitegrid")

# Plot 1: Top 20 Influential Tokens (Vertical Bars)
plt.figure(figsize=(15, 8))
# Use the index as x (tokens) and values as y (shapley)
ax = sns.barplot(x=tokens, y=values, palette="flare")

plt.title('Top 20 Tokens by Mean Shapley Value (Global Importance)', fontsize=16)
plt.xlabel('Token', fontsize=13)
plt.ylabel('Mean Shapley Value', fontsize=13)

# Rotate labels to improve readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('./results/top_tokens_shapley.png')
plt.show()

# Plot 2: Mean Shapley Values by Position
plt.figure(figsize=(14, 6))
sns.lineplot(data=pos_means, x='position', y='shapley_value', color='teal', linewidth=2)
plt.fill_between(pos_means['position'], pos_means['shapley_value'], alpha=0.3, color='teal')

plt.title('Mean Token Importance by Position in Prompt', fontsize=16)
plt.xlabel('Position (Token Index)', fontsize=13)
plt.ylabel('Mean Shapley Value', fontsize=13)

plt.tight_layout()
plt.savefig('./results/shapley_by_position.png')
plt.show()

# Quick console check
print("Top 10 tokens by importance (English labels applied):")
print(token_means.head(10))