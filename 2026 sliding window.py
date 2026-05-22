import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = '/Users/ts/Desktop/midprogress/data_results.csv'

try:
    df = pd.read_csv(file_path)
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce').astype(int)
    df['interest_similarity'] = pd.to_numeric(df['interest_similarity'], errors='coerce')
    df.dropna(inplace=True)
except FileNotFoundError:
    print(f"failed to find the document '{file_path}'")
    exit()

target_depths = [1, 2, 3, 4]
df_plot = df[df['depth'].isin(target_depths)].copy()


stats = df_plot.groupby('depth')['interest_similarity'].agg(['mean', 'count', 'std']).reset_index()

stats['sem'] = stats['std'] / np.sqrt(stats['count'])
stats['ci_95'] = 1.96 * stats['sem']


stats['mean_smoothed'] = stats['mean'].rolling(window=2, min_periods=1).mean()


y_values = stats['mean_smoothed']

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8, 5))


ax.fill_between(
    stats['depth'],
    stats['mean'] - stats['ci_95'],
    stats['mean'] + stats['ci_95'],
    color='#d65f5f',
    alpha=0.2,
    label='95% Confidence Interval'
)


ax.plot(
    stats['depth'],
    y_values,
    color='#d62728',
    marker='o',
    linewidth=2.5,
    markersize=8,
    label='Coefficient (β)'
)


ax.set_title('Interest Similarity vs. Depth (Smoothed)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Propagation Depth', fontsize=12)
ax.set_ylabel('Interest Similarity Coefficient', fontsize=12)


ax.set_xticks(target_depths)


ax.set_ylim(0.060, 0.080)

ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()