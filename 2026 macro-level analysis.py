import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression

sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 用来正常显示英文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

file_path = '/Users/ts/Desktop/midprogress/data_results.csv'

df = pd.read_csv(file_path, encoding='utf-8-sig')
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df['diffusion_type'] = df['depth'].apply(lambda x: 'Broadcast (Shallow)' if x <= 2 else 'Viral (Deep)')

plt.figure(figsize=(12, 6))
sns.lmplot(
    data=df,
    x='interest_similarity',
    y='retweeted',
    hue='diffusion_type',
    col='diffusion_type',
    logistic=True,  # 因为y是二分类变量，使用逻辑回归拟合
    truncate=False,
    scatter_kws={'alpha': 0.3, 's': 10},
    line_kws={'linewidth': 2}
)

plt.suptitle('The effect of interest similarity on retweeting：Broadcast diffusion vs Viral diffsuion', y=1.02, fontsize=16)
plt.savefig('visualization_result_01.png', format='png', dpi=300, bbox_inches='tight')
print()
plt.show()