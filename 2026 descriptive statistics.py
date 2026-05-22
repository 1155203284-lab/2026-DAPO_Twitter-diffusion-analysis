import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque  # 【修复1】导入 deque


# ==========================================
# Section 1. Data loading and cleaning
# ==========================================
def load_and_preprocess(file_path):
    print("🚀 Loading the data...")
    df = pd.read_csv(file_path)

    # 统一列名
    mapping = {
        'tweet_id': 'tweet_id',
        'parent_id': 'parent_id',
        'retweeter_id': 'user_id',
        'interest_similarity': 'interest_similarity'
    }
    df = df.rename(columns=mapping)

    # 清洗
    df.dropna(subset=['tweet_id', 'user_id'], inplace=True)

    # 类型转换
    df['tweet_id'] = df['tweet_id'].astype(str)
    df['parent_id'] = df['parent_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)

    print(f"✅ Data loading completed, total records: {len(df)}")
    return df


FILE_PATH = '/Users/ts/Desktop/midprogress/data_results.csv'
df = load_and_preprocess(FILE_PATH)


# ==========================================
# Section 2. Analysis
# ==========================================
def analyze_depth_decay(df):
    print("\nConducting depth attenuation analysis...")

    # 1. 构建父子字典
    parent_to_children = defaultdict(list)
    for _, row in df.iterrows():
        parent = row['parent_id']
        child = row['tweet_id']
        if parent != 'nan':
            parent_to_children[parent].append(child)

    # 【调试】打印字典大小，看看有多少个父节点
    print(f"构建的父子关系组数: {len(parent_to_children)}")
    # 打印拥有最多子节点的那个父节点，它很可能才是真正的源头
    if parent_to_children:
        max_parent = max(parent_to_children, key=lambda k: len(parent_to_children[k]))
        print(f"拥有最多子节点的父节点 (可能是真根): {max_parent}, 子节点数: {len(parent_to_children[max_parent])}")

    # 2. 寻找真正的根节点
    # 逻辑：寻找一个节点，它作为 parent 出现过，但它的 parent 是空的（或者它不在其他节点的 children 列表里）
    # 简单策略：直接取拥有最多子节点的那个父节点作为根（假设它是源头）
    if parent_to_children:
        root_id = max(parent_to_children, key=lambda k: len(parent_to_children[k]))
        print(f"✅ 自动选定最大传播源为根节点: {root_id}")
    else:
        print("⚠️ 无父子关系")
        return df, None

    # 3. BFS 计算深度
    depth_map = {root_id: 0}
    queue = deque([root_id])  # 【修复2】使用 deque

    while queue:
        current = queue.popleft()
        children = parent_to_children.get(current, [])

        for child in children:
            if child not in depth_map:
                depth_map[child] = depth_map[current] + 1
                queue.append(child)

    # 4. 映射回 DataFrame
    df['depth'] = df['tweet_id'].map(depth_map).fillna(0).astype(int)

    print(f"Calculated Depths (Unique values): {df['depth'].unique()}")

    # --- 绘图 ---
    stats = df.groupby('depth')['interest_similarity'].agg(['mean', 'count', 'std']).reset_index()
    stats.columns = ['calculated_depth', 'Mean_Similarity', 'Node_Count', 'Std_Deviation']

    if stats.empty:
        print("No data to plot.")
        return df, None

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Propagation Depth')
    ax1.set_ylabel('Mean Interest Similarity', color=color)
    ax1.plot(stats['calculated_depth'], stats['Mean_Similarity'], marker='o', color=color, label='Mean Similarity')
    ax1.fill_between(stats['calculated_depth'],
                     stats['Mean_Similarity'] - stats['Std_Deviation'],
                     stats['Mean_Similarity'] + stats['Std_Deviation'],
                     alpha=0.2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Node Count (Log Scale)', color=color)
    ax2.bar(stats['calculated_depth'], stats['Node_Count'], alpha=0.3, color=color, label='Node Count')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    plt.title(f'Decay Analysis (Root: {root_id[:10]}...)')
    fig.tight_layout()
    plt.show()

    return df, stats


df_analyzed, decay_stats = analyze_depth_decay(df)