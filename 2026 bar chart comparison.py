import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Loading the data
# Setting the file path
FILE_PATH = '/Users/ts/Desktop/midprogress/data_results.csv'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ file successfully loaded: {FILE_PATH}")
    print(f"ℹ️ data overview: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"📋 checking the column name: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ Error: please check if the filepath is accurate: {FILE_PATH}")
    exit()
except Exception as e:
    print(f"❌ error in reading the data: {e}")
    exit()


def calculate_threshold_effect(data, depth_range, sim_col='interest_similarity', target_col='retweeted'):
    """
    Calculate the optimal threshold value within the given depths
    """
    # Filtering the data
    # Confirm the existence of the "depth" column
    if 'depth' not in data.columns:
        print(f"❌ error: the 'depth' column is absent，unable to filter depths")
        return None

    mask = data['depth'].isin(depth_range)
    subset = data[mask].copy()

    if len(subset) < 50:
        return None

    X = subset[sim_col].values
    y = subset[target_col].values

    # setting up the range of thresholds
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_t = None
    min_loss = float('inf')

    for t in thresholds:
        # Constructing Hinge features
        hinge = np.maximum(0, X - t)
        X_mat = np.column_stack([np.ones(len(X)), X, hinge])

        try:
            # (Might need to delete disp=False is there is version Incompatibility
            model = sm.Logit(y, X_mat)
            result = model.fit(maxiter=100, disp=False)
            pred = result.predict()
            loss = log_loss(y, pred)

            if loss < min_loss:
                min_loss = loss
                best_t = t
        except:
            continue

    if best_t is None:
        return None

    # Quantifying the effect
    low_group = subset[subset[sim_col] <= best_t]
    high_group = subset[subset[sim_col] > best_t]

    rate_low = low_group[target_col].mean() if len(low_group) > 0 else 0
    rate_high = high_group[target_col].mean() if len(high_group) > 0 else 0

    # Prevent 0 from being the denominator
    lift = (rate_high / rate_low) if rate_low > 0 else float('inf')

    return {
        'depth_range': depth_range,
        'threshold': best_t,
        'rate_low': rate_low,
        'rate_high': rate_high,
        'lift': lift,
        'samples': len(subset)
    }


# ==========================================
# 2. Comparing the data in the two stages of diffusion
# ==========================================
print("\n🔍 Conducting depth-based searching...\n")

# Defining shallow depths and deep depths
shallow_result = calculate_threshold_effect(df, depth_range=[1, 2])
deep_result = calculate_threshold_effect(df, depth_range=[3, 4])

results = [r for r in [shallow_result, deep_result] if r is not None]

if results:
    # Print the title of the table
    print(f"{'depth':<12} | {'optimal threshold':<10} | {'retweeting rate of the lower quartile':<12} | {'retweeting rate of the upper quartile':<12} | {'conversion lift':<10} | {'sample size'}")
    print("-" * 85)

    for res in results:
        range_str = f"Depth {res['depth_range']}"

        # Percentage formatting
        rate_low_str = f"{res['rate_low'] * 100:.2f}%"
        rate_high_str = f"{res['rate_high'] * 100:.2f}%"

        # Processing the infinitly large values of conversion lift
        lift_str = f"{res['lift']:.2f}x" if res['lift'] != float('inf') else "Inf"

        print(
            f"{range_str:<12} | {res['threshold']:.2f}      | {rate_low_str:<12} | {rate_high_str:<12} | {lift_str:<10} | {res['samples']}")

    # ==========================================
    # 3. Visualisation
    # Use a Chinese font
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False

    # Generating the graph
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Comparing the threshold values
    depths_plot = [f"Depth {r['depth_range']}" for r in results]
    thresholds_plot = [r['threshold'] for r in results]
    bars1 = ax[0].bar(depths_plot, thresholds_plot, color=['skyblue', 'salmon'], edgecolor='black', alpha=0.7)
    ax[0].set_ylabel('Optimal Threshold Value', fontsize=12)
    ax[0].set_title('Comparison of Optimal Interest Similarity Thresholds',
                    fontsize=14, fontweight='bold')
    ax[0].axhline(np.mean(thresholds_plot), color='gray', linestyle='--',
                  label=f'Average ({np.mean(thresholds_plot):.2f})')
    ax[0].legend()
    ax[0].grid(axis='y', alpha=0.3, linestyle='--')
    # Marking the columns with numerical values
    for bar, val in zip(bars1, thresholds_plot):
        ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom',
                   fontsize=11)

    # Comparing the conversion lift
    lifts_plot = [r['lift'] for r in results]
    max_lift_for_plot = max([l for l in lifts_plot if l != float('inf')] + [0]) * 1.1
    lifts_for_plot = [min(l, max_lift_for_plot) if l != float('inf') else max_lift_for_plot for l in lifts_plot]

    bars2 = ax[1].bar(depths_plot, lifts_for_plot, color=['skyblue', 'salmon'], edgecolor='black', alpha=0.7)
    ax[1].set_ylabel('Lift Ratio (High/Low Group)', fontsize=12)
    ax[1].set_title('Conversion Lift by Interest Similarity Threshold',
                    fontsize=14, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3, linestyle='--')

    # 在柱子上标数值
    for i, (bar, real_val) in enumerate(zip(bars2, lifts_plot)):
        label = f"{real_val:.2f}x" if real_val != float('inf') else "Inf"
        ax[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, label, ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Conclusions
    if len(results) == 2:
        t_shallow = results[0]['threshold']
        t_deep = results[1]['threshold']

        print("\n💡 analysis:")
        if abs(t_shallow - t_deep) < 0.05:
            print(f"   -> The high similarity in optimal threshold (约 {t_shallow:.2f}) indicates a stable standard of decision making among users")
        elif t_shallow > t_deep:
            print(f"   -> The threshold of the primary stage ({t_shallow:.2f}) is higher than that of the secondary stage ({t_deep:.2f})。")
            print("      Explanation：users at the primary stage are more picky that they only retweet when the content highly matches their interests.")
            print("      However, users' retweeting behavior at the secondary stage relies more on social connection, and they have lower demand for interest similarity")
        else:
            print(f"   -> The threshold of the secondary stage ({t_deep:.2f}) is higher than that of the primary stage ({t_shallow:.2f})。")
            print("      Explanation：As the propagation deepens, only the content that highly matches users' interests can be retweeted, indicating a strong filter effect")
else:
    print("❌ Inadequate data for conducting interpretation")