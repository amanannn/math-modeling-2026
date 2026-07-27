"""
用途: 机器学习回归预测 — 四模型对比演示脚本
     对比 LinearRegression, Ridge, RandomForestRegressor, MLPRegressor
     在合成数据上的预测性能，输出可视化结果

输入: 无（自动生成合成数据 y = 3*x1 + 2*x2 + 0.5*x3 + noise）
输出: 控制台打印各模型的 RMSE / R^2 / MAE 指标
     保存对比图至 output/ml_prediction_result.png (300dpi)

调用示例:
    python ml_prediction_demo.py

依赖:
    numpy, matplotlib, sklearn
"""

# ── 非交互后端（服务器环境 / Git Bash 无 GUI 时使用）──
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ════════════════════════════════════════════════════════════
# 1. 生成合成数据
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  ML Regression Prediction -- 4-Model Comparison")
print("=" * 60)

np.random.seed(42)
n_samples = 500
n_features = 5

X = np.random.randn(n_samples, n_features)
# 真实关系: y = 3*x1 + 2*x2 + 0.5*x3 + noise
# 特征 x4, x5 为冗余噪声特征
true_coef = np.array([3.0, 2.0, 0.5, 0.0, 0.0])
noise = np.random.randn(n_samples) * 0.8
y = X @ true_coef + noise

print(f"\n[Data] Samples: {n_samples}, Features: {n_features}")
print(f"[Data] True coefficients: {true_coef}")
print(f"[Data] Noise std: 0.8")

# ════════════════════════════════════════════════════════════
# 2. 训练 / 测试 分割
# ════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[Split] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# 标准化（Ridge / MLP 需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ════════════════════════════════════════════════════════════
# 3. 模型定义
# ════════════════════════════════════════════════════════════
models = {
    'Linear': LinearRegression(),
    'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu',
        alpha=0.001, max_iter=500, early_stopping=True,
        validation_fraction=0.1, random_state=42
    ),
}

# ════════════════════════════════════════════════════════════
# 4. 训练 & 评估
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Model Evaluation Results")
print("=" * 60)

results = []
predictions = {}

for name, model in models.items():
    # Ridge / MLP need scaled data; Linear / RF use raw data
    if name in ('Ridge', 'MLP'):
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    predictions[name] = y_pred

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

    # Extra model-specific info
    extra = ""
    if name == 'Linear':
        extra = f"  | Coefs: {np.round(model.coef_, 3)}"
    elif name == 'Ridge':
        extra = f"  | Best alpha: {model.alpha_:.4f}"
    elif name == 'RandomForest':
        extra = f"  | Feature importance: {np.round(model.feature_importances_, 3)}"
    elif name == 'MLP':
        extra = f"  | Iterations: {model.n_iter_}"

    print(f"  [{name:12s}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R^2={r2:.4f}{extra}")

# 找出最佳模型
best = max(results, key=lambda r: r['R2'])
print(f"\n>>> Best model: {best['Model']}  (R^2 = {best['R2']:.4f})")

# ════════════════════════════════════════════════════════════
# 5. 可视化：三合一对比图
# ════════════════════════════════════════════════════════════
print("\n[Plot] Generating comparison figure ...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = {'Linear': '#2196F3', 'Ridge': '#4CAF50',
          'RandomForest': '#FF9800', 'MLP': '#E91E63'}
model_labels = {'Linear': 'Linear Regression', 'Ridge': 'Ridge',
                'RandomForest': 'Random Forest', 'MLP': 'MLP Neural Network'}

# ─── (a) 预测值 vs 真实值 (左上) ───
ax1 = axes[0, 0]
for name in models:
    ax1.scatter(y_test, predictions[name], alpha=0.35, s=10,
                color=colors[name], label=model_labels[name], edgecolors='none')

# 理想对角线
lims = [y_test.min() - 1, y_test.max() + 1]
ax1.plot(lims, lims, 'k--', lw=1.0, alpha=0.6)
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')
ax1.set_title('(a) Predictions vs Actual')
ax1.legend(fontsize=8, markerscale=2)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ─── (b) R² 条形对比图 (右上) ───
ax2 = axes[0, 1]
model_names = [r['Model'] for r in results]
r2_scores = [r['R2'] for r in results]
bar_colors = [colors[n] for n in model_names]
bars = ax2.bar(model_names, r2_scores, color=bar_colors, edgecolor='white', width=0.6)

# 在每个柱上标注数值
for bar, score in zip(bars, r2_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylim(0, 1.1)
ax2.set_ylabel('$R^2$ Score')
ax2.set_title('(b) Model Comparison — $R^2$')
ax2.grid(True, axis='y', alpha=0.3)

# ─── (c) 残差图 (左下) — 以最佳模型为例 ───
ax3 = axes[1, 0]
best_pred = predictions[best['Model']]
residuals = y_test - best_pred
ax3.scatter(best_pred, residuals, alpha=0.5, s=15,
            color=colors[best['Model']], edgecolors='k', linewidth=0.3)
ax3.axhline(y=0, color='r', linestyle='--', lw=1.5)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Residual (Actual - Predicted)')
ax3.set_title(f'(c) Residual Plot — {model_labels[best["Model"]]}')
ax3.grid(True, alpha=0.3)

# ─── (d) 各模型残差箱线图 (右下) ───
ax4 = axes[1, 1]
residual_data = [y_test - predictions[n] for n in model_names]
bp = ax4.boxplot(residual_data, tick_labels=model_names, patch_artist=True,
                 widths=0.5, showmeans=True, meanprops=dict(marker='D',
                 markerfacecolor='white', markeredgecolor='black'))
for patch, color in zip(bp['boxes'], bar_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax4.axhline(y=0, color='r', linestyle='--', lw=1.0, alpha=0.6)
ax4.set_ylabel('Residual')
ax4.set_title('(d) Residual Distribution by Model')
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout(pad=2.0)

# 保存
output_path = 'D:/虚拟C盘/数学建模培训/output/ml_prediction_result.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[Save] Figure saved to: {output_path}")

# ════════════════════════════════════════════════════════════
# 6. 汇总表
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Model Comparison Summary")
print("=" * 60)
sep_line = '-' * 42
print(f"  {'Model':12s}  {'RMSE':>8s}  {'MAE':>8s}  {'R^2':>8s}")
print(f"  {sep_line}")
for r in results:
    print(f"  {r['Model']:12s}  {r['RMSE']:8.4f}  {r['MAE']:8.4f}  {r['R2']:8.4f}")
print("=" * 60)
print("  Script finished successfully")
print("=" * 60)
