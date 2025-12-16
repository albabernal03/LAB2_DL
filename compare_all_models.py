"""
Unified Model Comparison Script
Compares all optimized models: Decision Tree, FNN, LSTM, CNN, Hybrid
Analyzes overfitting, hyperparameter importance, and final performance
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================
# CONFIGURATION
# ============================================
MODELS_DIR = "models_dl"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

GESTURE_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

# ============================================
# LOAD ALL RESULTS
# ============================================
print("="*60)
print("📊 UNIFIED MODEL COMPARISON")
print("="*60)

results_summary = []

# Decision Tree (from models/)
try:
    with open('models/dt_grid_search_results.pkl', 'rb') as f:
        dt_results = pickle.load(f)
    results_summary.append({
        'Model': 'Decision Tree',
        'Best Val Acc': dt_results.best_score_,
        'Combinations Tested': len(dt_results.cv_results_['params']),
        'Has L2': False,
        'Has Dropout': False
    })
    print("✅ Loaded Decision Tree results")
except:
    print("⚠️  Decision Tree results not found")

# FNN
try:
    with open(os.path.join(MODELS_DIR, '../models/nn_hyperparameter_results.pkl'), 'rb') as f:
        fnn_results = pickle.load(f)
    results_summary.append({
        'Model': 'FNN (Enhanced)',
        'Best Val Acc': fnn_results['best_val_acc'],
        'Combinations Tested': len(fnn_results['all_results']),
        'Has L2': True,
        'Has Dropout': True,
        'Best L2': fnn_results['best_params'].get('l2_lambda', 0)
    })
    print("✅ Loaded FNN results")
except Exception as e:
    print(f"⚠️  FNN results not found: {e}")

# LSTM
try:
    with open(os.path.join(MODELS_DIR, 'lstm_hyperparameter_results.pkl'), 'rb') as f:
        lstm_results = pickle.load(f)
    results_summary.append({
        'Model': 'LSTM (Optimized)',
        'Best Val Acc': lstm_results['best_val_acc'],
        'Combinations Tested': len(lstm_results['all_results']),
        'Has L2': True,
        'Has Dropout': True,
        'Has Recurrent Dropout': True,
        'Best L2': lstm_results['best_params']['l2_lambda']
    })
    print("✅ Loaded LSTM results")
except Exception as e:
    print(f"⚠️  LSTM results not found: {e}")

# CNN
try:
    with open(os.path.join(MODELS_DIR, 'cnn_hyperparameter_results.pkl'), 'rb') as f:
        cnn_results = pickle.load(f)
    results_summary.append({
        'Model': 'CNN 1D (Optimized)',
        'Best Val Acc': cnn_results['best_val_acc'],
        'Combinations Tested': len(cnn_results['all_results']),
        'Has L2': True,
        'Has Dropout': True,
        'Best L2': cnn_results['best_params']['l2_lambda']
    })
    print("✅ Loaded CNN results")
except Exception as e:
    print(f"⚠️  CNN results not found: {e}")

# Hybrid
try:
    with open(os.path.join(MODELS_DIR, 'hybrid_hyperparameter_results.pkl'), 'rb') as f:
        hybrid_results = pickle.load(f)
    results_summary.append({
        'Model': 'Hybrid CNN+LSTM',
        'Best Val Acc': hybrid_results['best_val_acc'],
        'Combinations Tested': len(hybrid_results['all_results']),
        'Has L2': True,
        'Has Dropout': True,
        'Best L2': hybrid_results['best_params']['l2_lambda']
    })
    print("✅ Loaded Hybrid results")
except Exception as e:
    print(f"⚠️  Hybrid results not found: {e}")

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "="*60)
print("📋 MODEL COMPARISON SUMMARY")
print("="*60)

df_summary = pd.DataFrame(results_summary)
print("\n", df_summary.to_string(index=False))

# ============================================
# VISUALIZATION 1: Model Performance Comparison
# ============================================
print("\n📈 Generating comparison visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of validation accuracy
models = df_summary['Model'].values
val_accs = df_summary['Best Val Acc'].values * 100

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
bars = axes[0].bar(range(len(models)), val_accs, color=colors[:len(models)])
axes[0].set_xticks(range(len(models)))
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 105])

# Add value labels
for i, (bar, acc) in enumerate(zip(bars, val_accs)):
    axes[0].text(bar.get_x() + bar.get_width()/2, acc + 1, 
                f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Combinations tested
combinations = df_summary['Combinations Tested'].values
bars2 = axes[1].bar(range(len(models)), combinations, color=colors[:len(models)])
axes[1].set_xticks(range(len(models)))
axes[1].set_xticklabels(models, rotation=45, ha='right')
axes[1].set_ylabel('Combinations Tested', fontsize=12)
axes[1].set_title('Hyperparameter Search Exhaustiveness', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, comb in zip(bars2, combinations):
    axes[1].text(bar.get_x() + bar.get_width()/2, comb * 1.2, 
                f'{comb:,}', ha='center', fontsize=9, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison_summary.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison_summary.png")

# ============================================
# VISUALIZATION 2: Regularization Analysis
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# FNN L2 impact
if 'fnn_results' in locals():
    fnn_df = pd.DataFrame(fnn_results['all_results'])
    for l2 in sorted(fnn_df['l2_lambda'].unique()):
        data = fnn_df[fnn_df['l2_lambda'] == l2]['val_accuracy']
        axes[0, 0].scatter([l2]*len(data), data, alpha=0.3, s=20)
    axes[0, 0].set_xlabel('L2 Lambda')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('FNN: L2 Regularization Impact')
    axes[0, 0].grid(True, alpha=0.3)

# LSTM L2 impact
if 'lstm_results' in locals():
    lstm_df = pd.DataFrame(lstm_results['all_results'])
    axes[0, 1].boxplot([lstm_df[lstm_df['l2_lambda']==l]['val_accuracy'].values 
                        for l in sorted(lstm_df['l2_lambda'].unique())])
    axes[0, 1].set_xticklabels([f'{l:.3f}' for l in sorted(lstm_df['l2_lambda'].unique())])
    axes[0, 1].set_xlabel('L2 Lambda')
    axes[0, 1].set_ylabel('Validation Accuracy')
    axes[0, 1].set_title('LSTM: L2 Regularization Impact')
    axes[0, 1].grid(True, alpha=0.3)

# CNN L2 impact
if 'cnn_results' in locals():
    cnn_df = pd.DataFrame(cnn_results['all_results'])
    axes[1, 0].boxplot([cnn_df[cnn_df['l2_lambda']==l]['val_accuracy'].values 
                        for l in sorted(cnn_df['l2_lambda'].unique())])
    axes[1, 0].set_xticklabels([f'{l:.3f}' for l in sorted(cnn_df['l2_lambda'].unique())])
    axes[1, 0].set_xlabel('L2 Lambda')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].set_title('CNN: L2 Regularization Impact')
    axes[1, 0].grid(True, alpha=0.3)

# Hybrid L2 impact
if 'hybrid_results' in locals():
    hybrid_df = pd.DataFrame(hybrid_results['all_results'])
    axes[1, 1].boxplot([hybrid_df[hybrid_df['l2_lambda']==l]['val_accuracy'].values 
                        for l in sorted(hybrid_df['l2_lambda'].unique())])
    axes[1, 1].set_xticklabels([f'{l:.3f}' for l in sorted(hybrid_df['l2_lambda'].unique())])
    axes[1, 1].set_xlabel('L2 Lambda')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Hybrid: L2 Regularization Impact')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'l2_regularization_analysis.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: l2_regularization_analysis.png")

# ============================================
# VISUALIZATION 3: Overfitting Analysis
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_idx = 0
for results_name, results_data in [('FNN', fnn_results if 'fnn_results' in locals() else None),
                                     ('LSTM', lstm_results if 'lstm_results' in locals() else None),
                                     ('CNN', cnn_results if 'cnn_results' in locals() else None),
                                     ('Hybrid', hybrid_results if 'hybrid_results' in locals() else None)]:
    if results_data is None:
        continue
    
    df = pd.DataFrame(results_data['all_results'])
    ax = axes[plot_idx // 2, plot_idx % 2]
    
    scatter = ax.scatter(df['train_accuracy'], df['val_accuracy'], 
                        c=df['overfitting_gap'], cmap='RdYlGn_r', 
                        alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect fit')
    ax.set_xlabel('Train Accuracy')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'{results_name}: Overfitting Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Overfitting Gap')
    
    plot_idx += 1

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'overfitting_analysis_all_models.png'), dpi=300, bbox_inches='tight')
print("✅ Saved: overfitting_analysis_all_models.png")

# ============================================
# EXPORT RESULTS TO CSV
# ============================================
print("\n💾 Exporting detailed results...")

# Export summary
df_summary.to_csv(os.path.join(FIGURES_DIR, 'model_comparison_summary.csv'), index=False)
print("✅ Saved: model_comparison_summary.csv")

# Export detailed results for each model
for name, results in [('fnn', fnn_results if 'fnn_results' in locals() else None),
                      ('lstm', lstm_results if 'lstm_results' in locals() else None),
                      ('cnn', cnn_results if 'cnn_results' in locals() else None),
                      ('hybrid', hybrid_results if 'hybrid_results' in locals() else None)]:
    if results is not None:
        df = pd.DataFrame(results['all_results'])
        df.to_csv(os.path.join(FIGURES_DIR, f'{name}_detailed_results.csv'), index=False)
        print(f"✅ Saved: {name}_detailed_results.csv")

# ============================================
# FINAL REPORT
# ============================================
print("\n" + "="*60)
print("📊 FINAL REPORT")
print("="*60)

print("\n🏆 Best Model by Validation Accuracy:")
best_idx = df_summary['Best Val Acc'].idxmax()
best_model = df_summary.loc[best_idx]
print(f"  Model: {best_model['Model']}")
print(f"  Validation Accuracy: {best_model['Best Val Acc']*100:.2f}%")
print(f"  Combinations Tested: {best_model['Combinations Tested']:,}")

print("\n📈 Regularization Techniques Used:")
for _, row in df_summary.iterrows():
    print(f"\n  {row['Model']}:")
    print(f"    - L2 Regularization: {'✅' if row.get('Has L2', False) else '❌'}")
    print(f"    - Dropout: {'✅' if row.get('Has Dropout', False) else '❌'}")
    if row.get('Has Recurrent Dropout', False):
        print(f"    - Recurrent Dropout: ✅")
    if row.get('Best L2') is not None:
        print(f"    - Best L2 Lambda: {row['Best L2']}")

print("\n📊 Total Hyperparameter Combinations Tested:")
total_combinations = df_summary['Combinations Tested'].sum()
print(f"  {total_combinations:,} combinations across all models")

print("\n" + "="*60)
print("✅ COMPARISON COMPLETE")
print("="*60)
print(f"\n📁 Results saved in: {FIGURES_DIR}/")
print("   - model_comparison_summary.png")
print("   - l2_regularization_analysis.png")
print("   - overfitting_analysis_all_models.png")
print("   - *.csv (detailed results)")
