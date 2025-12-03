"""
Decision Tree Optimized Model with GridSearchCV
Lab 2 - Arduino Gesture Recognition
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# Configuration
BASE_DIR = "."

GESTURE_DIRS = [
    "clockwise_dataset",
    "horizontal_swipe_dataset",
    "forward_thrust_dataset",
    "vertical_updown_dataset",
    "wrist_twist_dataset",
]

LABEL_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

def load_all_pkl_files(gesture_dir, split):
    """Load all .pkl files from a folder"""
    split_path = os.path.join(BASE_DIR, gesture_dir, split)
    
    if not os.path.exists(split_path):
        print(f"⚠️  Does not exist: {split_path}")
        return None
    
    all_data = []
    pkl_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        file_path = os.path.join(split_path, pkl_file)
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            all_data.append(df)
    
    if len(all_data) == 0:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def load_dataset_split(split):
    """Load train, valid or test from ALL gestures"""
    all_dfs = []
    
    for gesture_dir in GESTURE_DIRS:
        df = load_all_pkl_files(gesture_dir, split)
        
        if df is not None:
            all_dfs.append(df)
            print(f"  ✓ {gesture_dir}/{split}: {len(df)} samples")
    
    if len(all_dfs) == 0:
        return None, None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Extract gyroscope data (ωx, ωy, ωz)
    X = combined_df[['ωx', 'ωy', 'ωz']].values
    y = combined_df['label'].values
    
    return X, y

# ============================================
# LOAD DATA
# ============================================
print("="*60)
print("DECISION TREE WITH HYPERPARAMETER OPTIMIZATION")
print("="*60)

print("\n📂 Loading data...")
print("\n--- TRAIN ---")
X_train, y_train = load_dataset_split('train')

print("\n--- VALID ---")
X_valid, y_valid = load_dataset_split('valid')

print("\n--- TEST ---")
X_test, y_test = load_dataset_split('test')

if X_train is None or X_valid is None or X_test is None:
    print("❌ Error: Could not load all splits")
    exit(1)

print(f"\n📊 Dataset sizes:")
print(f"  Train: {X_train.shape}")
print(f"  Valid: {X_valid.shape}")
print(f"  Test:  {X_test.shape}")

# ============================================
# NORMALIZATION
# ============================================
print("\n⚙️  Normalizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# ============================================
# HYPERPARAMETER OPTIMIZATION
# ============================================
print("\n🔍 Starting hyperparameter optimization with GridSearchCV...")
print("This may take a few minutes...\n")

# Define parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}

# Create base model
dt_base = DecisionTreeClassifier(random_state=42)

# GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=dt_base,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# Fit on training data
grid_search.fit(X_train_scaled, y_train)

# ============================================
# BEST MODEL RESULTS
# ============================================
print("\n" + "="*60)
print("🏆 BEST HYPERPARAMETERS FOUND")
print("="*60)
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best CV Score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# Get best model
best_model = grid_search.best_estimator_

# ============================================
# EVALUATION ON VALIDATION SET
# ============================================
print("\n" + "="*60)
print("📊 VALIDATION SET PERFORMANCE")
print("="*60)
y_pred_valid = best_model.predict(X_valid_scaled)
acc_valid = accuracy_score(y_valid, y_pred_valid)
f1_valid = f1_score(y_valid, y_pred_valid, average='weighted')

print(f"Accuracy: {acc_valid:.4f} ({acc_valid*100:.2f}%)")
print(f"F1-Score: {f1_valid:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_valid, target_names=LABEL_NAMES))

# ============================================
# EVALUATION ON TEST SET
# ============================================
print("\n" + "="*60)
print("📊 TEST SET PERFORMANCE")
print("="*60)
y_pred_test = best_model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average='weighted')

print(f"Accuracy: {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"F1-Score: {f1_test:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=LABEL_NAMES))

print("\n📋 Confusion Matrix (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# ============================================
# FEATURE IMPORTANCE
# ============================================
print("\n" + "="*60)
print("🔍 FEATURE IMPORTANCE")
print("="*60)
feature_names = ['ωx (gyro_x)', 'ωy (gyro_y)', 'ωz (gyro_z)']
importances = best_model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")

# ============================================
# SAVE MODEL
# ============================================
os.makedirs('models', exist_ok=True)

print("\n💾 Saving model...")
with open('models/decision_tree_optimized.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('models/scaler_dt_optimized.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save grid search results
with open('models/dt_grid_search_results.pkl', 'wb') as f:
    pickle.dump(grid_search, f)
    
print("✅ Model saved: models/decision_tree_optimized.pkl")
print("✅ Scaler saved: models/scaler_dt_optimized.pkl")
print("✅ Grid search results saved: models/dt_grid_search_results.pkl")

# ============================================
# VISUALIZATION - CONFUSION MATRIX
# ============================================
print("\n📈 Generating visualizations...")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Validation confusion matrix
cm_valid = confusion_matrix(y_valid, y_pred_valid)
sns.heatmap(cm_valid, annot=True, fmt='d', cmap='Blues', 
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title(f'Validation Set\nAccuracy: {acc_valid*100:.2f}%', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)
axes[0].tick_params(axis='x', rotation=45)

# Test confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title(f'Test Set\nAccuracy: {acc_test*100:.2f}%', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/decision_tree_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrices saved: figures/decision_tree_confusion_matrices.png")

# Feature importance plot
plt.figure(figsize=(10, 6))
bars = plt.bar(feature_names, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('Feature Importance - Decision Tree', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance', fontsize=12)
plt.ylim(0, max(importances) * 1.2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('figures/decision_tree_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance saved: figures/decision_tree_feature_importance.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("📊 FINAL SUMMARY - DECISION TREE OPTIMIZED")
print("="*60)
print(f"Best CV Score:        {grid_search.best_score_*100:.2f}%")
print(f"Validation Accuracy:  {acc_valid*100:.2f}%")
print(f"Test Accuracy:        {acc_test*100:.2f}%")
print(f"Validation F1-Score:  {f1_valid:.4f}")
print(f"Test F1-Score:        {f1_test:.4f}")
print(f"\nOverfitting Check:")
print(f"  Train-Valid gap:    {abs(grid_search.best_score_ - acc_valid)*100:.2f}%")
print(f"  Valid-Test gap:     {abs(acc_valid - acc_test)*100:.2f}%")
print("="*60)
print("\n✅ Decision Tree optimization complete!")
