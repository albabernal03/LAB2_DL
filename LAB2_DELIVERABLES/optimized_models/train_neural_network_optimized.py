"""
Neural Network Optimized Model with Hyperparameter Tuning
Lab 2 - Arduino Gesture Recognition
Using Keras/TensorFlow with GridSearchCV
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from itertools import product

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
BASE_DIR = "../.."  # Point to LAB2_DL root directory

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

def create_model(hidden_layers=2, neurons=64, dropout_rate=0.3, learning_rate=0.001, 
                 l2_lambda=0.0, activation='relu', optimizer_name='adam'):
    """
    Create a neural network model with configurable architecture
    
    Args:
        hidden_layers: Number of hidden layers
        neurons: Number of neurons per hidden layer
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        l2_lambda: L2 regularization strength
        activation: Activation function ('relu', 'elu', 'selu')
        optimizer_name: Optimizer to use ('adam', 'rmsprop')
    """
    from tensorflow.keras.optimizers import RMSprop
    
    model = Sequential()
    
    # L2 regularizer
    l2_reg = keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None
    
    # Input layer
    model.add(Dense(neurons, activation=activation, input_shape=(3,),
                   kernel_regularizer=l2_reg))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation,
                       kernel_regularizer=l2_reg))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer (5 classes)
    model.add(Dense(5, activation='softmax',
                   kernel_regularizer=l2_reg))
    
    # Compile model
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)
        
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# LOAD DATA
# ============================================
print("="*60)
print("NEURAL NETWORK WITH HYPERPARAMETER OPTIMIZATION")
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
# ENCODE LABELS
# ============================================
print("\n🔤 Encoding labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)
y_test_encoded = label_encoder.transform(y_test)

print(f"  Classes: {label_encoder.classes_}")

# ============================================
# NORMALIZATION
# ============================================
print("\n⚙️  Normalizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# ============================================
# HYPERPARAMETER OPTIMIZATION (Manual Grid Search)
# ============================================
print("\n🔍 Starting hyperparameter optimization...")
print("Testing different architectures...\n")

# Define OPTIMIZED parameter grid (reduced but still exhaustive)
param_grid = {
    'hidden_layers': [2, 3],                     # 2 options (reduced from 3)
    'neurons': [64, 128],                        # 2 options (reduced from 4)
    'dropout_rate': [0.3, 0.4, 0.5],             # 3 options (reduced from 4)
    'learning_rate': [0.0001, 0.001],            # 2 options (reduced from 4)
    'batch_size': [32, 64],                      # 2 options (reduced from 3)
    'l2_lambda': [0.0, 0.001, 0.01],             # 3 options (reduced from 4)
    'activation': ['relu', 'elu'],               # 2 options (reduced from 3)
    'optimizer': ['adam']                        # 1 option (reduced from 2)
}

# Total: 2 × 2 × 3 × 2 × 2 × 3 × 2 × 1 = 576 combinations (~30-60 min)

# Generate all combinations
param_combinations = list(product(
    param_grid['hidden_layers'],
    param_grid['neurons'],
    param_grid['dropout_rate'],
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['l2_lambda'],
    param_grid['activation'],
    param_grid['optimizer']
))

print(f"Testing {len(param_combinations)} parameter combinations...")

# Track best model
best_val_acc = 0
best_params = None
best_history = None
results = []

# Test each combination
for idx, (hidden_layers, neurons, dropout_rate, learning_rate, batch_size, l2_lambda, activation, optimizer) in enumerate(param_combinations, 1):
    print(f"\n[{idx}/{len(param_combinations)}] Testing: layers={hidden_layers}, neurons={neurons}, dropout={dropout_rate}, lr={learning_rate}, batch={batch_size}, l2={l2_lambda}, act={activation}, opt={optimizer}")
    
    # Create model
    model = create_model(
        hidden_layers=hidden_layers,
        neurons=neurons,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        activation=activation,
        optimizer_name=optimizer
    )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train_encoded,
        validation_data=(X_valid_scaled, y_valid_encoded),
        epochs=50,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Get validation accuracy
    val_acc = max(history.history['val_accuracy'])
    
    # Store results
    results.append({
        'hidden_layers': hidden_layers,
        'neurons': neurons,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'l2_lambda': l2_lambda,
        'activation': activation,
        'optimizer': optimizer,
        'val_accuracy': val_acc
    })
    
    print(f"  → Validation Accuracy: {val_acc*100:.2f}%")
    
    # Update best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {
            'hidden_layers': hidden_layers,
            'neurons': neurons,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'l2_lambda': l2_lambda,
            'activation': activation,
            'optimizer': optimizer
        }
        print(f"  ✨ New best model!")

# ============================================
# BEST MODEL RESULTS
# ============================================
print("\n" + "="*60)
print("🏆 BEST HYPERPARAMETERS FOUND")
print("="*60)
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best Validation Score: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# ============================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================
print("\n🚀 Training final model with best parameters and early stopping...")

# Create final model with best parameters
final_model = create_model(
    hidden_layers=best_params['hidden_layers'],
    neurons=best_params['neurons'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate'],
    l2_lambda=best_params['l2_lambda'],
    activation=best_params['activation'],
    optimizer_name=best_params['optimizer']
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train final model
history = final_model.fit(
    X_train_scaled, y_train_encoded,
    validation_data=(X_valid_scaled, y_valid_encoded),
    epochs=200,  # Use more epochs with early stopping
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================
# EVALUATION ON VALIDATION SET
# ============================================
print("\n" + "="*60)
print("📊 VALIDATION SET PERFORMANCE")
print("="*60)

y_pred_valid_proba = final_model.predict(X_valid_scaled)
y_pred_valid = np.argmax(y_pred_valid_proba, axis=1)
y_pred_valid_labels = label_encoder.inverse_transform(y_pred_valid)

acc_valid = accuracy_score(y_valid_encoded, y_pred_valid)
f1_valid = f1_score(y_valid_encoded, y_pred_valid, average='weighted')

print(f"Accuracy: {acc_valid:.4f} ({acc_valid*100:.2f}%)")
print(f"F1-Score: {f1_valid:.4f}")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_valid_labels, target_names=LABEL_NAMES))

# ============================================
# EVALUATION ON TEST SET
# ============================================
print("\n" + "="*60)
print("📊 TEST SET PERFORMANCE")
print("="*60)

y_pred_test_proba = final_model.predict(X_test_scaled)
y_pred_test = np.argmax(y_pred_test_proba, axis=1)
y_pred_test_labels = label_encoder.inverse_transform(y_pred_test)

acc_test = accuracy_score(y_test_encoded, y_pred_test)
f1_test = f1_score(y_test_encoded, y_pred_test, average='weighted')

print(f"Accuracy: {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"F1-Score: {f1_test:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test_labels, target_names=LABEL_NAMES))

print("\n📋 Confusion Matrix (Test):")
cm = confusion_matrix(y_test_encoded, y_pred_test)
print(cm)

# ============================================
# SAVE MODEL
# ============================================
os.makedirs('models', exist_ok=True)

print("\n💾 Saving model...")
final_model.save('models/neural_network_optimized.h5')
final_model.save('models/neural_network_optimized.keras')

with open('models/scaler_nn_optimized.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/label_encoder_nn.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save hyperparameter search results
with open('models/nn_hyperparameter_results.pkl', 'wb') as f:
    pickle.dump({
        'best_params': best_params,
        'best_val_acc': best_val_acc,
        'all_results': results
    }, f)

# Save training history
with open('models/nn_training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
    
print("✅ Model saved: models/neural_network_optimized.keras")
print("✅ Scaler saved: models/scaler_nn_optimized.pkl")
print("✅ Label encoder saved: models/label_encoder_nn.pkl")
print("✅ Hyperparameter results saved: models/nn_hyperparameter_results.pkl")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n📈 Generating visualizations...")

# 1. Training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/neural_network_training_history.png', dpi=300, bbox_inches='tight')
print("✅ Training history saved: figures/neural_network_training_history.png")

# 2. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Validation confusion matrix
cm_valid = confusion_matrix(y_valid_encoded, y_pred_valid)
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
plt.savefig('figures/neural_network_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrices saved: figures/neural_network_confusion_matrices.png")

# ============================================
# MODEL ARCHITECTURE SUMMARY
# ============================================
print("\n" + "="*60)
print("🏗️  MODEL ARCHITECTURE")
print("="*60)
final_model.summary()

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("📊 FINAL SUMMARY - NEURAL NETWORK OPTIMIZED")
print("="*60)
print(f"Best Validation Score: {best_val_acc*100:.2f}%")
print(f"Validation Accuracy:   {acc_valid*100:.2f}%")
print(f"Test Accuracy:         {acc_test*100:.2f}%")
print(f"Validation F1-Score:   {f1_valid:.4f}")
print(f"Test F1-Score:         {f1_test:.4f}")
print(f"\nTraining Info:")
print(f"  Total epochs:       {len(history.history['loss'])}")
print(f"  Best epoch:         {np.argmin(history.history['val_loss']) + 1}")
print(f"  Final train acc:    {history.history['accuracy'][-1]*100:.2f}%")
print(f"  Final val acc:      {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"\nOverfitting Check:")
print(f"  Train-Valid gap:    {abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1])*100:.2f}%")
print(f"  Valid-Test gap:     {abs(acc_valid - acc_test)*100:.2f}%")
print("="*60)
print("\n✅ Neural Network optimization complete!")
