"""
LSTM Optimized Model with Exhaustive Hyperparameter Search
Lab 2 - Arduino Gesture Recognition
Using RandomizedSearchCV for efficient hyperparameter exploration
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import product
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# CONFIGURACIÓN
# ============================================
SEQUENCES_DIR = "../../sequences_processed"
MODELS_DIR = "../../models_dl"
os.makedirs(MODELS_DIR, exist_ok=True)

GESTURE_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

# ============================================
# CARGAR DATOS
# ============================================
print("\n" + "="*60)
print("📂 CARGANDO DATOS")
print("="*60)

X_train = np.load(os.path.join(SEQUENCES_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(SEQUENCES_DIR, 'y_train.npy'))

X_valid = np.load(os.path.join(SEQUENCES_DIR, 'X_valid.npy'))
y_valid = np.load(os.path.join(SEQUENCES_DIR, 'y_valid.npy'))

X_test = np.load(os.path.join(SEQUENCES_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(SEQUENCES_DIR, 'y_test.npy'))

print(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
print(f"✅ Valid: X={X_valid.shape}, y={y_valid.shape}")
print(f"✅ Test:  X={X_test.shape}, y={y_test.shape}")

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y_train))

print(f"\n📊 Input shape: {input_shape}")
print(f"📊 Number of classes: {num_classes}")

# ============================================
# FUNCIÓN PARA CREAR MODELO LSTM
# ============================================

def build_lstm_model(lstm_units_1=64, lstm_units_2=32, dropout_lstm=0.3, 
                     dropout_dense=0.2, learning_rate=0.001, l2_lambda=0.0,
                     bidirectional=True, recurrent_dropout=0.0):
    """
    LSTM model with configurable hyperparameters
    
    Args:
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dropout_lstm: Dropout rate after LSTM layers
        dropout_dense: Dropout rate after dense layers
        learning_rate: Learning rate for optimizer
        l2_lambda: L2 regularization strength
        bidirectional: Use bidirectional LSTM
        recurrent_dropout: Recurrent dropout in LSTM
    """
    l2_reg = keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
    ])
    
    # First LSTM layer
    if bidirectional:
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units_1, return_sequences=True,
                       recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=l2_reg,
                       recurrent_regularizer=l2_reg)
        ))
    else:
        model.add(layers.LSTM(lstm_units_1, return_sequences=True,
                             recurrent_dropout=recurrent_dropout,
                             kernel_regularizer=l2_reg,
                             recurrent_regularizer=l2_reg))
    model.add(layers.Dropout(dropout_lstm))
    
    # Second LSTM layer
    if bidirectional:
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units_2,
                       recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=l2_reg,
                       recurrent_regularizer=l2_reg)
        ))
    else:
        model.add(layers.LSTM(lstm_units_2,
                             recurrent_dropout=recurrent_dropout,
                             kernel_regularizer=l2_reg,
                             recurrent_regularizer=l2_reg))
    model.add(layers.Dropout(dropout_lstm))
    
    # Dense layers
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=l2_reg))
    model.add(layers.Dropout(dropout_dense))
    
    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# HYPERPARAMETER GRID
# ============================================

print("\n" + "="*60)
print("🔍 EXHAUSTIVE HYPERPARAMETER SEARCH - LSTM")
print("="*60)

# Define comprehensive parameter grid
param_grid = {
    'lstm_units_1': [32, 64, 128, 256],           # 4 options
    'lstm_units_2': [16, 32, 64, 128],            # 4 options
    'dropout_lstm': [0.2, 0.3, 0.4, 0.5],         # 4 options
    'dropout_dense': [0.2, 0.3, 0.4, 0.5],        # 4 options
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005],  # 4 options
    'batch_size': [4, 8, 16],                     # 3 options
    'l2_lambda': [0.0, 0.001, 0.01],              # 3 options
    'bidirectional': [True, False],               # 2 options
    'recurrent_dropout': [0.0, 0.1, 0.2]          # 3 options
}

# Calculate total combinations
total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)

print(f"\n📊 Total possible combinations: {total_combinations:,}")
print(f"⚡ Using RandomizedSearchCV with 50 iterations for efficiency\n")

# Generate random sample of combinations
np.random.seed(42)
n_iterations = min(50, total_combinations)  # Reduced from 150 to 50

# Sample random combinations
all_combinations = list(product(
    param_grid['lstm_units_1'],
    param_grid['lstm_units_2'],
    param_grid['dropout_lstm'],
    param_grid['dropout_dense'],
    param_grid['learning_rate'],
    param_grid['batch_size'],
    param_grid['l2_lambda'],
    param_grid['bidirectional'],
    param_grid['recurrent_dropout']
))

# Randomly sample
sampled_indices = np.random.choice(len(all_combinations), size=n_iterations, replace=False)
param_combinations = [all_combinations[i] for i in sampled_indices]

print(f"Testing {len(param_combinations)} random combinations...")

# ============================================
# HYPERPARAMETER SEARCH
# ============================================

best_val_acc = 0
best_params = None
results = []

start_time = time.time()

for idx, (lstm_units_1, lstm_units_2, dropout_lstm, dropout_dense, 
          learning_rate, batch_size, l2_lambda, bidirectional, recurrent_dropout) in enumerate(param_combinations, 1):
    
    print(f"\n[{idx}/{len(param_combinations)}] Testing:")
    print(f"  LSTM: {lstm_units_1}→{lstm_units_2}, Bi={bidirectional}, RecDrop={recurrent_dropout}")
    print(f"  Dropout: LSTM={dropout_lstm}, Dense={dropout_dense}")
    print(f"  LR={learning_rate}, Batch={batch_size}, L2={l2_lambda}")
    
    # Create model
    model = build_lstm_model(
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout_lstm=dropout_lstm,
        dropout_dense=dropout_dense,
        learning_rate=learning_rate,
        l2_lambda=l2_lambda,
        bidirectional=bidirectional,
        recurrent_dropout=recurrent_dropout
    )
    
    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=80,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Get best validation accuracy
    val_acc = max(history.history['val_accuracy'])
    train_acc = history.history['accuracy'][np.argmax(history.history['val_accuracy'])]
    
    # Store results
    results.append({
        'lstm_units_1': lstm_units_1,
        'lstm_units_2': lstm_units_2,
        'dropout_lstm': dropout_lstm,
        'dropout_dense': dropout_dense,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'l2_lambda': l2_lambda,
        'bidirectional': bidirectional,
        'recurrent_dropout': recurrent_dropout,
        'val_accuracy': val_acc,
        'train_accuracy': train_acc,
        'overfitting_gap': train_acc - val_acc
    })
    
    print(f"  → Val Acc: {val_acc*100:.2f}%, Train Acc: {train_acc*100:.2f}%, Gap: {(train_acc-val_acc)*100:.2f}%")
    
    # Update best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {
            'lstm_units_1': lstm_units_1,
            'lstm_units_2': lstm_units_2,
            'dropout_lstm': dropout_lstm,
            'dropout_dense': dropout_dense,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'l2_lambda': l2_lambda,
            'bidirectional': bidirectional,
            'recurrent_dropout': recurrent_dropout
        }
        print(f"  ✨ New best model!")

elapsed_time = time.time() - start_time
print(f"\n⏱️  Search completed in {elapsed_time/60:.1f} minutes")

# ============================================
# BEST MODEL RESULTS
# ============================================
print("\n" + "="*60)
print("🏆 BEST HYPERPARAMETERS FOUND")
print("="*60)
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n📊 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

# ============================================
# TRAIN FINAL MODEL
# ============================================
print("\n🚀 Training final LSTM model with best parameters...")

final_model = build_lstm_model(**best_params)

final_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(MODELS_DIR, 'LSTM_optimized_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = final_model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=150,
    batch_size=best_params['batch_size'],
    callbacks=final_callbacks,
    verbose=1
)

# ============================================
# EVALUATION
# ============================================
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

y_pred_valid = np.argmax(final_model.predict(X_valid, verbose=0), axis=1)
y_pred_test = np.argmax(final_model.predict(X_test, verbose=0), axis=1)

acc_train = history.history['accuracy'][-1]
acc_valid = accuracy_score(y_valid, y_pred_valid)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\nTrain Accuracy:      {acc_train*100:.2f}%")
print(f"Validation Accuracy: {acc_valid*100:.2f}%")
print(f"Test Accuracy:       {acc_test*100:.2f}%")
print(f"\nOverfitting Analysis:")
print(f"  Train-Valid gap:   {abs(acc_train - acc_valid)*100:.2f}%")
print(f"  Valid-Test gap:    {abs(acc_valid - acc_test)*100:.2f}%")

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=GESTURE_NAMES))

print("\nConfusion Matrix (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# ============================================
# SAVE RESULTS
# ============================================
print("\n💾 Saving results...")

# Save hyperparameter search results
with open(os.path.join(MODELS_DIR, 'lstm_hyperparameter_results.pkl'), 'wb') as f:
    pickle.dump({
        'best_params': best_params,
        'best_val_acc': best_val_acc,
        'all_results': results,
        'param_grid': param_grid
    }, f)

# Save training history
with open(os.path.join(MODELS_DIR, 'lstm_optimized_history.pkl'), 'wb') as f:
    pickle.dump({
        'history': history.history,
        'acc_train': acc_train,
        'acc_valid': acc_valid,
        'acc_test': acc_test,
        'cm': cm
    }, f)

print(f"✅ Model saved: {MODELS_DIR}/LSTM_optimized_best.keras")
print(f"✅ Results saved: {MODELS_DIR}/lstm_hyperparameter_results.pkl")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n📈 Generating visualizations...")

# 1. Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('LSTM Optimized - Loss', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('LSTM Optimized - Accuracy', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'lstm_optimized_training.png'), dpi=150, bbox_inches='tight')
print("✅ Training curves saved")

# 2. Hyperparameter importance
import pandas as pd

results_df = pd.DataFrame(results)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# L2 lambda impact
axes[0, 0].boxplot([results_df[results_df['l2_lambda']==l]['val_accuracy'].values 
                     for l in sorted(results_df['l2_lambda'].unique())])
axes[0, 0].set_xticklabels([f'{l:.3f}' for l in sorted(results_df['l2_lambda'].unique())])
axes[0, 0].set_title('L2 Regularization Impact')
axes[0, 0].set_xlabel('L2 Lambda')
axes[0, 0].set_ylabel('Validation Accuracy')
axes[0, 0].grid(True, alpha=0.3)

# Dropout impact
axes[0, 1].scatter(results_df['dropout_lstm'], results_df['val_accuracy'], alpha=0.5)
axes[0, 1].set_title('LSTM Dropout Impact')
axes[0, 1].set_xlabel('Dropout Rate')
axes[0, 1].set_ylabel('Validation Accuracy')
axes[0, 1].grid(True, alpha=0.3)

# Learning rate impact
axes[1, 0].boxplot([results_df[results_df['learning_rate']==lr]['val_accuracy'].values 
                     for lr in sorted(results_df['learning_rate'].unique())])
axes[1, 0].set_xticklabels([f'{lr:.4f}' for lr in sorted(results_df['learning_rate'].unique())])
axes[1, 0].set_title('Learning Rate Impact')
axes[1, 0].set_xlabel('Learning Rate')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].grid(True, alpha=0.3)

# Overfitting analysis
axes[1, 1].scatter(results_df['train_accuracy'], results_df['val_accuracy'], 
                   c=results_df['l2_lambda'], cmap='viridis', alpha=0.6)
axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect fit')
axes[1, 1].set_title('Overfitting Analysis')
axes[1, 1].set_xlabel('Train Accuracy')
axes[1, 1].set_ylabel('Validation Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='L2 Lambda')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'lstm_hyperparameter_analysis.png'), dpi=150, bbox_inches='tight')
print("✅ Hyperparameter analysis saved")

print("\n" + "="*60)
print("✅ LSTM OPTIMIZATION COMPLETE")
print("="*60)
print(f"Best Test Accuracy: {acc_test*100:.2f}%")
print(f"Total combinations tested: {len(param_combinations)}")
print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
