"""
Hybrid CNN+LSTM Optimized Model with Hyperparameter Search
Lab 2 - Arduino Gesture Recognition
Combines best of CNN (feature extraction) and LSTM (temporal modeling)
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
SEQUENCES_DIR = "sequences_processed"
MODELS_DIR = "models_dl"
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
# FUNCIÓN PARA CREAR MODELO HÍBRIDO
# ============================================

def build_hybrid_model(cnn_filters_1=64, cnn_filters_2=128, kernel_size=5,
                       lstm_units_1=64, lstm_units_2=32, dropout_cnn=0.2,
                       dropout_lstm=0.3, dropout_dense=0.2, learning_rate=0.001,
                       l2_lambda=0.0, use_batch_norm=True, bidirectional=True):
    """
    Hybrid CNN+LSTM model with configurable hyperparameters
    
    Args:
        cnn_filters_1: Filters in first CNN block
        cnn_filters_2: Filters in second CNN block
        kernel_size: Kernel size for CNN
        lstm_units_1: Units in first LSTM layer
        lstm_units_2: Units in second LSTM layer
        dropout_cnn: Dropout after CNN layers
        dropout_lstm: Dropout after LSTM layers
        dropout_dense: Dropout after dense layers
        learning_rate: Learning rate
        l2_lambda: L2 regularization strength
        use_batch_norm: Use batch normalization in CNN
        bidirectional: Use bidirectional LSTM
    """
    l2_reg = keras.regularizers.l2(l2_lambda) if l2_lambda > 0 else None
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN Block 1 - Feature extraction
        layers.Conv1D(cnn_filters_1, kernel_size=kernel_size, activation='relu',
                     padding='same', kernel_regularizer=l2_reg),
    ])
    
    if use_batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_cnn))
    
    # CNN Block 2
    model.add(layers.Conv1D(cnn_filters_2, kernel_size=max(3, kernel_size-2),
                           activation='relu', padding='same', kernel_regularizer=l2_reg))
    if use_batch_norm:
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(dropout_cnn))
    
    # LSTM Block - Temporal modeling
    if bidirectional:
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units_1, return_sequences=True,
                       kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        ))
    else:
        model.add(layers.LSTM(lstm_units_1, return_sequences=True,
                             kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg))
    model.add(layers.Dropout(dropout_lstm))
    
    # Second LSTM
    model.add(layers.LSTM(lstm_units_2, kernel_regularizer=l2_reg,
                         recurrent_regularizer=l2_reg))
    model.add(layers.Dropout(dropout_lstm))
    
    # Dense layers
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=l2_reg))
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
print("🔍 EXHAUSTIVE HYPERPARAMETER SEARCH - HYBRID CNN+LSTM")
print("="*60)

# Define parameter grid (focused on key hyperparameters)
param_grid = {
    'cnn_filters_1': [32, 64, 128],               # 3 options
    'cnn_filters_2': [64, 128, 256],              # 3 options
    'kernel_size': [3, 5],                        # 2 options
    'lstm_units_1': [32, 64, 128],                # 3 options
    'lstm_units_2': [16, 32, 64],                 # 3 options
    'dropout_cnn': [0.2, 0.3],                    # 2 options
    'dropout_lstm': [0.3, 0.4],                   # 2 options
    'dropout_dense': [0.2, 0.3],                  # 2 options
    'learning_rate': [0.0001, 0.0005, 0.001],     # 3 options
    'l2_lambda': [0.0, 0.001, 0.01],              # 3 options
    'use_batch_norm': [True, False],              # 2 options
    'bidirectional': [True, False],               # 2 options
    'batch_size': [4, 8]                          # 2 options
}

# Calculate total combinations
total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)

print(f"\n📊 Total possible combinations: {total_combinations:,}")
print(f"⚡ Testing all {total_combinations} combinations\n")

# Generate all combinations
param_combinations = list(product(
    param_grid['cnn_filters_1'],
    param_grid['cnn_filters_2'],
    param_grid['kernel_size'],
    param_grid['lstm_units_1'],
    param_grid['lstm_units_2'],
    param_grid['dropout_cnn'],
    param_grid['dropout_lstm'],
    param_grid['dropout_dense'],
    param_grid['learning_rate'],
    param_grid['l2_lambda'],
    param_grid['use_batch_norm'],
    param_grid['bidirectional'],
    param_grid['batch_size']
))

print(f"Testing {len(param_combinations)} combinations...")

# ============================================
# HYPERPARAMETER SEARCH
# ============================================

best_val_acc = 0
best_params = None
results = []

start_time = time.time()

for idx, (cnn_f1, cnn_f2, kernel, lstm_u1, lstm_u2, drop_cnn, drop_lstm,
          drop_dense, lr, l2, batch_norm, bidir, batch_size) in enumerate(param_combinations, 1):
    
    print(f"\n[{idx}/{len(param_combinations)}] Testing:")
    print(f"  CNN: {cnn_f1}→{cnn_f2}, K={kernel}, BN={batch_norm}")
    print(f"  LSTM: {lstm_u1}→{lstm_u2}, Bi={bidir}")
    print(f"  Dropout: CNN={drop_cnn}, LSTM={drop_lstm}, Dense={drop_dense}")
    print(f"  LR={lr}, L2={l2}, Batch={batch_size}")
    
    # Create model
    model = build_hybrid_model(
        cnn_filters_1=cnn_f1,
        cnn_filters_2=cnn_f2,
        kernel_size=kernel,
        lstm_units_1=lstm_u1,
        lstm_units_2=lstm_u2,
        dropout_cnn=drop_cnn,
        dropout_lstm=drop_lstm,
        dropout_dense=drop_dense,
        learning_rate=lr,
        l2_lambda=l2,
        use_batch_norm=batch_norm,
        bidirectional=bidir
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
        'cnn_filters_1': cnn_f1,
        'cnn_filters_2': cnn_f2,
        'kernel_size': kernel,
        'lstm_units_1': lstm_u1,
        'lstm_units_2': lstm_u2,
        'dropout_cnn': drop_cnn,
        'dropout_lstm': drop_lstm,
        'dropout_dense': drop_dense,
        'learning_rate': lr,
        'l2_lambda': l2,
        'use_batch_norm': batch_norm,
        'bidirectional': bidir,
        'batch_size': batch_size,
        'val_accuracy': val_acc,
        'train_accuracy': train_acc,
        'overfitting_gap': train_acc - val_acc
    })
    
    print(f"  → Val Acc: {val_acc*100:.2f}%, Train Acc: {train_acc*100:.2f}%, Gap: {(train_acc-val_acc)*100:.2f}%")
    
    # Update best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {
            'cnn_filters_1': cnn_f1,
            'cnn_filters_2': cnn_f2,
            'kernel_size': kernel,
            'lstm_units_1': lstm_u1,
            'lstm_units_2': lstm_u2,
            'dropout_cnn': drop_cnn,
            'dropout_lstm': drop_lstm,
            'dropout_dense': drop_dense,
            'learning_rate': lr,
            'l2_lambda': l2,
            'use_batch_norm': batch_norm,
            'bidirectional': bidir,
            'batch_size': batch_size
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
print("\n🚀 Training final Hybrid model with best parameters...")

final_model = build_hybrid_model(**{k: v for k, v in best_params.items() if k != 'batch_size'})

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
        filepath=os.path.join(MODELS_DIR, 'Hybrid_optimized_best.keras'),
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

with open(os.path.join(MODELS_DIR, 'hybrid_hyperparameter_results.pkl'), 'wb') as f:
    pickle.dump({
        'best_params': best_params,
        'best_val_acc': best_val_acc,
        'all_results': results,
        'param_grid': param_grid
    }, f)

with open(os.path.join(MODELS_DIR, 'hybrid_optimized_history.pkl'), 'wb') as f:
    pickle.dump({
        'history': history.history,
        'acc_train': acc_train,
        'acc_valid': acc_valid,
        'acc_test': acc_test,
        'cm': cm
    }, f)

print(f"✅ Model saved: {MODELS_DIR}/Hybrid_optimized_best.keras")
print(f"✅ Results saved: {MODELS_DIR}/hybrid_hyperparameter_results.pkl")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n📈 Generating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Hybrid CNN+LSTM - Loss', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Hybrid CNN+LSTM - Accuracy', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'hybrid_optimized_training.png'), dpi=150, bbox_inches='tight')
print("✅ Training curves saved")

print("\n" + "="*60)
print("✅ HYBRID MODEL OPTIMIZATION COMPLETE")
print("="*60)
print(f"Best Test Accuracy: {acc_test*100:.2f}%")
print(f"Total combinations tested: {len(param_combinations)}")
print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
