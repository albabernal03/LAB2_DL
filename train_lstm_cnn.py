"""
Entrenamiento de modelos Deep Learning (LSTM y CNN) para clasificación de gestos
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

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

BATCH_SIZE = 4  # Dataset pequeño
EPOCHS = 100
PATIENCE = 20  # Early stopping

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

input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
num_classes = len(np.unique(y_train))

print(f"\n📊 Input shape: {input_shape}")
print(f"📊 Number of classes: {num_classes}")

# ============================================
# MODELO 1: LSTM BIDIRECCIONAL
# ============================================

def build_lstm_model(input_shape, num_classes):
    """
    LSTM Bidireccional con regularización
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # LSTM bidireccional
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),
        
        # Capas densas
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ], name='LSTM_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# MODELO 2: CNN 1D
# ============================================

def build_cnn1d_model(input_shape, num_classes):
    """
    CNN 1D para series temporales
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloque convolucional 1
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Bloque convolucional 2
        layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Bloque convolucional 3
        layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        
        # Capas densas
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN1D_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# MODELO 3: CNN + LSTM HÍBRIDO
# ============================================

def build_hybrid_model(input_shape, num_classes):
    """
    Modelo híbrido: CNN + LSTM
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN para extracción de features
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # LSTM para patrones temporales
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.LSTM(32),
        layers.Dropout(0.3),
        
        # Capas densas
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ], name='Hybrid_CNN_LSTM')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# CALLBACKS
# ============================================

def get_callbacks(model_name):
    """
    Callbacks para entrenamiento
    """
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
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
            filepath=os.path.join(MODELS_DIR, f'{model_name}_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

# ============================================
# FUNCIÓN DE ENTRENAMIENTO
# ============================================

def train_and_evaluate_model(model, model_name):
    """
    Entrena y evalúa un modelo
    """
    print("\n" + "="*60)
    print(f"🚀 ENTRENANDO: {model_name}")
    print("="*60)
    
    # Resumen del modelo
    model.summary()
    
    # Entrenamiento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1
    )
    
    # Evaluación en test
    print("\n" + "="*60)
    print(f"📊 EVALUACIÓN: {model_name}")
    print("="*60)
    
    # Predicciones
    y_pred_valid = np.argmax(model.predict(X_valid, verbose=0), axis=1)
    y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # Métricas
    acc_train = history.history['accuracy'][-1]
    acc_valid = accuracy_score(y_valid, y_pred_valid)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTrain Accuracy:      {acc_train*100:.2f}%")
    print(f"Validation Accuracy: {acc_valid*100:.2f}%")
    print(f"Test Accuracy:       {acc_test*100:.2f}%")
    
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=GESTURE_NAMES))
    
    print("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Guardar historia
    history_dict = {
        'model_name': model_name,
        'history': history.history,
        'acc_train': acc_train,
        'acc_valid': acc_valid,
        'acc_test': acc_test,
        'cm': cm
    }
    
    with open(os.path.join(MODELS_DIR, f'{model_name}_history.pkl'), 'wb') as f:
        pickle.dump(history_dict, f)
    
    return history_dict

# ============================================
# ENTRENAMIENTO DE TODOS LOS MODELOS
# ============================================

results = {}

# Modelo 1: LSTM
print("\n" + "🔵"*30)
lstm_model = build_lstm_model(input_shape, num_classes)
results['LSTM'] = train_and_evaluate_model(lstm_model, 'LSTM')

# Modelo 2: CNN 1D
print("\n" + "🟢"*30)
cnn_model = build_cnn1d_model(input_shape, num_classes)
results['CNN1D'] = train_and_evaluate_model(cnn_model, 'CNN1D')

# Modelo 3: Híbrido
print("\n" + "🟡"*30)
hybrid_model = build_hybrid_model(input_shape, num_classes)
results['Hybrid'] = train_and_evaluate_model(hybrid_model, 'Hybrid')

# ============================================
# COMPARACIÓN DE RESULTADOS
# ============================================

print("\n" + "="*60)
print("📊 COMPARACIÓN FINAL DE MODELOS")
print("="*60)

comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'Train Acc': result['acc_train'],
        'Valid Acc': result['acc_valid'],
        'Test Acc': result['acc_test']
    })

# Tabla de comparación
print(f"\n{'Model':<15} {'Train Acc':<12} {'Valid Acc':<12} {'Test Acc':<12}")
print("-" * 60)
for data in comparison_data:
    print(f"{data['Model']:<15} {data['Train Acc']*100:>10.2f}% {data['Valid Acc']*100:>10.2f}% {data['Test Acc']*100:>10.2f}%")

# Mejor modelo
best_model_name = max(comparison_data, key=lambda x: x['Test Acc'])['Model']
best_acc = max(comparison_data, key=lambda x: x['Test Acc'])['Test Acc']

print("\n" + "="*60)
print(f"🏆 MEJOR MODELO: {best_model_name}")
print(f"🎯 Test Accuracy: {best_acc*100:.2f}%")
print("="*60)

# ============================================
# VISUALIZACIÓN
# ============================================

print("\n📈 Generando visualizaciones...")

# Gráfico 1: Curvas de entrenamiento
fig, axes = plt.subplots(len(results), 2, figsize=(14, 4*len(results)))

for idx, (model_name, result) in enumerate(results.items()):
    history = result['history']
    
    # Loss
    axes[idx, 0].plot(history['loss'], label='Train Loss')
    axes[idx, 0].plot(history['val_loss'], label='Val Loss')
    axes[idx, 0].set_title(f'{model_name} - Loss')
    axes[idx, 0].set_xlabel('Epoch')
    axes[idx, 0].set_ylabel('Loss')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[idx, 1].plot(history['accuracy'], label='Train Acc')
    axes[idx, 1].plot(history['val_accuracy'], label='Val Acc')
    axes[idx, 1].set_title(f'{model_name} - Accuracy')
    axes[idx, 1].set_xlabel('Epoch')
    axes[idx, 1].set_ylabel('Accuracy')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
print(f"✅ Curvas de entrenamiento guardadas")

# Gráfico 2: Matrices de confusión
fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))

for idx, (model_name, result) in enumerate(results.items()):
    cm = result['cm']
    ax = axes[idx] if len(results) > 1 else axes
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'{model_name}\nTest Acc: {result["acc_test"]*100:.1f}%', fontweight='bold')
    
    tick_marks = np.arange(len(GESTURE_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([g[:8] for g in GESTURE_NAMES], rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([g[:8] for g in GESTURE_NAMES])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10)
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
print(f"✅ Matrices de confusión guardadas")

# Gráfico 3: Comparación de accuracy
fig, ax = plt.subplots(figsize=(10, 6))
models = [d['Model'] for d in comparison_data]
train_accs = [d['Train Acc']*100 for d in comparison_data]
valid_accs = [d['Valid Acc']*100 for d in comparison_data]
test_accs = [d['Test Acc']*100 for d in comparison_data]

x = np.arange(len(models))
width = 0.25

ax.bar(x - width, train_accs, width, label='Train', color='#3498db')
ax.bar(x, valid_accs, width, label='Valid', color='#2ecc71')
ax.bar(x + width, test_accs, width, label='Test', color='#e74c3c')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Comparación de Modelos - Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 105])

# Añadir valores en las barras
for i, (train, valid, test) in enumerate(zip(train_accs, valid_accs, test_accs)):
    ax.text(i - width, train + 1, f'{train:.1f}', ha='center', fontsize=9)
    ax.text(i, valid + 1, f'{valid:.1f}', ha='center', fontsize=9)
    ax.text(i + width, test + 1, f'{test:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
print(f"✅ Comparación de modelos guardada")

print("\n" + "="*60)
print("✅ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"\n📁 Modelos y resultados guardados en: {MODELS_DIR}/")
print(f"   - Mejores modelos: *_best.keras")
print(f"   - Historias: *_history.pkl")
print(f"   - Visualizaciones: *.png")