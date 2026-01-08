"""
Validación cruzada para medir el accuracy REAL del modelo
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Cargar TODOS los datos
X_all = np.load('sequences_processed/X_train.npy')
y_all = np.load('sequences_processed/y_train.npy')

# Agregar valid y test también
X_valid = np.load('sequences_processed/X_valid.npy')
y_valid = np.load('sequences_processed/y_valid.npy')
X_test = np.load('sequences_processed/X_test.npy')
y_test = np.load('sequences_processed/y_test.npy')

# Combinar todo
X_combined = np.vstack([X_all, X_valid, X_test])
y_combined = np.concatenate([y_all, y_valid, y_test])

print(f"📊 Total de datos: {X_combined.shape}")
print(f"   Muestras: {len(X_combined)}")
print(f"   Por gesto: {len(X_combined) // 5}")

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X_combined)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*50}")
    
    X_train_fold = X_combined[train_idx]
    y_train_fold = y_combined[train_idx]
    X_test_fold = X_combined[test_idx]
    y_test_fold = y_combined[test_idx]
    
    print(f"Train: {len(X_train_fold)} samples")
    print(f"Test: {len(X_test_fold)} samples")
    
    # Modelo simple CNN
    model = models.Sequential([
        layers.Input(shape=(150, 3)),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar
    model.fit(
        X_train_fold, y_train_fold,
        epochs=50,
        batch_size=4,
        verbose=0
    )
    
    # Evaluar
    y_pred = np.argmax(model.predict(X_test_fold, verbose=0), axis=1)
    acc = accuracy_score(y_test_fold, y_pred)
    accuracies.append(acc)
    
    print(f"Accuracy: {acc*100:.2f}%")

print("\n" + "="*50)
print("📊 RESULTADOS DE VALIDACIÓN CRUZADA")
print("="*50)
print(f"Accuracy por fold: {[f'{a*100:.1f}%' for a in accuracies]}")
print(f"Media: {np.mean(accuracies)*100:.2f}%")
print(f"Std: {np.std(accuracies)*100:.2f}%")
print(f"Min: {np.min(accuracies)*100:.2f}%")
print(f"Max: {np.max(accuracies)*100:.2f}%")

print("\n💡 Interpretación:")
if np.mean(accuracies) > 0.95:
    print("   ⚠️  >95% - Posible overfitting o dataset muy fácil")
elif np.mean(accuracies) > 0.85:
    print("   ✅ 85-95% - Buen rendimiento realista")
elif np.mean(accuracies) > 0.70:
    print("   ⚠️  70-85% - Rendimiento moderado, necesita más datos")
else:
    print("   ❌ <70% - Modelo necesita mejoras")