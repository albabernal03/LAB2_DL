"""
Crear modelo CNN1D más pequeño para Arduino
Reducido para caber en 1MB de Flash
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

print("="*60)
print("CREANDO MODELO CNN1D COMPACTO PARA ARDUINO")
print("="*60)

# ============================================
# CONFIGURACIÓN
# ============================================
SEQUENCES_DIR = "sequences_processed"
MODELS_DIR = "models_dl"

# Cargar datos
print("\n📂 Cargando datos...")
X_train = np.load(os.path.join(SEQUENCES_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(SEQUENCES_DIR, 'y_train.npy'))
X_valid = np.load(os.path.join(SEQUENCES_DIR, 'X_valid.npy'))
y_valid = np.load(os.path.join(SEQUENCES_DIR, 'y_valid.npy'))
X_test = np.load(os.path.join(SEQUENCES_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(SEQUENCES_DIR, 'y_test.npy'))

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = 5

print(f"✅ Datos cargados")
print(f"   Input shape: {input_shape}")

# ============================================
# MODELO COMPACTO
# ============================================

def build_compact_cnn(input_shape, num_classes):
    """
    CNN1D ultra-compacto para Arduino
    Objetivo: <300 KB en TFLite
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloque 1 - reducido
        layers.Conv1D(16, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Bloque 2 - reducido
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Global pooling
        layers.GlobalAveragePooling1D(),
        
        # Capa densa pequeña
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ], name='Compact_CNN1D')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("\n🏗️  Construyendo modelo compacto...")
model = build_compact_cnn(input_shape, num_classes)
model.summary()

# Contar parámetros
total_params = model.count_params()
print(f"\n📊 Parámetros totales: {total_params:,}")
print(f"   Tamaño estimado: ~{total_params * 4 / 1024:.1f} KB")

# ============================================
# ENTRENAMIENTO
# ============================================

print("\n🚀 Entrenando modelo...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=4,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# EVALUACIÓN
# ============================================

print("\n📊 Evaluando modelo...")

from sklearn.metrics import accuracy_score, classification_report

y_pred_test = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\nTest Accuracy: {acc_test*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['clockwise', 'horizontal_swipe', 
                                      'forward_thrust', 'vertical_updown', 'wrist_twist']))

# ============================================
# GUARDAR MODELO
# ============================================

print("\n💾 Guardando modelo...")
model_path = os.path.join(MODELS_DIR, 'CNN1D_compact_best.keras')
model.save(model_path)
print(f"✅ Modelo guardado: {model_path}")

# ============================================
# CONVERTIR A TFLITE
# ============================================

print("\n⚙️  Convirtiendo a TensorFlow Lite...")

# Convertir con optimizaciones agresivas
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Cuantización para reducir tamaño
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1].astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

# Guardar
tflite_path = os.path.join('arduino_code', 'CNN1D_compact_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"✅ Modelo TFLite guardado: {tflite_path}")
print(f"   Tamaño: {size_kb:.2f} KB")

if size_kb > 400:
    print(f"\n⚠️  ADVERTENCIA: El modelo aún es grande ({size_kb:.0f} KB)")
    print(f"   Podría no caber en Arduino (límite ~400 KB)")
else:
    print(f"\n✅ ¡Perfecto! El modelo cabe en Arduino")

# ============================================
# CREAR HEADER
# ============================================

print("\n📝 Generando header para Arduino...")

hex_array = ', '.join([f'0x{byte:02x}' for byte in tflite_model])

header_content = f"""/*
 * Compact CNN1D Model for Arduino
 * Size: {size_kb:.2f} KB
 * Test Accuracy: {acc_test*100:.2f}%
 */

#ifndef CNN1D_COMPACT_MODEL_H
#define CNN1D_COMPACT_MODEL_H

const unsigned int cnn1d_model_len = {len(tflite_model)};

alignas(8) const unsigned char cnn1d_model_data[] = {{
{hex_array}
}};

#endif
"""

header_path = os.path.join('arduino_code', 'cnn1d_model.h')
with open(header_path, 'w') as f:
    f.write(header_content)

print(f"✅ Header guardado: {header_path}")

# ============================================
# RESUMEN
# ============================================

print("\n" + "="*60)
print("📦 RESUMEN")
print("="*60)
print(f"Modelo: Compact CNN1D")
print(f"Parámetros: {total_params:,}")
print(f"Test Accuracy: {acc_test*100:.2f}%")
print(f"Tamaño TFLite: {size_kb:.2f} KB")
print(f"¿Cabe en Arduino? {'✅ SÍ' if size_kb <= 400 else '❌ NO (muy grande)'}")
print("="*60)

if size_kb <= 400:
    print("\n✅ ¡Listo! Usa el nuevo cnn1d_model.h con el sketch de Arduino")
else:
    print("\n⚠️  El modelo aún es grande. Opciones:")
    print("   1. Usar Decision Tree (más simple, funciona bien)")
    print("   2. Reducir más el modelo (menos capas/filtros)")