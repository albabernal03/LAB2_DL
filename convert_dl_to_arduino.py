"""
Convertir modelos Deep Learning a TensorFlow Lite para Arduino
Convierte CNN1D y Hybrid models a formato compatible con Arduino
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("="*60)
print("CONVERSIÓN DE MODELOS A TENSORFLOW LITE")
print("="*60)

# ============================================
# CONFIGURACIÓN
# ============================================
MODELS_DIR = "models_dl"
ARDUINO_DIR = "arduino_code"
SEQUENCES_DIR = "sequences_processed"

os.makedirs(ARDUINO_DIR, exist_ok=True)

GESTURE_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

# ============================================
# CARGAR DATOS DE PRUEBA
# ============================================
print("\n📂 Cargando datos de prueba...")
X_test = np.load(os.path.join(SEQUENCES_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(SEQUENCES_DIR, 'y_test.npy'))

# Tomar una muestra para verificar
test_sample = X_test[0:1]
print(f"✅ Muestra de prueba shape: {test_sample.shape}")

# ============================================
# FUNCIÓN DE CONVERSIÓN
# ============================================

def convert_model_to_tflite(model_path, model_name, test_data):
    """
    Convierte un modelo Keras a TensorFlow Lite
    """
    print(f"\n{'='*60}")
    print(f"🔄 Convirtiendo: {model_name}")
    print(f"{'='*60}")
    
    # Cargar modelo
    print(f"📥 Cargando modelo desde: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✅ Modelo cargado")
    
    # Mostrar información del modelo
    print(f"\n📊 Información del modelo:")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Probar predicción con el modelo original
    print(f"\n🧪 Probando modelo original...")
    original_prediction = model.predict(test_data, verbose=0)
    original_class = np.argmax(original_prediction[0])
    print(f"   Predicción: clase {original_class} ({GESTURE_NAMES[original_class]})")
    print(f"   Confianza: {original_prediction[0][original_class]*100:.2f}%")
    
    # Convertir a TensorFlow Lite
    print(f"\n⚙️  Convirtiendo a TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimizaciones para Arduino
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # Convertir
    tflite_model = converter.convert()
    
    # Guardar modelo .tflite
    tflite_path = os.path.join(ARDUINO_DIR, f'{model_name}_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Modelo TFLite guardado: {tflite_path}")
    print(f"   Tamaño: {len(tflite_model) / 1024:.2f} KB")
    
    # Verificar modelo TFLite
    print(f"\n🧪 Probando modelo TFLite...")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Hacer predicción con TFLite
    interpreter.set_tensor(input_details[0]['index'], test_data.astype(np.float32))
    interpreter.invoke()
    tflite_prediction = interpreter.get_tensor(output_details[0]['index'])
    tflite_class = np.argmax(tflite_prediction[0])
    
    print(f"   Predicción: clase {tflite_class} ({GESTURE_NAMES[tflite_class]})")
    print(f"   Confianza: {tflite_prediction[0][tflite_class]*100:.2f}%")
    
    # Verificar que las predicciones coinciden
    if original_class == tflite_class:
        print(f"   ✅ Predicciones coinciden!")
    else:
        print(f"   ⚠️  Las predicciones difieren (original: {original_class}, tflite: {tflite_class})")
    
    return tflite_path, len(tflite_model)

# ============================================
# CONVERTIR MODELOS
# ============================================

models_to_convert = [
    ('CNN1D_best.keras', 'CNN1D'),
    ('Hybrid_best.keras', 'Hybrid')
]

converted_models = []

for model_file, model_name in models_to_convert:
    model_path = os.path.join(MODELS_DIR, model_file)
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Modelo no encontrado: {model_path}")
        print(f"   Saltando {model_name}...")
        continue
    
    try:
        tflite_path, size = convert_model_to_tflite(model_path, model_name, test_sample)
        converted_models.append({
            'name': model_name,
            'path': tflite_path,
            'size': size
        })
    except Exception as e:
        print(f"\n❌ Error al convertir {model_name}: {str(e)}")
        continue

# ============================================
# CONVERTIR A FORMATO C HEADER
# ============================================

print("\n" + "="*60)
print("📝 GENERANDO ARCHIVOS HEADER PARA ARDUINO")
print("="*60)

def tflite_to_c_array(tflite_path, model_name):
    """
    Convierte un archivo .tflite a un array de C
    """
    print(f"\n🔄 Convirtiendo {model_name} a header file...")
    
    # Leer archivo tflite
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    # Crear array de C
    hex_array = ', '.join([f'0x{byte:02x}' for byte in tflite_data])
    
    # Crear contenido del header
    header_content = f"""/*
 * {model_name} Model for Arduino
 * Generated automatically from TensorFlow Lite model
 * 
 * Model size: {len(tflite_data)} bytes ({len(tflite_data)/1024:.2f} KB)
 * Input shape: (119, 3) - 119 timesteps, 3 features (wx, wy, wz)
 * Output shape: (5,) - 5 gesture classes
 * 
 * Gesture classes:
 * 0: clockwise
 * 1: horizontal_swipe
 * 2: forward_thrust
 * 3: vertical_updown
 * 4: wrist_twist
 */

#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H

const unsigned int {model_name.lower()}_model_len = {len(tflite_data)};

// Model data
alignas(8) const unsigned char {model_name.lower()}_model_data[] = {{
{hex_array}
}};

#endif  // {model_name.upper()}_MODEL_H
"""
    
    # Guardar header file
    header_path = os.path.join(ARDUINO_DIR, f'{model_name.lower()}_model.h')
    with open(header_path, 'w') as f:
        f.write(header_content)
    
    print(f"✅ Header guardado: {header_path}")
    return header_path

# Generar headers para todos los modelos convertidos
header_files = []
for model_info in converted_models:
    header_path = tflite_to_c_array(model_info['path'], model_info['name'])
    header_files.append(header_path)

# ============================================
# CREAR README CON INSTRUCCIONES
# ============================================

readme_content = f"""# Deep Learning Models for Arduino

## Modelos Convertidos

"""

for model_info in converted_models:
    readme_content += f"""
### {model_info['name']}
- Archivo TFLite: `{os.path.basename(model_info['path'])}`
- Header file: `{model_info['name'].lower()}_model.h`
- Tamaño: {model_info['size']/1024:.2f} KB
"""

readme_content += """

## Cómo usar en Arduino

### Requisitos
1. **Arduino Nano 33 BLE Sense Rev2** (con sensor BMI270)
2. **Librería TensorFlow Lite**: Instalar `Arduino_TensorFlowLite` desde Library Manager

### Pasos siguientes

1. Los archivos `.h` generados contienen el modelo en formato de array de C
2. Necesitas crear el sketch de Arduino que:
   - Incluya el header del modelo
   - Inicialice el intérprete de TensorFlow Lite
   - Capture datos del giroscopio
   - Haga inferencia con el modelo

### Estructura de datos

**Input:**
- Shape: (1, 119, 3)
- 119 timesteps de datos del giroscopio
- 3 features: wx, wy, wz (gyroscope x, y, z)

**Output:**
- Shape: (1, 5)
- 5 probabilidades para cada clase de gesto
- Clase predicha = argmax de las probabilidades

### Gestos
0. clockwise
1. horizontal_swipe
2. forward_thrust
3. vertical_updown
4. wrist_twist

## Archivos generados

"""

for header in header_files:
    readme_content += f"- `{os.path.basename(header)}`\n"

readme_path = os.path.join(ARDUINO_DIR, 'README_DL.md')
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"\n✅ README guardado: {readme_path}")

# ============================================
# RESUMEN FINAL
# ============================================

print("\n" + "="*60)
print("📦 RESUMEN DE CONVERSIÓN")
print("="*60)

if len(converted_models) > 0:
    print(f"\n✅ Modelos convertidos exitosamente: {len(converted_models)}")
    for model_info in converted_models:
        print(f"   • {model_info['name']}: {model_info['size']/1024:.2f} KB")
    
    print(f"\n📁 Archivos generados en: {ARDUINO_DIR}/")
    print(f"   - Modelos TFLite: *.tflite")
    print(f"   - Headers para Arduino: *_model.h")
    print(f"   - Instrucciones: README_DL.md")
    
    print("\n🎯 SIGUIENTE PASO:")
    print("   Crear el sketch de Arduino que use estos modelos")
    print("   (necesitarás instalar Arduino_TensorFlowLite)")
else:
    print("\n❌ No se pudo convertir ningún modelo")
    print("   Verifica que los archivos .keras existen en models_dl/")

print("\n" + "="*60)
print("✅ CONVERSIÓN COMPLETADA")
print("="*60)