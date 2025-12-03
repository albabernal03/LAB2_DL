"""
Exportar Decision Tree a código C para Arduino
"""

import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Cargar el modelo
print("="*60)
print("EXPORTANDO DECISION TREE A CÓDIGO C PARA ARDUINO")
print("="*60)

with open('models/decision_tree_optimized.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler_dt_optimized.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"\n✅ Modelo cargado")
print(f"   Max depth: {model.get_depth()}")
print(f"   Número de nodos: {model.tree_.node_count}")
print(f"   Número de hojas: {model.get_n_leaves()}")

# Nombres de gestos
GESTURE_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

# ============================================
# GENERAR CÓDIGO C
# ============================================

c_code = """
/*
 * Decision Tree para clasificación de gestos
 * Generado automáticamente desde modelo entrenado
 * 
 * Uso:
 *   1. Calcular mean, std, median de gyro_x, gyro_y, gyro_z
 *   2. Normalizar con scaler
 *   3. Llamar a predict_gesture()
 */

#ifndef GESTURE_CLASSIFIER_H
#define GESTURE_CLASSIFIER_H

// Nombres de gestos
const char* GESTURE_NAMES[] = {
    "clockwise",
    "horizontal_swipe", 
    "forward_thrust",
    "vertical_updown",
    "wrist_twist"
};

// Parámetros del scaler (StandardScaler)
const float SCALER_MEAN[] = {""" + f"{scaler.mean_[0]:.6f}, {scaler.mean_[1]:.6f}, {scaler.mean_[2]:.6f}" + """};
const float SCALER_SCALE[] = {""" + f"{scaler.scale_[0]:.6f}, {scaler.scale_[1]:.6f}, {scaler.scale_[2]:.6f}" + """};

// Función para normalizar features
void normalize_features(float* features, int n_features) {
    for (int i = 0; i < n_features; i++) {
        features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

"""

# Función recursiva para generar el árbol
def generate_tree_code(tree, node_id=0, depth=0):
    indent = "    " * depth
    
    # Si es hoja, retornar la clase
    if tree.children_left[node_id] == tree.children_right[node_id]:
        class_id = np.argmax(tree.value[node_id][0])
        return f"{indent}return {class_id}; // {GESTURE_NAMES[class_id]}\n"
    
    # Nodo interno
    feature = tree.feature[node_id]
    threshold = tree.threshold[node_id]
    
    feature_names = ['gyro_x', 'gyro_y', 'gyro_z']
    
    code = f"{indent}if (features[{feature}] <= {threshold:.6f}f) {{ // {feature_names[feature]}\n"
    code += generate_tree_code(tree, tree.children_left[node_id], depth + 1)
    code += f"{indent}}} else {{\n"
    code += generate_tree_code(tree, tree.children_right[node_id], depth + 1)
    code += f"{indent}}}\n"
    
    return code

# Generar función de predicción
c_code += """
// Función de predicción del árbol de decisión
// Input: features[3] = {gyro_x, gyro_y, gyro_z} (ya normalizados)
// Output: clase predicha (0-4)
int predict_gesture(float* features) {
"""

c_code += generate_tree_code(model.tree_)

c_code += """}

// Función completa: desde features raw hasta predicción
int classify_gesture(float gyro_x, float gyro_y, float gyro_z) {
    float features[3] = {gyro_x, gyro_y, gyro_z};
    normalize_features(features, 3);
    return predict_gesture(features);
}

// Función para obtener el nombre del gesto
const char* get_gesture_name(int gesture_id) {
    if (gesture_id >= 0 && gesture_id < 5) {
        return GESTURE_NAMES[gesture_id];
    }
    return "unknown";
}

#endif // GESTURE_CLASSIFIER_H
"""

# Guardar código C
output_file = 'gesture_classifier.h'
with open(output_file, 'w') as f:
    f.write(c_code)

print(f"\n✅ Código C generado: {output_file}")
print(f"\n📊 Estadísticas del código:")
print(f"   Líneas de código: {len(c_code.splitlines())}")
print(f"   Tamaño: {len(c_code)} bytes")

# Generar ejemplo de uso
example_code = """
/*
 * EJEMPLO DE USO EN ARDUINO
 */

#include "gesture_classifier.h"

void setup() {
    Serial.begin(115200);
}

void loop() {
    // 1. Leer datos del giroscopio
    float gyro_x = readGyroX(); // Implementar según tu sensor
    float gyro_y = readGyroY();
    float gyro_z = readGyroZ();
    
    // 2. Calcular estadísticas (mean, std, median)
    // Necesitas acumular datos durante el gesto
    float mean_x = calculateMean(gyro_x_buffer, buffer_size);
    float mean_y = calculateMean(gyro_y_buffer, buffer_size);
    float mean_z = calculateMean(gyro_z_buffer, buffer_size);
    
    // 3. Clasificar gesto
    int gesture_id = classify_gesture(mean_x, mean_y, mean_z);
    const char* gesture_name = get_gesture_name(gesture_id);
    
    // 4. Mostrar resultado
    Serial.print("Gesto detectado: ");
    Serial.println(gesture_name);
    
    delay(1000);
}
"""

with open('arduino_example.ino', 'w') as f:
    f.write(example_code)

print(f"✅ Ejemplo de Arduino generado: arduino_example.ino")

# Generar script de prueba
test_code = """
// Script de prueba del clasificador
#include <stdio.h>
#include "gesture_classifier.h"

int main() {
    printf("=== PRUEBA DEL CLASIFICADOR ===\\n\\n");
    
    // Datos de prueba (ejemplo)
    float test_data[][3] = {
        {1.4, 0.8, -1.5},   // Ejemplo clockwise
        {0.01, 0.04, 0.03}, // Ejemplo forward_thrust
        {-0.09, 0.07, 0.03} // Ejemplo wrist_twist
    };
    
    for (int i = 0; i < 3; i++) {
        int prediction = classify_gesture(
            test_data[i][0],
            test_data[i][1],
            test_data[i][2]
        );
        
        printf("Test %d: gyro=(%.2f, %.2f, %.2f) -> %s\\n",
               i + 1,
               test_data[i][0],
               test_data[i][1],
               test_data[i][2],
               get_gesture_name(prediction));
    }
    
    return 0;
}
"""

with open('test_classifier.c', 'w') as f:
    f.write(test_code)

print(f"✅ Script de prueba generado: test_classifier.c")

print(f"\n{'='*60}")
print(f"📁 ARCHIVOS GENERADOS:")
print(f"{'='*60}")
print(f"1. gesture_classifier.h  - Código del clasificador")
print(f"2. arduino_example.ino   - Ejemplo de uso en Arduino")
print(f"3. test_classifier.c     - Script de prueba")

print(f"\n💡 PRÓXIMOS PASOS:")
print(f"1. Copia gesture_classifier.h a tu proyecto Arduino")
print(f"2. Implementa la lectura del giroscopio")
print(f"3. Calcula estadísticas (mean, std, median) durante el gesto")
print(f"4. Llama a classify_gesture() con las estadísticas")

print(f"\n⚠️  IMPORTANTE:")
print(f"   El modelo usa ESTADÍSTICAS (mean, std, median) de cada eje")
print(f"   NO usa la secuencia temporal completa")
print(f"   Necesitas acumular ~100-200 muestras y calcular stats")
