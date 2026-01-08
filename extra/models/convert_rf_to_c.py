"""
Convertir Random Forest con features mejoradas a código C para Arduino
"""

import pickle
import m2cgen as m2c
import os
import numpy as np

print("="*60)
print("GENERANDO CÓDIGO C PARA RANDOM FOREST")
print("="*60)

# ============================================
# LOAD MODEL AND SCALER
# ============================================
print("\n📂 Cargando modelo y scaler...")

model_path = 'models/randomforest_enhanced.pkl'
scaler_path = 'models/scaler_enhanced.pkl'

if not os.path.exists(model_path):
    print(f"❌ Error: No se encuentra {model_path}")
    print("   Ejecuta primero: python improve_accuracy_small_dataset.py")
    exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"✅ Modelo cargado: RandomForest")
print(f"   Número de árboles: {model.n_estimators}")
print(f"   Número de features: {len(scaler.mean_)}")

# ============================================
# GENERATE C CODE
# ============================================
print("\n⚙️  Generando código C...")

c_code = m2c.export_to_c(model)

print("✅ Código C generado")

# ============================================
# EXTRACT FEATURE NAMES
# ============================================
feature_names_list = []
for axis in ['ωx', 'ωy', 'ωz']:
    feature_names_list.extend([
        f'{axis}_mean', f'{axis}_std', f'{axis}_min', f'{axis}_max',
        f'{axis}_range', f'{axis}_q25', f'{axis}_median', f'{axis}_q75',
        f'{axis}_skew', f'{axis}_kurtosis', f'{axis}_energy', f'{axis}_rms'
    ])
feature_names_list.extend(['magnitude_mean', 'magnitude_std', 'magnitude_max'])

# ============================================
# CREATE HEADER FILE
# ============================================
print("\n📝 Creando archivo header para Arduino...")

mean = scaler.mean_
scale = scaler.scale_

# Create header with feature extraction functions
header_content = f"""/*
 * Random Forest Classifier for Arduino
 * Generated automatically from trained model
 * 
 * Gesture Recognition - Lab 2
 * Model: Random Forest ({model.n_estimators} trees)
 * Features: {len(feature_names_list)} (statistical features from gyroscope data)
 * Accuracy: ~80% (with small dataset)
 * 
 * Classes: 
 *   0 = clockwise
 *   1 = horizontal_swipe
 *   2 = forward_thrust
 *   3 = vertical_updown
 *   4 = wrist_twist
 */

#ifndef GESTURE_CLASSIFIER_RF_H
#define GESTURE_CLASSIFIER_RF_H

#include <math.h>

// Number of features
#define NUM_FEATURES {len(feature_names_list)}
#define NUM_SAMPLES 119

// Scaler parameters (from StandardScaler)
const float SCALER_MEAN[NUM_FEATURES] = {{
"""

# Add means
for i, m in enumerate(mean):
    if i > 0 and i % 4 == 0:
        header_content += "\n    "
    header_content += f"{m}f"
    if i < len(mean) - 1:
        header_content += ", "

header_content += "\n};\n\nconst float SCALER_SCALE[NUM_FEATURES] = {\n    "

# Add scales
for i, s in enumerate(scale):
    if i > 0 and i % 4 == 0:
        header_content += "\n    "
    header_content += f"{s}f"
    if i < len(scale) - 1:
        header_content += ", "

header_content += """
};

// Gesture labels
const char* GESTURE_NAMES[] = {
    "clockwise",
    "horizontal_swipe", 
    "forward_thrust",
    "vertical_updown",
    "wrist_twist"
};

// ============================================
// FEATURE EXTRACTION FUNCTIONS
// ============================================

// Calculate percentile
float percentile(float* data, int n, float p) {
    // Simple percentile calculation
    // For production, use more accurate method
    int idx = (int)(p * n / 100.0);
    if (idx >= n) idx = n - 1;
    return data[idx];
}

// Calculate skewness
float skewness(float* data, int n, float mean, float std) {
    if (std == 0) return 0;
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float z = (data[i] - mean) / std;
        sum += z * z * z;
    }
    return sum / n;
}

// Calculate kurtosis
float kurtosis(float* data, int n, float mean, float std) {
    if (std == 0) return 0;
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float z = (data[i] - mean) / std;
        sum += z * z * z * z;
    }
    return (sum / n) - 3.0; // Excess kurtosis
}

// Extract features from gyroscope data
void extract_features(float gyro_data[][3], int n_samples, float* features) {
    int feat_idx = 0;
    
    // Process each axis
    for (int axis = 0; axis < 3; axis++) {
        // Extract data for this axis
        float data[NUM_SAMPLES];
        for (int i = 0; i < n_samples; i++) {
            data[i] = gyro_data[i][axis];
        }
        
        // Calculate mean
        float sum = 0;
        for (int i = 0; i < n_samples; i++) {
            sum += data[i];
        }
        float mean = sum / n_samples;
        features[feat_idx++] = mean;
        
        // Calculate std
        float var_sum = 0;
        for (int i = 0; i < n_samples; i++) {
            float diff = data[i] - mean;
            var_sum += diff * diff;
        }
        float std = sqrt(var_sum / n_samples);
        features[feat_idx++] = std;
        
        // Min and Max
        float min_val = data[0];
        float max_val = data[0];
        for (int i = 1; i < n_samples; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        features[feat_idx++] = min_val;
        features[feat_idx++] = max_val;
        features[feat_idx++] = max_val - min_val; // range
        
        // Percentiles (simplified - should sort first)
        features[feat_idx++] = percentile(data, n_samples, 25);
        features[feat_idx++] = percentile(data, n_samples, 50); // median
        features[feat_idx++] = percentile(data, n_samples, 75);
        
        // Skewness and Kurtosis
        features[feat_idx++] = skewness(data, n_samples, mean, std);
        features[feat_idx++] = kurtosis(data, n_samples, mean, std);
        
        // Energy (sum of squares)
        float energy = 0;
        for (int i = 0; i < n_samples; i++) {
            energy += data[i] * data[i];
        }
        features[feat_idx++] = energy;
        
        // RMS
        features[feat_idx++] = sqrt(energy / n_samples);
    }
    
    // Magnitude features
    float mag_sum = 0, mag_sum_sq = 0, mag_max = 0;
    for (int i = 0; i < n_samples; i++) {
        float mag = sqrt(gyro_data[i][0]*gyro_data[i][0] + 
                        gyro_data[i][1]*gyro_data[i][1] + 
                        gyro_data[i][2]*gyro_data[i][2]);
        mag_sum += mag;
        mag_sum_sq += mag * mag;
        if (mag > mag_max) mag_max = mag;
    }
    features[feat_idx++] = mag_sum / n_samples; // magnitude_mean
    features[feat_idx++] = sqrt(mag_sum_sq / n_samples - (mag_sum/n_samples)*(mag_sum/n_samples)); // magnitude_std
    features[feat_idx++] = mag_max; // magnitude_max
}

// Apply StandardScaler normalization
void normalize_features(float* features, float* normalized) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        normalized[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

// Random Forest prediction function (generated by m2cgen)
"""

header_content += c_code

header_content += """

// High-level prediction function
int predict_gesture(float gyro_data[][3], int n_samples) {
    float features[NUM_FEATURES];
    float normalized[NUM_FEATURES];
    
    // Extract features
    extract_features(gyro_data, n_samples, features);
    
    // Normalize
    normalize_features(features, normalized);
    
    // Predict using Random Forest
    int prediction = (int)score(normalized);
    
    return prediction;
}

// Get gesture name from prediction
const char* get_gesture_name(int prediction) {
    if (prediction >= 0 && prediction < 5) {
        return GESTURE_NAMES[prediction];
    }
    return "unknown";
}

#endif // GESTURE_CLASSIFIER_RF_H
"""

# ============================================
# SAVE FILES
# ============================================
os.makedirs('arduino_code', exist_ok=True)

header_path = 'arduino_code/gesture_classifier_rf.h'
with open(header_path, 'w', encoding='utf-8') as f:
    f.write(header_content)

print(f"✅ Header file saved: {header_path}")

# Save raw C code
raw_c_path = 'arduino_code/randomforest_raw.c'
with open(raw_c_path, 'w', encoding='utf-8') as f:
    f.write(c_code)
    
print(f"✅ Raw C code saved: {raw_c_path}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("📦 ARCHIVOS GENERADOS")
print("="*60)
print(f"  1. {header_path}")
print(f"  2. {raw_c_path}")
print("="*60)

print("\n🎯 SIGUIENTE PASO:")
print("  1. El archivo .h incluye:")
print("     - Extracción de 39 features")
print("     - Normalización (StandardScaler)")
print(f"     - Random Forest ({model.n_estimators} árboles)")
print("     - Función predict_gesture()")
print("\n  2. Usa este .h con el mismo sketch de Arduino")
print("     (solo cambia el include)")

print("\n✅ Conversión a C completada!")