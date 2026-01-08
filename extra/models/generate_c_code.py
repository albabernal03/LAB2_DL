"""
Generate Arduino-compatible C code from trained Decision Tree model
Fixed version that handles Arduino compiler constraints
"""

import pickle
import m2cgen as m2c
import os
import re

print("="*60)
print("GENERATING ARDUINO-COMPATIBLE C CODE")
print("="*60)

# ============================================
# LOAD TRAINED MODEL
# ============================================
print("\n📂 Loading trained model...")

model_path = 'models/decision_tree_optimized.pkl'
scaler_path = 'models/scaler_dt_optimized.pkl'

if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}")
    print("Please run train_decision_tree_optimized.py first!")
    exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"✅ Model loaded successfully")
print(f"   Tree depth: {model.get_depth()}")
print(f"   Number of leaves: {model.get_n_leaves()}")

# ============================================
# GENERATE C CODE
# ============================================
print("\n⚙️  Generating C code...")

# Generate the C code for the model
c_code_raw = m2c.export_to_c(model)

print("✅ Raw C code generated")
print("🔧 Fixing Arduino compatibility issues...")

# ============================================
# FIX ARDUINO COMPATIBILITY
# ============================================

# 1. Replace double with float
c_code = c_code_raw.replace('double', 'float')

# 2. Fix the memcpy array initialization issues
# Pattern: memcpy(var0, (float[]){values}, size)
# Replace with proper array declaration and copy

def replace_memcpy(match):
    """Replace memcpy with array literal to proper C code"""
    indent = match.group(1)
    values = match.group(2)
    
    # Create static array declaration
    result = f"{indent}static const float temp[] = {{{values}}};\n"
    result += f"{indent}memcpy(var0, temp, 5 * sizeof(float));"
    
    return result

# Find all memcpy patterns and replace them
pattern = r'(\s+)memcpy\(var0, \(float\[\]\)\{([^}]+)\}, 5 \* sizeof\(float\)\);'
c_code = re.sub(pattern, replace_memcpy, c_code)

# 3. Fix function signature to use float*
c_code = re.sub(r'void score\(float\* input, float\* output\)', 
                'void score(float* input, float* output)', c_code)

print("✅ Code fixed for Arduino compatibility")

# ============================================
# GET SCALER PARAMETERS
# ============================================
mean = scaler.mean_
scale = scaler.scale_

# ============================================
# CREATE HEADER FILE
# ============================================
print("\n📝 Creating Arduino header file...")

header_content = f"""/*
 * Decision Tree Classifier for Arduino
 * Generated automatically from trained model
 * 
 * Gesture Recognition - Lab 2
 * Features: wx, wy, wz (gyroscope data)
 * Classes: 0=clockwise, 1=horizontal_swipe, 2=forward_thrust, 
 *          3=vertical_updown, 4=wrist_twist
 */

#ifndef GESTURE_CLASSIFIER_H
#define GESTURE_CLASSIFIER_H

#include <string.h>  // For memcpy

// Scaler parameters (from StandardScaler)
const float SCALER_MEAN[3] = {{{mean[0]}f, {mean[1]}f, {mean[2]}f}};
const float SCALER_SCALE[3] = {{{scale[0]}f, {scale[1]}f, {scale[2]}f}};

// Gesture labels
const char* GESTURE_NAMES[] = {{
    "clockwise",
    "horizontal_swipe", 
    "forward_thrust",
    "vertical_updown",
    "wrist_twist"
}};

// Apply StandardScaler normalization
void normalize_features(float* features, float* normalized) {{
    for (int i = 0; i < 3; i++) {{
        normalized[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }}
}}

// Decision tree prediction function (returns probabilities)
{c_code}

// High-level prediction function with normalization
int predict_gesture(float omega_x, float omega_y, float omega_z) {{
    float features[3] = {{omega_x, omega_y, omega_z}};
    float normalized[3];
    float probabilities[5];
    
    // Normalize input
    normalize_features(features, normalized);
    
    // Get probabilities from decision tree
    score(normalized, probabilities);
    
    // Find class with highest probability
    int prediction = 0;
    float max_prob = probabilities[0];
    for (int i = 1; i < 5; i++) {{
        if (probabilities[i] > max_prob) {{
            max_prob = probabilities[i];
            prediction = i;
        }}
    }}
    
    return prediction;
}}

// Get gesture name from prediction
const char* get_gesture_name(int prediction) {{
    if (prediction >= 0 && prediction < 5) {{
        return GESTURE_NAMES[prediction];
    }}
    return "unknown";
}}

#endif // GESTURE_CLASSIFIER_H
"""

# ============================================
# SAVE FILES
# ============================================
os.makedirs('arduino_code', exist_ok=True)

# Save the header file
header_path = 'arduino_code/gesture_classifier.h'
with open(header_path, 'w', encoding='utf-8') as f:
    f.write(header_content)

print(f"✅ Header file saved: {header_path}")

# Also save raw C code for reference
raw_c_path = 'arduino_code/model_raw_fixed.c'
with open(raw_c_path, 'w', encoding='utf-8') as f:
    f.write(c_code)
    
print(f"✅ Fixed C code saved: {raw_c_path}")

# ============================================
# CREATE EXAMPLE ARDUINO SKETCH
# ============================================
print("\n📝 Creating example Arduino sketch...")

sketch_content = """/*
 * Gesture Recognition on Arduino
 * Using Decision Tree Classifier
 * 
 * Hardware: Arduino Nano 33 BLE Sense (or compatible with gyroscope)
 * Sensors: LSM9DS1 IMU (gyroscope)
 */

#include <Arduino_LSM9DS1.h>
#include "gesture_classifier.h"

// Gesture detection parameters
const int SAMPLES_PER_GESTURE = 119;  // Same as training data
const int SAMPLE_DELAY_MS = 10;       // 100Hz sampling rate

// Data buffer for one gesture
float gyro_buffer[SAMPLES_PER_GESTURE][3];
int sample_count = 0;

// State machine
enum State {
  IDLE,
  SAMPLING,
  PREDICTING
};

State current_state = IDLE;
unsigned long last_sample_time = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  Serial.println("=================================");
  Serial.println("Gesture Recognition - Decision Tree");
  Serial.println("=================================");
  
  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  Serial.println("IMU initialized");
  Serial.println("\\nReady! Move the sensor to record a gesture.");
  Serial.println("Press any key to start sampling...");
  Serial.println();
}

void loop() {
  switch (current_state) {
    case IDLE:
      // Wait for user input to start sampling
      if (Serial.available() > 0) {
        Serial.read();  // Clear input
        start_sampling();
      }
      break;
      
    case SAMPLING:
      collect_sample();
      break;
      
    case PREDICTING:
      make_prediction();
      current_state = IDLE;
      Serial.println("\\nPress any key to capture another gesture...");
      break;
  }
}

void start_sampling() {
  Serial.println("Starting gesture capture...");
  Serial.println("Perform your gesture NOW!");
  
  sample_count = 0;
  current_state = SAMPLING;
  last_sample_time = millis();
}

void collect_sample() {
  unsigned long current_time = millis();
  
  // Sample at fixed rate
  if (current_time - last_sample_time >= SAMPLE_DELAY_MS) {
    if (IMU.gyroscopeAvailable()) {
      float gx, gy, gz;
      IMU.readGyroscope(gx, gy, gz);
      
      // Store in buffer
      gyro_buffer[sample_count][0] = gx;
      gyro_buffer[sample_count][1] = gy;
      gyro_buffer[sample_count][2] = gz;
      
      sample_count++;
      
      // Show progress
      if (sample_count % 20 == 0) {
        Serial.print(".");
      }
      
      // Check if we have enough samples
      if (sample_count >= SAMPLES_PER_GESTURE) {
        Serial.println(" Done!");
        current_state = PREDICTING;
      }
    }
    
    last_sample_time = current_time;
  }
}

void make_prediction() {
  Serial.println("\\nMaking prediction...");
  
  // Calculate mean values for the gesture
  float mean_gx = 0, mean_gy = 0, mean_gz = 0;
  
  for (int i = 0; i < SAMPLES_PER_GESTURE; i++) {
    mean_gx += gyro_buffer[i][0];
    mean_gy += gyro_buffer[i][1];
    mean_gz += gyro_buffer[i][2];
  }
  
  mean_gx /= SAMPLES_PER_GESTURE;
  mean_gy /= SAMPLES_PER_GESTURE;
  mean_gz /= SAMPLES_PER_GESTURE;
  
  // Make prediction
  int prediction = predict_gesture(mean_gx, mean_gy, mean_gz);
  const char* gesture_name = get_gesture_name(prediction);
  
  // Display results
  Serial.println("=================================");
  Serial.println("PREDICTION RESULTS");
  Serial.println("=================================");
  Serial.print("Gesture detected: ");
  Serial.println(gesture_name);
  Serial.print("Class ID: ");
  Serial.println(prediction);
  Serial.println("=================================");
  Serial.println();
  
  // Show input features (for debugging)
  Serial.println("Input features (mean values):");
  Serial.print("  wx: "); Serial.println(mean_gx, 4);
  Serial.print("  wy: "); Serial.println(mean_gy, 4);
  Serial.print("  wz: "); Serial.println(mean_gz, 4);
  Serial.println();
}
"""

sketch_path = 'arduino_code/gesture_recognition_dt/gesture_recognition_dt.ino'
os.makedirs('arduino_code/gesture_recognition_dt', exist_ok=True)

with open(sketch_path, 'w', encoding='utf-8') as f:
    f.write(sketch_content)

print(f"✅ Arduino sketch saved: {sketch_path}")

# ============================================
# CREATE README
# ============================================
readme_content = """# Arduino Gesture Recognition - Decision Tree

## Files Generated

1. **gesture_classifier.h** - Main header file for Arduino
   - Contains the decision tree model in C code
   - Includes StandardScaler normalization
   - Helper functions for prediction
   - **Fixed for Arduino compatibility** (uses float, not double)

2. **gesture_recognition_dt.ino** - Example Arduino sketch
   - Complete working example
   - Reads gyroscope data from LSM9DS1 IMU
   - Makes real-time gesture predictions

3. **model_raw_fixed.c** - Fixed C code from m2cgen (for reference)

## How to Use

### Step 1: Upload to Arduino

1. Open `gesture_recognition_dt/gesture_recognition_dt.ino` in Arduino IDE
2. Make sure `gesture_classifier.h` is in the same folder
3. Install required library: `Arduino_LSM9DS1`
4. Select your board (e.g., Arduino Nano 33 BLE Sense)
5. Upload the sketch

### Step 2: Test Gesture Recognition

1. Open Serial Monitor (115200 baud)
2. Press any key to start sampling
3. Perform a gesture within ~1.2 seconds
4. View the prediction result

## Gesture Classes

- 0: clockwise
- 1: horizontal_swipe
- 2: forward_thrust
- 3: vertical_updown
- 4: wrist_twist

## Model Information

- Algorithm: Decision Tree with optimized hyperparameters
- Features: 3 (wx, wy, wz from gyroscope)
- Normalization: StandardScaler (included in header file)
- Sampling: 119 samples per gesture at 100Hz

## Arduino Compatibility Fixes

This version includes fixes for Arduino compiler compatibility:
- Changed `double` to `float` (Arduino works better with float)
- Fixed array initialization in memcpy calls
- Proper function signatures

## Hardware Requirements

- Arduino Nano 33 BLE Sense (recommended)
- Or any Arduino with LSM9DS1 or compatible IMU

"""

readme_path = 'arduino_code/README.md'
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
    
print(f"✅ README saved: {readme_path}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("📦 FILES GENERATED")
print("="*60)
print(f"  1. {header_path}")
print(f"  2. {sketch_path}")
print(f"  3. {raw_c_path}")
print(f"  4. {readme_path}")
print("="*60)

print("\n🎯 NEXT STEPS:")
print("  1. Copy 'gesture_classifier.h' to your Arduino sketch folder")
print("  2. Open 'gesture_recognition_dt.ino' in Arduino IDE")
print("  3. Install Arduino_LSM9DS1 library")
print("  4. Upload to your Arduino board")
print("  5. Test with real gestures!")

print("\n✅ Arduino-compatible C code generation complete!")