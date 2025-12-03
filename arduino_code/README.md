# Arduino Gesture Recognition - Decision Tree

## Files Generated

1. **gesture_classifier.h** - Main header file for Arduino
   - Contains the decision tree model in C code
   - Includes StandardScaler normalization
   - Helper functions for prediction

2. **gesture_recognition_dt.ino** - Example Arduino sketch
   - Complete working example
   - Reads gyroscope data from LSM9DS1 IMU
   - Makes real-time gesture predictions

3. **model_raw.c** - Raw C code from m2cgen (for reference)

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

## Customization

You can modify the example sketch to:
- Change sampling parameters (SAMPLES_PER_GESTURE, SAMPLE_DELAY_MS)
- Use different feature aggregation (max, std, etc. instead of mean)
- Add confidence scores or multi-class probabilities
- Integrate with other sensors or actuators

## Hardware Requirements

- Arduino Nano 33 BLE Sense (recommended)
- Or any Arduino with LSM9DS1 or compatible IMU

