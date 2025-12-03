/*
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
  Serial.println("\nReady! Move the sensor to record a gesture.");
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
      Serial.println("\nPress any key to capture another gesture...");
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
  Serial.println("\nMaking prediction...");
  
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
