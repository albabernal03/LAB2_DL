
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
