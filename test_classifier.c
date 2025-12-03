
// Script de prueba del clasificador
#include <stdio.h>
#include "gesture_classifier.h"

int main() {
    printf("=== PRUEBA DEL CLASIFICADOR ===\n\n");
    
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
        
        printf("Test %d: gyro=(%.2f, %.2f, %.2f) -> %s\n",
               i + 1,
               test_data[i][0],
               test_data[i][1],
               test_data[i][2],
               get_gesture_name(prediction));
    }
    
    return 0;
}
