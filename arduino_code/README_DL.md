# Deep Learning Models for Arduino

## Modelos Convertidos


### CNN1D
- Archivo TFLite: `CNN1D_model.tflite`
- Header file: `cnn1d_model.h`
- Tamaño: 725.67 KB


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

- `cnn1d_model.h`
