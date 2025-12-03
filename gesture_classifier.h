
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
const float SCALER_MEAN[] = {0.258406, 0.173197, -0.298782};
const float SCALER_SCALE[] = {1.952963, 2.143302, 3.173224};

// Función para normalizar features
void normalize_features(float* features, int n_features) {
    for (int i = 0; i < n_features; i++) {
        features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}


// Función de predicción del árbol de decisión
// Input: features[3] = {gyro_x, gyro_y, gyro_z} (ya normalizados)
// Output: clase predicha (0-4)
int predict_gesture(float* features) {
if (features[0] <= 0.733057f) { // gyro_x
    if (features[2] <= -0.645902f) { // gyro_z
        if (features[1] <= -0.403623f) { // gyro_y
            if (features[0] <= -0.837595f) { // gyro_x
                if (features[2] <= -1.040186f) { // gyro_z
                    if (features[1] <= -0.688664f) { // gyro_y
                        if (features[1] <= -2.349455f) { // gyro_y
                            return 0; // clockwise
                        } else {
                            if (features[0] <= -1.187507f) { // gyro_x
                                if (features[0] <= -1.414146f) { // gyro_x
                                    return 0; // clockwise
                                } else {
                                    if (features[0] <= -1.376237f) { // gyro_x
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            } else {
                                if (features[0] <= -0.958960f) { // gyro_x
                                    if (features[0] <= -1.011869f) { // gyro_x
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        }
                    } else {
                        if (features[0] <= -1.841787f) { // gyro_x
                            return 0; // clockwise
                        } else {
                            if (features[0] <= -0.963323f) { // gyro_x
                                if (features[2] <= -1.883138f) { // gyro_z
                                    return 0; // clockwise
                                } else {
                                    if (features[1] <= -0.584786f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[2] <= -1.441688f) { // gyro_z
                                    return 0; // clockwise
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= -2.040309f) { // gyro_y
                        if (features[1] <= -2.562429f) { // gyro_y
                            return 2; // forward_thrust
                        } else {
                            if (features[0] <= -1.410055f) { // gyro_x
                                return 2; // forward_thrust
                            } else {
                                return 3; // vertical_updown
                            }
                        }
                    } else {
                        if (features[1] <= -0.635731f) { // gyro_y
                            if (features[2] <= -0.899190f) { // gyro_z
                                if (features[2] <= -0.927222f) { // gyro_z
                                    return 0; // clockwise
                                } else {
                                    return 0; // clockwise
                                }
                            } else {
                                if (features[2] <= -0.848331f) { // gyro_z
                                    return 3; // vertical_updown
                                } else {
                                    if (features[1] <= -1.313169f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            }
                        } else {
                            return 1; // horizontal_swipe
                        }
                    }
                }
            } else {
                if (features[1] <= -0.815155f) { // gyro_y
                    if (features[0] <= 0.116142f) { // gyro_x
                        if (features[2] <= -1.174468f) { // gyro_z
                            if (features[2] <= -2.130552f) { // gyro_z
                                return 0; // clockwise
                            } else {
                                if (features[1] <= -2.009742f) { // gyro_y
                                    if (features[2] <= -1.631528f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= -0.234862f) { // gyro_x
                                        return 0; // clockwise
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        } else {
                            if (features[0] <= -0.264044f) { // gyro_x
                                if (features[0] <= -0.594320f) { // gyro_x
                                    if (features[2] <= -0.926550f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[1] <= -1.048257f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[1] <= -2.295777f) { // gyro_y
                                    if (features[1] <= -2.535092f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[2] <= -1.011483f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[1] <= -1.277384f) { // gyro_y
                            if (features[2] <= -0.954581f) { // gyro_z
                                if (features[2] <= -1.619611f) { // gyro_z
                                    if (features[1] <= -1.624800f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    if (features[2] <= -1.092724f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            } else {
                                if (features[0] <= 0.500691f) { // gyro_x
                                    if (features[1] <= -1.531858f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[1] <= -1.711033f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= -0.922014f) { // gyro_y
                                if (features[2] <= -0.938636f) { // gyro_z
                                    if (features[0] <= 0.392145f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    return 3; // vertical_updown
                                }
                            } else {
                                if (features[2] <= -1.431449f) { // gyro_z
                                    return 3; // vertical_updown
                                } else {
                                    return 3; // vertical_updown
                                }
                            }
                        }
                    }
                } else {
                    if (features[2] <= -1.258561f) { // gyro_z
                        if (features[0] <= -0.505410f) { // gyro_x
                            if (features[2] <= -1.460487f) { // gyro_z
                                if (features[2] <= -1.596783f) { // gyro_z
                                    return 0; // clockwise
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            if (features[0] <= 0.197416f) { // gyro_x
                                if (features[2] <= -1.745164f) { // gyro_z
                                    if (features[2] <= -1.886663f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= 0.003777f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[0] <= 0.386417f) { // gyro_x
                                    if (features[1] <= -0.551486f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    if (features[0] <= 0.459782f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[0] <= -0.334681f) { // gyro_x
                            if (features[2] <= -0.855213f) { // gyro_z
                                if (features[0] <= -0.634138f) { // gyro_x
                                    return 0; // clockwise
                                } else {
                                    if (features[0] <= -0.468591f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                return 3; // vertical_updown
                            }
                        } else {
                            if (features[2] <= -0.785555f) { // gyro_z
                                if (features[1] <= -0.477182f) { // gyro_y
                                    if (features[2] <= -0.892476f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= 0.087778f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            } else {
                                if (features[2] <= -0.714721f) { // gyro_z
                                    if (features[1] <= -0.616844f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    return 3; // vertical_updown
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (features[0] <= -0.820140f) { // gyro_x
                if (features[0] <= -1.239326f) { // gyro_x
                    if (features[1] <= -0.128522f) { // gyro_y
                        if (features[2] <= -1.311938f) { // gyro_z
                            if (features[0] <= -1.580511f) { // gyro_x
                                return 0; // clockwise
                            } else {
                                if (features[1] <= -0.250044f) { // gyro_y
                                    return 0; // clockwise
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        } else {
                            if (features[2] <= -1.070231f) { // gyro_z
                                return 0; // clockwise
                            } else {
                                return 1; // horizontal_swipe
                            }
                        }
                    } else {
                        if (features[0] <= -1.428328f) { // gyro_x
                            if (features[1] <= 0.136638f) { // gyro_y
                                if (features[0] <= -1.794877f) { // gyro_x
                                    return 0; // clockwise
                                } else {
                                    if (features[2] <= -1.410971f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            if (features[2] <= -1.727540f) { // gyro_z
                                return 0; // clockwise
                            } else {
                                if (features[1] <= 0.515367f) { // gyro_y
                                    if (features[2] <= -1.202331f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= 0.223368f) { // gyro_y
                        if (features[2] <= -0.987648f) { // gyro_z
                            if (features[2] <= -1.418524f) { // gyro_z
                                if (features[0] <= -0.914505f) { // gyro_x
                                    if (features[2] <= -1.663085f) { // gyro_z
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            } else {
                                if (features[0] <= -1.035870f) { // gyro_x
                                    if (features[0] <= -1.078415f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    if (features[2] <= -1.235398f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= -0.100441f) { // gyro_y
                                if (features[1] <= -0.227429f) { // gyro_y
                                    return 1; // horizontal_swipe
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            } else {
                                if (features[0] <= -0.999869f) { // gyro_x
                                    return 1; // horizontal_swipe
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            }
                        }
                    } else {
                        if (features[2] <= -1.710754f) { // gyro_z
                            if (features[2] <= -1.887502f) { // gyro_z
                                return 1; // horizontal_swipe
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            if (features[1] <= 0.824265f) { // gyro_y
                                if (features[0] <= -0.909596f) { // gyro_x
                                    if (features[1] <= 0.375207f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    if (features[2] <= -1.251008f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            } else {
                                return 0; // clockwise
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 0.186837f) { // gyro_y
                    if (features[0] <= -0.241680f) { // gyro_x
                        if (features[1] <= -0.074099f) { // gyro_y
                            if (features[0] <= -0.474591f) { // gyro_x
                                if (features[2] <= -0.883748f) { // gyro_z
                                    if (features[1] <= -0.208045f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[0] <= -0.697957f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            } else {
                                if (features[2] <= -1.277529f) { // gyro_z
                                    if (features[1] <= -0.208045f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[2] <= -0.883580f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            }
                        } else {
                            if (features[2] <= -1.340137f) { // gyro_z
                                if (features[1] <= -0.020917f) { // gyro_y
                                    if (features[0] <= -0.362499f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[1] <= 0.030276f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[0] <= -0.427409f) { // gyro_x
                                    if (features[2] <= -0.751984f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[1] <= -0.010728f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[2] <= -1.405096f) { // gyro_z
                            if (features[1] <= 0.034003f) { // gyro_y
                                if (features[2] <= -1.573452f) { // gyro_z
                                    if (features[1] <= -0.226435f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[0] <= -0.180861f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[2] <= -1.698669f) { // gyro_z
                                    if (features[1] <= 0.122473f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[0] <= -0.125497f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[0] <= 0.029141f) { // gyro_x
                                if (features[1] <= -0.134238f) { // gyro_y
                                    if (features[2] <= -1.079463f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[1] <= 0.014620f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[2] <= -1.269640f) { // gyro_z
                                    if (features[1] <= -0.015202f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                } else {
                                    if (features[0] <= 0.118869f) { // gyro_x
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (features[2] <= -1.565227f) { // gyro_z
                        if (features[2] <= -1.900931f) { // gyro_z
                            if (features[0] <= -0.639320f) { // gyro_x
                                if (features[0] <= -0.697139f) { // gyro_x
                                    return 1; // horizontal_swipe
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            } else {
                                if (features[1] <= 0.550158f) { // gyro_y
                                    if (features[0] <= 0.008686f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            }
                        } else {
                            if (features[0] <= -0.057314f) { // gyro_x
                                if (features[0] <= -0.439955f) { // gyro_x
                                    if (features[1] <= 0.388627f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[1] <= 0.413478f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[1] <= 0.777296f) { // gyro_y
                                    if (features[2] <= -1.634046f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[1] <= 1.173421f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[1] <= 0.887883f) { // gyro_y
                            if (features[0] <= 0.085596f) { // gyro_x
                                if (features[0] <= -0.500773f) { // gyro_x
                                    if (features[1] <= 0.742754f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 0; // clockwise
                                    }
                                } else {
                                    if (features[1] <= 0.524810f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[1] <= 0.580974f) { // gyro_y
                                    if (features[1] <= 0.428637f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[0] <= 0.199325f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[2] <= -1.090542f) { // gyro_z
                                if (features[1] <= 1.206722f) { // gyro_y
                                    if (features[0] <= -0.451682f) { // gyro_x
                                        return 0; // clockwise
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    return 0; // clockwise
                                }
                            } else {
                                if (features[0] <= 0.095960f) { // gyro_x
                                    return 0; // clockwise
                                } else {
                                    return 3; // vertical_updown
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (features[2] <= 0.560618f) { // gyro_z
            if (features[1] <= 1.172676f) { // gyro_y
                if (features[2] <= 0.042962f) { // gyro_z
                    if (features[1] <= -0.672510f) { // gyro_y
                        if (features[0] <= -0.648320f) { // gyro_x
                            if (features[2] <= -0.472846f) { // gyro_z
                                if (features[0] <= -1.068870f) { // gyro_x
                                    if (features[1] <= -1.566898f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[0] <= -0.802140f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[1] <= -0.955315f) { // gyro_y
                                    if (features[0] <= -1.207689f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= -0.751288f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= -0.971716f) { // gyro_y
                                if (features[0] <= -0.022405f) { // gyro_x
                                    if (features[2] <= -0.324969f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= 0.467691f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            } else {
                                if (features[2] <= -0.248428f) { // gyro_z
                                    if (features[0] <= -0.054860f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= -0.420863f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[0] <= -0.439682f) { // gyro_x
                            if (features[2] <= -0.268067f) { // gyro_z
                                if (features[1] <= 0.373965f) { // gyro_y
                                    if (features[1] <= -0.379517f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[1] <= 0.606073f) { // gyro_y
                                        return 0; // clockwise
                                    } else {
                                        return 0; // clockwise
                                    }
                                }
                            } else {
                                if (features[0] <= -0.811958f) { // gyro_x
                                    if (features[0] <= -1.087961f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= -0.230660f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[2] <= -0.143689f) { // gyro_z
                                if (features[1] <= 0.099858f) { // gyro_y
                                    if (features[1] <= -0.165799f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                } else {
                                    if (features[2] <= -0.351657f) { // gyro_z
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            } else {
                                if (features[0] <= 0.208870f) { // gyro_x
                                    if (features[1] <= -0.290799f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                } else {
                                    if (features[1] <= -0.064655f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= 0.118745f) { // gyro_y
                        if (features[0] <= 0.149688f) { // gyro_x
                            if (features[0] <= -0.397954f) { // gyro_x
                                if (features[0] <= -1.247235f) { // gyro_x
                                    if (features[1] <= -0.708296f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= -0.127777f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            } else {
                                if (features[2] <= 0.310183f) { // gyro_z
                                    if (features[1] <= -0.042786f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= -0.442142f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= -0.179964f) { // gyro_y
                                if (features[2] <= 0.249085f) { // gyro_z
                                    if (features[1] <= -0.228175f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= -0.334537f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            } else {
                                if (features[1] <= 0.055624f) { // gyro_y
                                    if (features[0] <= 0.218688f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= 0.085445f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[0] <= 0.113960f) { // gyro_x
                            if (features[2] <= 0.346439f) { // gyro_z
                                if (features[1] <= 0.588180f) { // gyro_y
                                    if (features[0] <= -0.191225f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[0] <= -0.297862f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[0] <= -0.628684f) { // gyro_x
                                    if (features[2] <= 0.441107f) { // gyro_z
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= 0.239024f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        } else {
                            if (features[2] <= 0.348453f) { // gyro_z
                                if (features[1] <= 0.497474f) { // gyro_y
                                    if (features[0] <= 0.420236f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[2] <= 0.131588f) { // gyro_z
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            } else {
                                if (features[0] <= 0.352872f) { // gyro_x
                                    if (features[2] <= 0.475853f) { // gyro_z
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[1] <= 0.621481f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 1.616515f) { // gyro_y
                    if (features[2] <= 0.097514f) { // gyro_z
                        if (features[0] <= -0.222316f) { // gyro_x
                            if (features[0] <= -2.398429f) { // gyro_x
                                return 0; // clockwise
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            return 0; // clockwise
                        }
                    } else {
                        if (features[0] <= -0.640957f) { // gyro_x
                            return 0; // clockwise
                        } else {
                            if (features[0] <= 0.203143f) { // gyro_x
                                if (features[0] <= -0.305226f) { // gyro_x
                                    return 3; // vertical_updown
                                } else {
                                    if (features[2] <= 0.358692f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                return 3; // vertical_updown
                            }
                        }
                    }
                } else {
                    if (features[2] <= 0.436407f) { // gyro_z
                        if (features[0] <= 0.666784f) { // gyro_x
                            return 0; // clockwise
                        } else {
                            return 0; // clockwise
                        }
                    } else {
                        if (features[1] <= 2.049170f) { // gyro_y
                            return 3; // vertical_updown
                        } else {
                            return 0; // clockwise
                        }
                    }
                }
            }
        } else {
            if (features[1] <= -0.191395f) { // gyro_y
                if (features[0] <= -0.134497f) { // gyro_x
                    if (features[1] <= -0.423503f) { // gyro_y
                        if (features[2] <= 1.961678f) { // gyro_z
                            if (features[0] <= -0.809231f) { // gyro_x
                                if (features[0] <= -1.660421f) { // gyro_x
                                    return 2; // forward_thrust
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            } else {
                                if (features[0] <= -0.351863f) { // gyro_x
                                    if (features[2] <= 0.916800f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[1] <= -0.669031f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[0] <= -0.586138f) { // gyro_x
                                if (features[1] <= -0.599449f) { // gyro_y
                                    if (features[1] <= -0.969480f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    return 4; // wrist_twist
                                }
                            } else {
                                if (features[1] <= -0.553226f) { // gyro_y
                                    if (features[1] <= -0.649896f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[2] <= 2.613615f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[2] <= 1.629163f) { // gyro_z
                            if (features[0] <= -0.760412f) { // gyro_x
                                return 4; // wrist_twist
                            } else {
                                if (features[2] <= 1.300509f) { // gyro_z
                                    if (features[2] <= 1.016000f) { // gyro_z
                                        return 4; // wrist_twist
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= -0.286680f) { // gyro_x
                                        return 4; // wrist_twist
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[0] <= -0.271953f) { // gyro_x
                                if (features[0] <= -0.639047f) { // gyro_x
                                    return 4; // wrist_twist
                                } else {
                                    if (features[1] <= -0.324597f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            } else {
                                if (features[0] <= -0.210316f) { // gyro_x
                                    return 1; // horizontal_swipe
                                } else {
                                    if (features[1] <= -0.368086f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (features[2] <= 1.317294f) { // gyro_z
                        if (features[1] <= -0.498305f) { // gyro_y
                            if (features[1] <= -1.054222f) { // gyro_y
                                return 1; // horizontal_swipe
                            } else {
                                if (features[1] <= -0.641695f) { // gyro_y
                                    return 1; // horizontal_swipe
                                } else {
                                    if (features[0] <= 0.244325f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            }
                        } else {
                            if (features[0] <= 0.008686f) { // gyro_x
                                if (features[2] <= 0.768419f) { // gyro_z
                                    return 1; // horizontal_swipe
                                } else {
                                    if (features[0] <= -0.078587f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[0] <= 0.264780f) { // gyro_x
                                    if (features[1] <= -0.211276f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            }
                        }
                    } else {
                        if (features[1] <= -0.609886f) { // gyro_y
                            if (features[1] <= -0.790801f) { // gyro_y
                                if (features[1] <= -0.806706f) { // gyro_y
                                    return 1; // horizontal_swipe
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            } else {
                                return 1; // horizontal_swipe
                            }
                        } else {
                            if (features[2] <= 1.568569f) { // gyro_z
                                if (features[2] <= 1.413138f) { // gyro_z
                                    return 1; // horizontal_swipe
                                } else {
                                    if (features[1] <= -0.221713f) { // gyro_y
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[2] <= 2.365530f) { // gyro_z
                                    if (features[0] <= 0.368144f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    return 1; // horizontal_swipe
                                }
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 0.610298f) { // gyro_y
                    if (features[0] <= -0.235134f) { // gyro_x
                        if (features[0] <= -0.583410f) { // gyro_x
                            if (features[2] <= 0.955573f) { // gyro_z
                                if (features[1] <= 0.196529f) { // gyro_y
                                    if (features[1] <= 0.009898f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                } else {
                                    return 3; // vertical_updown
                                }
                            } else {
                                if (features[0] <= -0.678866f) { // gyro_x
                                    if (features[2] <= 2.406653f) { // gyro_z
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                } else {
                                    if (features[2] <= 1.473732f) { // gyro_z
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            }
                        } else {
                            if (features[2] <= 1.649809f) { // gyro_z
                                if (features[2] <= 0.804675f) { // gyro_z
                                    if (features[1] <= 0.207215f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= -0.350499f) { // gyro_x
                                        return 4; // wrist_twist
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[2] <= 1.857442f) { // gyro_z
                                    if (features[1] <= 0.229084f) { // gyro_y
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                } else {
                                    if (features[0] <= -0.413772f) { // gyro_x
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[0] <= 0.112869f) { // gyro_x
                            if (features[1] <= 0.164720f) { // gyro_y
                                if (features[2] <= 1.481789f) { // gyro_z
                                    if (features[0] <= -0.014768f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= -0.035223f) { // gyro_x
                                        return 4; // wrist_twist
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[2] <= 1.556987f) { // gyro_z
                                    if (features[2] <= 1.075755f) { // gyro_z
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[0] <= -0.000586f) { // gyro_x
                                        return 4; // wrist_twist
                                    } else {
                                        return 4; // wrist_twist
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= 0.295933f) { // gyro_y
                                if (features[2] <= 0.791414f) { // gyro_z
                                    if (features[0] <= 0.220870f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                } else {
                                    if (features[0] <= 0.352053f) { // gyro_x
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 1; // horizontal_swipe
                                    }
                                }
                            } else {
                                if (features[2] <= 0.865605f) { // gyro_z
                                    if (features[0] <= 0.382054f) { // gyro_x
                                        return 3; // vertical_updown
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[2] <= 1.185698f) { // gyro_z
                                        return 1; // horizontal_swipe
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= 2.190821f) { // gyro_y
                        if (features[2] <= 1.685897f) { // gyro_z
                            if (features[0] <= 0.282235f) { // gyro_x
                                if (features[0] <= -0.699866f) { // gyro_x
                                    return 4; // wrist_twist
                                } else {
                                    if (features[1] <= 1.022327f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            } else {
                                if (features[2] <= 0.784868f) { // gyro_z
                                    if (features[1] <= 0.927148f) { // gyro_y
                                        return 2; // forward_thrust
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                } else {
                                    if (features[1] <= 0.878192f) { // gyro_y
                                        return 3; // vertical_updown
                                    } else {
                                        return 3; // vertical_updown
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= 1.255927f) { // gyro_y
                                if (features[0] <= -0.303590f) { // gyro_x
                                    return 4; // wrist_twist
                                } else {
                                    return 4; // wrist_twist
                                }
                            } else {
                                return 3; // vertical_updown
                            }
                        }
                    } else {
                        if (features[1] <= 2.389878f) { // gyro_y
                            if (features[2] <= 0.953391f) { // gyro_z
                                return 0; // clockwise
                            } else {
                                if (features[0] <= -0.034678f) { // gyro_x
                                    return 3; // vertical_updown
                                } else {
                                    return 3; // vertical_updown
                                }
                            }
                        } else {
                            return 0; // clockwise
                        }
                    }
                }
            }
        }
    }
} else {
    if (features[0] <= 1.813613f) { // gyro_x
        if (features[1] <= -0.349448f) { // gyro_y
            if (features[2] <= 0.332004f) { // gyro_z
                if (features[1] <= -0.621566f) { // gyro_y
                    if (features[0] <= 0.910877f) { // gyro_x
                        if (features[1] <= -1.657355f) { // gyro_y
                            if (features[2] <= -0.596218f) { // gyro_z
                                return 0; // clockwise
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            if (features[0] <= 0.811058f) { // gyro_x
                                return 0; // clockwise
                            } else {
                                if (features[1] <= -1.409342f) { // gyro_y
                                    return 0; // clockwise
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        }
                    } else {
                        if (features[1] <= -1.035335f) { // gyro_y
                            return 0; // clockwise
                        } else {
                            if (features[0] <= 1.180880f) { // gyro_x
                                return 0; // clockwise
                            } else {
                                return 0; // clockwise
                            }
                        }
                    }
                } else {
                    return 0; // clockwise
                }
            } else {
                if (features[2] <= 1.010797f) { // gyro_z
                    if (features[0] <= 1.118425f) { // gyro_x
                        return 1; // horizontal_swipe
                    } else {
                        if (features[1] <= -0.841497f) { // gyro_y
                            return 0; // clockwise
                        } else {
                            return 0; // clockwise
                        }
                    }
                } else {
                    return 1; // horizontal_swipe
                }
            }
        } else {
            if (features[1] <= 1.542956f) { // gyro_y
                if (features[2] <= -0.733688f) { // gyro_z
                    if (features[1] <= 0.672425f) { // gyro_y
                        if (features[2] <= -1.534174f) { // gyro_z
                            return 1; // horizontal_swipe
                        } else {
                            if (features[2] <= -1.116055f) { // gyro_z
                                return 0; // clockwise
                            } else {
                                if (features[0] <= 0.945787f) { // gyro_x
                                    return 4; // wrist_twist
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        }
                    } else {
                        if (features[0] <= 0.953423f) { // gyro_x
                            return 1; // horizontal_swipe
                        } else {
                            return 1; // horizontal_swipe
                        }
                    }
                } else {
                    if (features[2] <= 0.646222f) { // gyro_z
                        if (features[1] <= 0.042701f) { // gyro_y
                            if (features[2] <= 0.221557f) { // gyro_z
                                if (features[2] <= -0.175749f) { // gyro_z
                                    return 2; // forward_thrust
                                } else {
                                    return 2; // forward_thrust
                                }
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            if (features[0] <= 0.979332f) { // gyro_x
                                if (features[1] <= 0.557117f) { // gyro_y
                                    if (features[0] <= 0.800422f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                } else {
                                    if (features[0] <= 0.815967f) { // gyro_x
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            } else {
                                if (features[1] <= 0.157513f) { // gyro_y
                                    return 2; // forward_thrust
                                } else {
                                    if (features[2] <= 0.495995f) { // gyro_z
                                        return 2; // forward_thrust
                                    } else {
                                        return 2; // forward_thrust
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[2] <= 1.339451f) { // gyro_z
                            if (features[0] <= 1.096606f) { // gyro_x
                                return 2; // forward_thrust
                            } else {
                                if (features[1] <= 1.263382f) { // gyro_y
                                    return 0; // clockwise
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        } else {
                            return 1; // horizontal_swipe
                        }
                    }
                }
            } else {
                if (features[1] <= 1.967908f) { // gyro_y
                    if (features[2] <= 0.612652f) { // gyro_z
                        return 0; // clockwise
                    } else {
                        return 0; // clockwise
                    }
                } else {
                    return 0; // clockwise
                }
            }
        }
    } else {
        if (features[2] <= 0.389241f) { // gyro_z
            if (features[1] <= -0.078820f) { // gyro_y
                if (features[1] <= -0.387967f) { // gyro_y
                    return 0; // clockwise
                } else {
                    if (features[2] <= 0.205107f) { // gyro_z
                        return 0; // clockwise
                    } else {
                        return 0; // clockwise
                    }
                }
            } else {
                if (features[1] <= 1.936098f) { // gyro_y
                    if (features[0] <= 2.717986f) { // gyro_x
                        if (features[1] <= 0.328488f) { // gyro_y
                            return 0; // clockwise
                        } else {
                            if (features[2] <= 0.292894f) { // gyro_z
                                return 2; // forward_thrust
                            } else {
                                return 2; // forward_thrust
                            }
                        }
                    } else {
                        if (features[1] <= 0.443796f) { // gyro_y
                            if (features[0] <= 3.248446f) { // gyro_x
                                return 0; // clockwise
                            } else {
                                return 0; // clockwise
                            }
                        } else {
                            if (features[0] <= 3.134172f) { // gyro_x
                                return 2; // forward_thrust
                            } else {
                                if (features[2] <= 0.140652f) { // gyro_z
                                    return 0; // clockwise
                                } else {
                                    return 0; // clockwise
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= 2.565077f) { // gyro_y
                        return 0; // clockwise
                    } else {
                        return 0; // clockwise
                    }
                }
            }
        } else {
            if (features[2] <= 1.627149f) { // gyro_z
                if (features[1] <= -0.332797f) { // gyro_y
                    if (features[2] <= 0.860737f) { // gyro_z
                        if (features[0] <= 1.957069f) { // gyro_x
                            return 0; // clockwise
                        } else {
                            return 0; // clockwise
                        }
                    } else {
                        return 0; // clockwise
                    }
                } else {
                    return 0; // clockwise
                }
            } else {
                return 1; // horizontal_swipe
            }
        }
    }
}
}

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
