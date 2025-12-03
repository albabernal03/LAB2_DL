/*
 * Random Forest Classifier for Arduino
 * Generated automatically from trained model
 * 
 * Gesture Recognition - Lab 2
 * Model: Random Forest (10 trees)
 * Features: 39 (statistical features from gyroscope data)
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
#define NUM_FEATURES 39
#define NUM_SAMPLES 119

// Scaler parameters (from StandardScaler)
const float SCALER_MEAN[NUM_FEATURES] = {
0.3158669735325633f, 1.6947440132368656f, -6.304147336230769f, 6.113711096076923f, 
    12.417858432307694f, -0.6923188245461538f, 0.37960254076807687f, 1.4703913890326925f, 
    -0.3642126998996506f, 1.8739574463444455f, 7733.350511954879f, 1.7597328565053625f, 
    0.19883012609000356f, 1.9106509732690442f, -6.1008459895384615f, 4.736243441153847f, 
    10.837089430692306f, -0.9691234959453847f, 0.38136433382038465f, 1.5873040446153843f, 
    -0.4131687204662818f, 0.8498650784092618f, 8452.510073597809f, 1.932533292677306f, 
    -0.3371175235206698f, 2.3061656975486895f, -6.116578969384615f, 5.40219713f, 
    11.518776099384617f, -2.043011478515385f, -0.4081597979884615f, 1.2675815886249997f, 
    0.00376262544191312f, -0.16443580838179356f, 11782.755945003348f, 2.3913638746043584f, 
    3.612853589014312f, 1.737533069204978f, 10.43229778130241f
};

const float SCALER_SCALE[NUM_FEATURES] = {
    0.6026668934226668f, 1.185090890857543f, 3.466630573327453f, 3.572824900947573f, 
    6.648592061541848f, 0.546131609131852f, 0.5381196861624398f, 1.6334909047137964f, 
    0.528818520734591f, 1.699188566068305f, 9308.444512705568f, 1.281763353013164f, 
    0.33467792191273715f, 1.1655437290681419f, 3.232520781270547f, 2.545464137380137f, 
    5.075039812780032f, 1.0202802318543998f, 0.2876845672254224f, 1.3816680066278788f, 
    0.8164243499570095f, 2.3454290125406763f, 8882.464433249073f, 1.1941260943361967f, 
    0.6531186261093178f, 1.1344802016347308f, 2.0981843135177782f, 2.791901457976476f, 
    4.174928680026406f, 1.6231210497964959f, 0.7372895322802623f, 1.152875191537566f, 
    0.6637051055230315f, 1.4203872880385728f, 9280.140136423994f, 1.1945863451941314f, 
    1.5989292592154467f, 0.3005815638086672f, 2.364497831833856f
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
#include <string.h>
void add_vectors(double *v1, double *v2, int size, double *result) {
    for(int i = 0; i < size; ++i)
        result[i] = v1[i] + v2[i];
}
void mul_vector_number(double *v1, double num, int size, double *result) {
    for(int i = 0; i < size; ++i)
        result[i] = v1[i] * num;
}
void score(double * input, double * output) {
    double var0[5];
    double var1[5];
    double var2[5];
    double var3[5];
    double var4[5];
    double var5[5];
    double var6[5];
    double var7[5];
    double var8[5];
    double var9[5];
    double var10[5];
    if (input[20] <= -0.7181514874100685) {
        memcpy(var10, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
    } else {
        if (input[11] <= nan) {
            if (input[33] <= -0.14653228223323822) {
                if (input[35] <= nan) {
                    if (input[25] <= nan) {
                        memcpy(var10, (double[]){0.0, 0.5, 0.0, 0.5, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var10, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                } else {
                    if (input[33] <= -0.7171938717365265) {
                        memcpy(var10, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var10, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                }
            } else {
                memcpy(var10, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
            }
        } else {
            memcpy(var10, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
        }
    }
    double var11[5];
    if (input[29] <= 0.7815590500831604) {
        if (input[6] <= -0.6445512175559998) {
            memcpy(var11, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
        } else {
            if (input[23] <= 1.2866444885730743) {
                if (input[11] <= -0.8648785352706909) {
                    if (input[15] <= -0.6231710016727448) {
                        memcpy(var11, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var11, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                    }
                } else {
                    if (input[38] <= 0.43406352400779724) {
                        memcpy(var11, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var11, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                }
            } else {
                memcpy(var11, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        }
    } else {
        memcpy(var11, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
    }
    add_vectors(var10, var11, 5, var9);
    double var12[5];
    if (input[31] <= nan) {
        if (input[33] <= -0.6348484754562378) {
            if (input[37] <= 0.11081670224666595) {
                memcpy(var12, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            } else {
                memcpy(var12, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        } else {
            if (input[2] <= 0.06742671132087708) {
                if (input[9] <= -0.15108203887939453) {
                    memcpy(var12, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                } else {
                    memcpy(var12, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            } else {
                memcpy(var12, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
            }
        }
    } else {
        if (input[19] <= 1.1017118990421295) {
            if (input[7] <= -0.47177883982658386) {
                memcpy(var12, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
            } else {
                memcpy(var12, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        } else {
            memcpy(var12, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
        }
    }
    add_vectors(var9, var12, 5, var8);
    double var13[5];
    if (input[32] <= 0.6193855255842209) {
        if (input[9] <= 0.7446654140949249) {
            if (input[35] <= nan) {
                if (input[17] <= -0.4503939747810364) {
                    if (input[5] <= 0.09295178949832916) {
                        memcpy(var13, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var13, (double[]){0.0, 0.5, 0.0, 0.5, 0.0}, 5 * sizeof(double));
                    }
                } else {
                    memcpy(var13, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                }
            } else {
                if (input[20] <= 0.5576213598251343) {
                    memcpy(var13, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                } else {
                    memcpy(var13, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            }
        } else {
            memcpy(var13, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
        }
    } else {
        memcpy(var13, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
    }
    add_vectors(var8, var13, 5, var7);
    double var14[5];
    if (input[38] <= -0.5453862249851227) {
        memcpy(var14, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
    } else {
        if (input[21] <= 0.6693054735660553) {
            if (input[9] <= -1.149660348892212) {
                memcpy(var14, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            } else {
                if (input[2] <= inf) {
                    memcpy(var14, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                } else {
                    memcpy(var14, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            }
        } else {
            memcpy(var14, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
        }
    }
    add_vectors(var7, var14, 5, var6);
    double var15[5];
    if (input[36] <= -0.4655567705631256) {
        if (input[0] <= -0.6197699010372162) {
            memcpy(var15, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
        } else {
            memcpy(var15, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
        }
    } else {
        if (input[20] <= 0.44538548588752747) {
            memcpy(var15, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
        } else {
            if (input[9] <= -1.149660348892212) {
                memcpy(var15, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            } else {
                memcpy(var15, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        }
    }
    add_vectors(var6, var15, 5, var5);
    double var16[5];
    if (input[19] <= -0.5417670458555222) {
        if (input[28] <= nan) {
            memcpy(var16, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
        } else {
            if (input[30] <= -0.03842712938785553) {
                memcpy(var16, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
            } else {
                memcpy(var16, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        }
    } else {
        if (input[7] <= nan) {
            if (input[0] <= 0.7465983927249908) {
                memcpy(var16, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
            } else {
                if (input[31] <= nan) {
                    if (input[0] <= inf) {
                        memcpy(var16, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var16, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                    }
                } else {
                    memcpy(var16, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            }
        } else {
            memcpy(var16, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
        }
    }
    add_vectors(var5, var16, 5, var4);
    double var17[5];
    if (input[18] <= nan) {
        if (input[32] <= 0.6228610426187515) {
            if (input[38] <= 0.20104112662374973) {
                if (input[38] <= -0.5453862249851227) {
                    memcpy(var17, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                } else {
                    memcpy(var17, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            } else {
                memcpy(var17, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        } else {
            memcpy(var17, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
        }
    } else {
        if (input[20] <= -0.5319547653198242) {
            memcpy(var17, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
        } else {
            if (input[9] <= -0.5104867517948151) {
                memcpy(var17, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            } else {
                memcpy(var17, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
            }
        }
    }
    add_vectors(var4, var17, 5, var3);
    double var18[5];
    if (input[17] <= -0.7369964718818665) {
        memcpy(var18, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
    } else {
        if (input[26] <= nan) {
            if (input[37] <= -0.39794737100601196) {
                memcpy(var18, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
            } else {
                if (input[35] <= -0.48898858577013016) {
                    memcpy(var18, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                } else {
                    memcpy(var18, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                }
            }
        } else {
            if (input[34] <= nan) {
                if (input[13] <= nan) {
                    memcpy(var18, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                } else {
                    memcpy(var18, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            } else {
                memcpy(var18, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
            }
        }
    }
    add_vectors(var3, var18, 5, var2);
    double var19[5];
    if (input[34] <= -0.5614428617991507) {
        memcpy(var19, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
    } else {
        if (input[32] <= 0.10789210302755237) {
            memcpy(var19, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
        } else {
            if (input[24] <= nan) {
                if (input[20] <= 0.9460926353931427) {
                    if (input[30] <= nan) {
                        memcpy(var19, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    } else {
                        memcpy(var19, (double[]){0.3333333333333333, 0.6666666666666666, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                } else {
                    memcpy(var19, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                }
            } else {
                if (input[7] <= 0.44463977217674255) {
                    memcpy(var19, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                } else {
                    memcpy(var19, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            }
        }
    }
    add_vectors(var2, var19, 5, var1);
    mul_vector_number(var1, 0.1, 5, var0);
    memcpy(output, var0, 5 * sizeof(double));
}


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
