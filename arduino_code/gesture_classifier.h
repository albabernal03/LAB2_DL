/*
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

// Scaler parameters (from StandardScaler)
const float SCALER_MEAN[3] = {0.25840647934887806f, 0.17319652791701426f, -0.2987822896279074f};
const float SCALER_SCALE[3] = {1.9529634847972555f, 2.143301655951765f, 3.173223810876215f};

// Gesture labels
const char* GESTURE_NAMES[] = {
    "clockwise",
    "horizontal_swipe", 
    "forward_thrust",
    "vertical_updown",
    "wrist_twist"
};

// Apply StandardScaler normalization
void normalize_features(float* features, float* normalized) {
    for (int i = 0; i < 3; i++) {
        normalized[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }
}

// Decision tree prediction function
#include <string.h>
void score(double * input, double * output) {
    double var0[5];
    if (input[0] <= 0.7330572307109833) {
        if (input[2] <= -0.6459018588066101) {
            if (input[1] <= -0.40362270176410675) {
                if (input[0] <= -0.8375948667526245) {
                    if (input[2] <= -1.0401859879493713) {
                        if (input[1] <= -0.6886636018753052) {
                            if (input[1] <= -2.34945547580719) {
                                memcpy(var0, (double[]){0.47058823529411764, 0.0, 0.4117647058823529, 0.11764705882352941, 0.0}, 5 * sizeof(double));
                            } else {
                                if (input[0] <= -1.1875073909759521) {
                                    if (input[0] <= -1.4141461253166199) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[0] <= -1.3762366771697998) {
                                            memcpy(var0, (double[]){0.8, 0.2, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= -0.9589596688747406) {
                                        if (input[0] <= -1.0118693113327026) {
                                            memcpy(var0, (double[]){0.9545454545454546, 0.0, 0.0, 0.045454545454545456, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.0, 0.5, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[0] <= -1.8417866826057434) {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                if (input[0] <= -0.9633233547210693) {
                                    if (input[2] <= -1.883138358592987) {
                                        memcpy(var0, (double[]){0.8, 0.1, 0.0, 0.1, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[1] <= -0.5847864151000977) {
                                            memcpy(var0, (double[]){0.7894736842105263, 0.21052631578947367, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.49056603773584906, 0.5094339622641509, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= -1.4416878819465637) {
                                        memcpy(var0, (double[]){0.8, 0.2, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.6, 0.1, 0.0, 0.3, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[1] <= -2.040308952331543) {
                            if (input[1] <= -2.562428593635559) {
                                memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                if (input[0] <= -1.4100552201271057) {
                                    memcpy(var0, (double[]){0.0, 0.0, 0.8571428571428571, 0.14285714285714285, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.2727272727272727, 0.0, 0.2727272727272727, 0.45454545454545453, 0.0}, 5 * sizeof(double));
                                }
                            }
                        } else {
                            if (input[1] <= -0.6357309818267822) {
                                if (input[2] <= -0.899190366268158) {
                                    if (input[2] <= -0.9272216260433197) {
                                        memcpy(var0, (double[]){0.8421052631578947, 0.10526315789473684, 0.05263157894736842, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.6923076923076923, 0.0, 0.3076923076923077, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[2] <= -0.8483312129974365) {
                                        memcpy(var0, (double[]){0.3, 0.0, 0.3, 0.4, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[1] <= -1.3131690621376038) {
                                            memcpy(var0, (double[]){0.45454545454545453, 0.0, 0.5454545454545454, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.18181818181818182, 0.0, 0.7272727272727273, 0.09090909090909091, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                memcpy(var0, (double[]){0.25, 0.6666666666666666, 0.0, 0.0, 0.08333333333333333}, 5 * sizeof(double));
                            }
                        }
                    }
                } else {
                    if (input[1] <= -0.8151550889015198) {
                        if (input[0] <= 0.11614197120070457) {
                            if (input[2] <= -1.1744675040245056) {
                                if (input[2] <= -2.130552053451538) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[1] <= -2.0097423791885376) {
                                        if (input[2] <= -1.6315284371376038) {
                                            memcpy(var0, (double[]){0.8181818181818182, 0.0, 0.0, 0.18181818181818182, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.18518518518518517, 0.0, 0.0, 0.8148148148148148, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.23486153781414032) {
                                            memcpy(var0, (double[]){0.6761363636363636, 0.0, 0.0, 0.32386363636363635, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.4418604651162791, 0.0, 0.0, 0.5581395348837209, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[0] <= -0.2640436440706253) {
                                    if (input[0] <= -0.5943196713924408) {
                                        if (input[2] <= -0.9265502393245697) {
                                            memcpy(var0, (double[]){0.27692307692307694, 0.0, 0.0, 0.7230769230769231, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.09433962264150944, 0.0, 0.2641509433962264, 0.6415094339622641, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -1.0482574105262756) {
                                            memcpy(var0, (double[]){0.07931034482758621, 0.0, 0.006896551724137931, 0.9137931034482759, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.3333333333333333, 0.0, 0.16666666666666666, 0.5, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= -2.295777440071106) {
                                        if (input[1] <= -2.5350924730300903) {
                                            memcpy(var0, (double[]){0.2222222222222222, 0.0, 0.0, 0.7777777777777778, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.047619047619047616, 0.9523809523809523, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= -1.0114833116531372) {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.0, 0.5, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.2361111111111111, 0.0, 0.0, 0.7638888888888888, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[1] <= -1.2773836255073547) {
                                if (input[2] <= -0.9545814990997314) {
                                    if (input[2] <= -1.6196109652519226) {
                                        if (input[1] <= -1.6248003840446472) {
                                            memcpy(var0, (double[]){0.782608695652174, 0.0, 0.0, 0.21739130434782608, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.0, 0.5, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= -1.0927236676216125) {
                                            memcpy(var0, (double[]){0.9893617021276596, 0.0, 0.0, 0.010638297872340425, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8387096774193549, 0.0, 0.0, 0.16129032258064516, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= 0.5006913244724274) {
                                        if (input[1] <= -1.5318577289581299) {
                                            memcpy(var0, (double[]){0.6, 0.0, 0.0, 0.4, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.2, 0.0, 0.0, 0.8, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -1.7110334038734436) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9, 0.0, 0.0, 0.1, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[1] <= -0.9220143854618073) {
                                    if (input[2] <= -0.9386355578899384) {
                                        if (input[0] <= 0.39214472472667694) {
                                            memcpy(var0, (double[]){0.475, 0.0, 0.0, 0.525, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8235294117647058, 0.0, 0.0, 0.17647058823529413, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.2, 0.0, 0.0, 0.8, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[2] <= -1.4314489364624023) {
                                        memcpy(var0, (double[]){0.36363636363636365, 0.0, 0.0, 0.6363636363636364, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.25, 0.0, 0.0, 0.5, 0.25}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[2] <= -1.258561372756958) {
                            if (input[0] <= -0.5054097026586533) {
                                if (input[2] <= -1.4604873061180115) {
                                    if (input[2] <= -1.5967831015586853) {
                                        memcpy(var0, (double[]){0.6428571428571429, 0.2857142857142857, 0.0, 0.07142857142857142, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.2727272727272727, 0.45454545454545453, 0.0, 0.2727272727272727, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.5555555555555556, 0.05555555555555555, 0.0, 0.2222222222222222, 0.16666666666666666}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[0] <= 0.1974155306816101) {
                                    if (input[2] <= -1.7451640367507935) {
                                        if (input[2] <= -1.8866632580757141) {
                                            memcpy(var0, (double[]){0.5714285714285714, 0.07142857142857142, 0.0, 0.35714285714285715, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.4, 0.0, 0.0, 0.6, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.003777211473789066) {
                                            memcpy(var0, (double[]){0.18947368421052632, 0.0, 0.0, 0.8105263157894737, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.041666666666666664, 0.0, 0.0, 0.9583333333333334, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= 0.3864173889160156) {
                                        if (input[1] <= -0.551486074924469) {
                                            memcpy(var0, (double[]){0.9, 0.0, 0.0, 0.1, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.0, 0.5, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.45978181064128876) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.9, 0.1}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.5, 0.0, 0.0, 0.22727272727272727, 0.2727272727272727}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[0] <= -0.3346807211637497) {
                                if (input[2] <= -0.8552131652832031) {
                                    if (input[0] <= -0.6341382563114166) {
                                        memcpy(var0, (double[]){0.5714285714285715, 0.14285714285714288, 0.0, 0.28571428571428575, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[0] <= -0.46859118342399597) {
                                            memcpy(var0, (double[]){0.08333333333333333, 0.0, 0.0, 0.9166666666666666, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.38461538461538464, 0.07692307692307693, 0.0, 0.5384615384615384, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.1, 0.0, 0.1, 0.6, 0.2}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[2] <= -0.7855546772480011) {
                                    if (input[1] <= -0.477181613445282) {
                                        if (input[2] <= -0.8924762904644012) {
                                            memcpy(var0, (double[]){0.0759493670886076, 0.0, 0.0, 0.6835443037974683, 0.24050632911392406}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.047619047619047616, 0.0, 0.0, 0.9523809523809523, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.08777804672718048) {
                                            memcpy(var0, (double[]){0.060606060606060615, 0.0, 0.0, 0.5454545454545455, 0.393939393939394}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.18181818181818182, 0.8181818181818182}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= -0.7147211730480194) {
                                        if (input[1] <= -0.6168442070484161) {
                                            memcpy(var0, (double[]){0.1, 0.0, 0.4, 0.5, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0625, 0.9375, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.06666666666666667, 0.6666666666666666, 0.26666666666666666}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[0] <= -0.820140153169632) {
                    if (input[0] <= -1.2393261790275574) {
                        if (input[1] <= -0.12852218747138977) {
                            if (input[2] <= -1.3119382858276367) {
                                if (input[0] <= -1.5805113315582275) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[1] <= -0.2500435337424278) {
                                        memcpy(var0, (double[]){0.7333333333333333, 0.26666666666666666, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.9285714285714286, 0.0, 0.0, 0.07142857142857142, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            } else {
                                if (input[2] <= -1.0702314972877502) {
                                    memcpy(var0, (double[]){0.5833333333333334, 0.4166666666666667, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.13333333333333333, 0.8, 0.06666666666666667, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            }
                        } else {
                            if (input[0] <= -1.4283281564712524) {
                                if (input[1] <= 0.13663789629936218) {
                                    if (input[0] <= -1.7948771119117737) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[2] <= -1.4109709858894348) {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.75, 0.25, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[2] <= -1.727539598941803) {
                                    memcpy(var0, (double[]){0.631578947368421, 0.3684210526315789, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[1] <= 0.5153670608997345) {
                                        if (input[2] <= -1.2023309469223022) {
                                            memcpy(var0, (double[]){0.875, 0.125, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.7142857142857143, 0.2857142857142857, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[1] <= 0.22336788475513458) {
                            if (input[2] <= -0.9876483380794525) {
                                if (input[2] <= -1.4185243248939514) {
                                    if (input[0] <= -0.9145046770572662) {
                                        if (input[2] <= -1.6630845665931702) {
                                            memcpy(var0, (double[]){0.5555555555555556, 0.4074074074074074, 0.0, 0.037037037037037035, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9333333333333333, 0.06666666666666667, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.4210526315789474, 0.4736842105263158, 0.0, 0.0, 0.10526315789473685}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[0] <= -1.035869538784027) {
                                        if (input[0] <= -1.0784154534339905) {
                                            memcpy(var0, (double[]){0.42857142857142855, 0.5714285714285714, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8, 0.1, 0.0, 0.1, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= -1.235397756099701) {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.28, 0.72, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[1] <= -0.10044056549668312) {
                                    if (input[1] <= -0.22742915153503418) {
                                        memcpy(var0, (double[]){0.09090909090909091, 0.8181818181818182, 0.0, 0.09090909090909091, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[0] <= -0.999869167804718) {
                                        memcpy(var0, (double[]){0.08333333333333333, 0.6666666666666666, 0.0, 0.0, 0.25}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.9230769230769231, 0.0, 0.0, 0.07692307692307693}, 5 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[2] <= -1.71075439453125) {
                                if (input[2] <= -1.8875024914741516) {
                                    memcpy(var0, (double[]){0.1, 0.9, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.5, 0.5, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[1] <= 0.8242648839950562) {
                                    if (input[0] <= -0.9095955491065979) {
                                        if (input[1] <= 0.37520742416381836) {
                                            memcpy(var0, (double[]){0.7916666666666666, 0.20833333333333334, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9814814814814815, 0.018518518518518517, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= -1.2510080337524414) {
                                            memcpy(var0, (double[]){0.3, 0.7, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.8181818181818182, 0.18181818181818182, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            }
                        }
                    }
                } else {
                    if (input[1] <= 0.18683692067861557) {
                        if (input[0] <= -0.24167978763580322) {
                            if (input[1] <= -0.07409851625561714) {
                                if (input[0] <= -0.4745912253856659) {
                                    if (input[2] <= -0.8837479650974274) {
                                        if (input[1] <= -0.2080453634262085) {
                                            memcpy(var0, (double[]){0.2741935483870968, 0.5322580645161291, 0.0, 0.16129032258064518, 0.03225806451612904}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.12727272727272726, 0.8181818181818182, 0.0, 0.03636363636363636, 0.01818181818181818}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.6979570984840393) {
                                            memcpy(var0, (double[]){0.09090909090909091, 0.9090909090909091, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.04761904761904762, 0.4285714285714286, 0.0, 0.04761904761904762, 0.4761904761904762}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= -1.2775286436080933) {
                                        if (input[1] <= -0.2080453634262085) {
                                            memcpy(var0, (double[]){0.047619047619047616, 0.21428571428571427, 0.0, 0.7142857142857143, 0.023809523809523808}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.09523809523809523, 0.5476190476190477, 0.0, 0.30952380952380953, 0.047619047619047616}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= -0.8835801184177399) {
                                            memcpy(var0, (double[]){0.02127659574468085, 0.10638297872340426, 0.0, 0.6382978723404256, 0.23404255319148937}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.25, 0.0, 0.20833333333333334, 0.5416666666666666}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[2] <= -1.3401374816894531) {
                                    if (input[1] <= -0.020917391404509544) {
                                        if (input[0] <= -0.3624991774559021) {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.1, 0.9, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.030275652185082436) {
                                            memcpy(var0, (double[]){0.23529411764705882, 0.6470588235294118, 0.0, 0.11764705882352941, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.03418803418803419, 0.9145299145299145, 0.0, 0.05128205128205128, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= -0.42740893363952637) {
                                        if (input[2] <= -0.7519842982292175) {
                                            memcpy(var0, (double[]){0.08275862068965517, 0.8275862068965517, 0.0, 0.027586206896551724, 0.06206896551724138}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.03571428571428571, 0.9642857142857143, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.01072848029434681) {
                                            memcpy(var0, (double[]){0.0, 0.36363636363636365, 0.0, 0.0, 0.6363636363636364}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.026785714285714284, 0.7142857142857143, 0.0, 0.07142857142857142, 0.1875}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[2] <= -1.405096173286438) {
                                if (input[1] <= 0.034003302454948425) {
                                    if (input[2] <= -1.5734516978263855) {
                                        if (input[1] <= -0.2264351099729538) {
                                            memcpy(var0, (double[]){0.19230769230769232, 0.11538461538461539, 0.0, 0.6153846153846154, 0.07692307692307693}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.23404255319148937, 0.425531914893617, 0.0, 0.3404255319148936, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.1808610036969185) {
                                            memcpy(var0, (double[]){0.15384615384615385, 0.0, 0.0, 0.8461538461538461, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.1111111111111111, 0.1388888888888889, 0.0, 0.4166666666666667, 0.3333333333333333}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= -1.6986690759658813) {
                                        if (input[1] <= 0.12247283011674881) {
                                            memcpy(var0, (double[]){0.058823529411764705, 0.8823529411764706, 0.0, 0.0, 0.058823529411764705}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.09090909090909091, 0.7272727272727273, 0.0, 0.18181818181818182, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.12549681216478348) {
                                            memcpy(var0, (double[]){0.0, 0.6, 0.0, 0.16, 0.24}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.05263157894736842, 0.5263157894736842, 0.0, 0.0, 0.42105263157894735}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[0] <= 0.029141112230718136) {
                                    if (input[1] <= -0.13423791527748108) {
                                        if (input[2] <= -1.0794633626937866) {
                                            memcpy(var0, (double[]){0.03448275862068966, 0.0, 0.0, 0.5862068965517242, 0.37931034482758624}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.010309278350515464, 0.05154639175257732, 0.010309278350515464, 0.6804123711340206, 0.24742268041237114}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.014619525987654924) {
                                            memcpy(var0, (double[]){0.045454545454545456, 0.10909090909090909, 0.0, 0.2818181818181818, 0.5636363636363636}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.007575757575757576, 0.38636363636363635, 0.0, 0.25, 0.3560606060606061}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= -1.2696396112442017) {
                                        if (input[1] <= -0.0152016612701118) {
                                            memcpy(var0, (double[]){0.14814814814814814, 0.0, 0.0, 0.4074074074074074, 0.4444444444444444}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.07142857142857142, 0.21428571428571427, 0.0, 0.07142857142857142, 0.6428571428571429}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.11886927112936974) {
                                            memcpy(var0, (double[]){0.0, 0.010416666666666666, 0.0, 0.2916666666666667, 0.6979166666666666}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.024489795918367346, 0.02857142857142857, 0.0, 0.12653061224489795, 0.8204081632653061}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[2] <= -1.5652269124984741) {
                            if (input[2] <= -1.900930643081665) {
                                if (input[0] <= -0.6393201351165771) {
                                    if (input[0] <= -0.6971389055252075) {
                                        memcpy(var0, (double[]){0.2, 0.8, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.07692307692307693, 0.9230769230769231, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[1] <= 0.5501584708690643) {
                                        if (input[0] <= 0.008686351589858532) {
                                            memcpy(var0, (double[]){0.0, 0.9930555555555556, 0.0, 0.006944444444444444, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.65, 0.0, 0.35, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            } else {
                                if (input[0] <= -0.0573143046349287) {
                                    if (input[0] <= -0.4399545192718506) {
                                        if (input[1] <= 0.38862696290016174) {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.25, 0.75, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.41347795724868774) {
                                            memcpy(var0, (double[]){0.0449438202247191, 0.9325842696629213, 0.0, 0.02247191011235955, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.008333333333333333, 0.9916666666666667, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.7772964835166931) {
                                        if (input[2] <= -1.6340462565422058) {
                                            memcpy(var0, (double[]){0.06493506493506493, 0.8311688311688312, 0.0, 0.012987012987012988, 0.09090909090909091}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.045454545454545456, 0.5454545454545454, 0.0, 0.0, 0.4090909090909091}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 1.1734213829040527) {
                                            memcpy(var0, (double[]){0.011764705882352941, 0.9882352941176471, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.13333333333333333, 0.8666666666666667, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[1] <= 0.8878833949565887) {
                                if (input[0] <= 0.08559620752930641) {
                                    if (input[0] <= -0.5007732808589935) {
                                        if (input[1] <= 0.7427535951137543) {
                                            memcpy(var0, (double[]){0.3, 0.6333333333333333, 0.0, 0.044444444444444446, 0.022222222222222223}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.9090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.5248104333877563) {
                                            memcpy(var0, (double[]){0.030425963488843813, 0.7728194726166329, 0.0, 0.060851926977687626, 0.1359026369168357}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.09282700421940929, 0.8607594936708861, 0.0, 0.03375527426160337, 0.012658227848101266}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.5809736549854279) {
                                        if (input[1] <= 0.4286370575428009) {
                                            memcpy(var0, (double[]){0.027777777777777776, 0.2777777777777778, 0.0, 0.08333333333333333, 0.6111111111111112}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.5795454545454546, 0.0, 0.045454545454545456, 0.375}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.19932464510202408) {
                                            memcpy(var0, (double[]){0.05555555555555555, 0.9444444444444444, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.02666666666666667, 0.8133333333333334, 0.0, 0.05333333333333334, 0.10666666666666667}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[2] <= -1.0905416011810303) {
                                    if (input[1] <= 1.2067217230796814) {
                                        if (input[0] <= -0.451681911945343) {
                                            memcpy(var0, (double[]){0.75, 0.25, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.09375, 0.90625, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.9090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[0] <= 0.09595995023846626) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.4166666666666667, 0.0, 0.0, 0.5833333333333334, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[2] <= 0.5606178045272827) {
                if (input[1] <= 1.172675907611847) {
                    if (input[2] <= 0.042962491512298584) {
                        if (input[1] <= -0.6725104451179504) {
                            if (input[0] <= -0.6483202278614044) {
                                if (input[2] <= -0.47284647822380066) {
                                    if (input[0] <= -1.0688698887825012) {
                                        if (input[1] <= -1.5668976306915283) {
                                            memcpy(var0, (double[]){0.014925373134328358, 0.0, 0.9850746268656716, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.20833333333333334, 0.0, 0.75, 0.041666666666666664, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.8021399974822998) {
                                            memcpy(var0, (double[]){0.125, 0.0, 0.625, 0.25, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.25, 0.75, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= -0.9553146958351135) {
                                        if (input[0] <= -1.2076894640922546) {
                                            memcpy(var0, (double[]){0.002881844380403458, 0.0, 0.9971181556195965, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.06086956521739131, 0.0, 0.9391304347826087, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.7512880265712738) {
                                            memcpy(var0, (double[]){0.0, 0.019230769230769232, 0.9038461538461539, 0.0, 0.07692307692307693}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0967741935483871, 0.8387096774193549, 0.06451612903225806, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[1] <= -0.9717163741588593) {
                                    if (input[0] <= -0.022404870949685574) {
                                        if (input[2] <= -0.32496894896030426) {
                                            memcpy(var0, (double[]){0.05976095617529881, 0.0, 0.09561752988047809, 0.8446215139442231, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.02469135802469136, 0.0, 0.40740740740740744, 0.54320987654321, 0.02469135802469136}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.46769098937511444) {
                                            memcpy(var0, (double[]){0.1694915254237288, 0.0, 0.01694915254237288, 0.7966101694915254, 0.01694915254237288}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.6666666666666666, 0.0, 0.0, 0.3333333333333333, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= -0.24842845648527145) {
                                        if (input[0] <= -0.05485973320901394) {
                                            memcpy(var0, (double[]){0.012048192771084338, 0.0, 0.12048192771084337, 0.8313253012048193, 0.03614457831325301}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.04, 0.64, 0.32}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.42086340487003326) {
                                            memcpy(var0, (double[]){0.0, 0.04, 0.64, 0.12, 0.2}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.01, 0.03, 0.24, 0.49, 0.23}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[0] <= -0.4396817833185196) {
                                if (input[2] <= -0.2680671513080597) {
                                    if (input[1] <= 0.3739648759365082) {
                                        if (input[1] <= -0.37951724231243134) {
                                            memcpy(var0, (double[]){0.0, 0.125, 0.6875, 0.125, 0.0625}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.04230769230769232, 0.7000000000000001, 0.0653846153846154, 0.030769230769230774, 0.16153846153846158}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.6060731709003448) {
                                            memcpy(var0, (double[]){0.6, 0.0, 0.1, 0.1, 0.2}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= -0.8119582831859589) {
                                        if (input[0] <= -1.0879610180854797) {
                                            memcpy(var0, (double[]){0.0, 0.21333333333333335, 0.7866666666666666, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.1810344827586207, 0.6551724137931034, 0.017241379310344827, 0.14655172413793102}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.23065977543592453) {
                                            memcpy(var0, (double[]){0.0, 0.06976744186046512, 0.413953488372093, 0.04186046511627907, 0.4744186046511628}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.36693548387096775, 0.29838709677419356, 0.016129032258064516, 0.3185483870967742}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[2] <= -0.1436888426542282) {
                                    if (input[1] <= 0.09985842928290367) {
                                        if (input[1] <= -0.16579867154359818) {
                                            memcpy(var0, (double[]){0.0, 0.03096774193548387, 0.05032258064516129, 0.31225806451612903, 0.6064516129032258}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.06944444444444445, 0.018229166666666668, 0.1388888888888889, 0.7734375}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= -0.35165737569332123) {
                                            memcpy(var0, (double[]){0.028455284552845527, 0.36585365853658536, 0.0040650406504065045, 0.17886178861788618, 0.42276422764227645}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.12203389830508475, 0.06440677966101695, 0.13220338983050847, 0.6813559322033899}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= 0.20887019485235214) {
                                        if (input[1] <= -0.29079917073249817) {
                                            memcpy(var0, (double[]){0.0, 0.045871559633027525, 0.19877675840978593, 0.2996941896024465, 0.45565749235474007}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.03795966785290629, 0.18505338078291814, 0.08956109134045077, 0.6874258600237247}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.06465513817965984) {
                                            memcpy(var0, (double[]){0.0, 0.11538461538461539, 0.5192307692307693, 0.15384615384615385, 0.21153846153846154}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.021505376344086023, 0.8279569892473119, 0.11290322580645161, 0.03763440860215054}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[1] <= 0.11874518170952797) {
                            if (input[0] <= 0.1496877670288086) {
                                if (input[0] <= -0.3979540914297104) {
                                    if (input[0] <= -1.2472352981567383) {
                                        if (input[1] <= -0.7082958817481995) {
                                            memcpy(var0, (double[]){0.0, 0.009708737864077669, 0.9902912621359223, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0975609756097561, 0.7317073170731707, 0.0, 0.17073170731707318}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.12777666002511978) {
                                            memcpy(var0, (double[]){0.0, 0.2732732732732733, 0.2732732732732733, 0.04504504504504504, 0.4084084084084084}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.20425531914893616, 0.548936170212766, 0.03829787234042553, 0.20851063829787234}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 0.3101827800273895) {
                                        if (input[1] <= -0.042786262929439545) {
                                            memcpy(var0, (double[]){0.004739336492890996, 0.11753554502369669, 0.25971563981042656, 0.24739336492890995, 0.3706161137440758}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.004178272980501393, 0.02924791086350975, 0.4805013927576602, 0.12674094707520892, 0.3593314763231198}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.44214171171188354) {
                                            memcpy(var0, (double[]){0.05714285714285714, 0.7428571428571429, 0.08571428571428572, 0.05714285714285714, 0.05714285714285714}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.17355371900826447, 0.08471074380165289, 0.39462809917355374, 0.34710743801652894}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[1] <= -0.17996374517679214) {
                                    if (input[2] <= 0.24908465892076492) {
                                        if (input[1] <= -0.22817467898130417) {
                                            memcpy(var0, (double[]){0.18421052631578946, 0.18421052631578946, 0.23684210526315788, 0.10526315789473684, 0.2894736842105263}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.07142857142857142, 0.0, 0.7142857142857143, 0.14285714285714285, 0.07142857142857142}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.33453692495822906) {
                                            memcpy(var0, (double[]){0.17647058823529413, 0.8235294117647058, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.07142857142857142, 0.2857142857142857, 0.07142857142857142, 0.14285714285714285, 0.42857142857142855}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.055623672902584076) {
                                        if (input[0] <= 0.2186884805560112) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.765625, 0.09375, 0.140625}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.03278688524590164, 0.01092896174863388, 0.8852459016393442, 0.01639344262295082, 0.0546448087431694}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.0854448527097702) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.9452054794520548, 0.0410958904109589, 0.0136986301369863}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[0] <= 0.1139601320028305) {
                                if (input[2] <= 0.34643878042697906) {
                                    if (input[1] <= 0.5881803929805756) {
                                        if (input[0] <= -0.19122473895549774) {
                                            memcpy(var0, (double[]){0.0, 0.058981233243967826, 0.6863270777479893, 0.07506702412868632, 0.17962466487935658}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.013513513513513514, 0.0075075075075075074, 0.7927927927927928, 0.10960960960960961, 0.07657657657657657}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.2978621870279312) {
                                            memcpy(var0, (double[]){0.07407407407407407, 0.0, 0.7777777777777778, 0.14814814814814814, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.010638297872340425, 0.0, 0.4787234042553192, 0.5106382978723404, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= -0.6286836564540863) {
                                        if (input[2] <= 0.44110725820064545) {
                                            memcpy(var0, (double[]){0.0, 0.05263157894736842, 0.7894736842105263, 0.0, 0.15789473684210525}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.5, 0.08333333333333333, 0.4166666666666667}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.23902401328086853) {
                                            memcpy(var0, (double[]){0.0, 0.11458333333333333, 0.2708333333333333, 0.5, 0.11458333333333333}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.022222222222222223, 0.44761904761904764, 0.5301587301587302, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[2] <= 0.34845300018787384) {
                                    if (input[1] <= 0.4974743574857712) {
                                        if (input[0] <= 0.42023591697216034) {
                                            memcpy(var0, (double[]){0.00424929178470255, 0.00141643059490085, 0.953257790368272, 0.029745042492917848, 0.0113314447592068}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0030211480362537764, 0.0, 0.9969788519637462, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 0.13158831745386124) {
                                            memcpy(var0, (double[]){0.1, 0.0, 0.625, 0.275, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.9428571428571428, 0.05714285714285714, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= 0.3528716117143631) {
                                        if (input[2] <= 0.4758526384830475) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.8449612403100775, 0.15503875968992248, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.5510204081632653, 0.4489795918367347, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.6214807629585266) {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.8666666666666667, 0.13333333333333333, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[1] <= 1.616514503955841) {
                        if (input[2] <= 0.09751437231898308) {
                            if (input[0] <= -0.2223159670829773) {
                                if (input[0] <= -2.398428797721863) {
                                    memcpy(var0, (double[]){0.9, 0.0, 0.1, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            } else {
                                memcpy(var0, (double[]){0.6, 0.0, 0.0, 0.4, 0.0}, 5 * sizeof(double));
                            }
                        } else {
                            if (input[0] <= -0.6409565210342407) {
                                memcpy(var0, (double[]){0.6363636363636364, 0.0, 0.36363636363636365, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                if (input[0] <= 0.20314286649227142) {
                                    if (input[0] <= -0.3052258789539337) {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.2, 0.8, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[2] <= 0.3586919605731964) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.1, 0.9, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.0, 0.0, 0.36363636363636365, 0.6363636363636364, 0.0}, 5 * sizeof(double));
                                }
                            }
                        }
                    } else {
                        if (input[2] <= 0.4364074170589447) {
                            if (input[0] <= 0.6667838394641876) {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                memcpy(var0, (double[]){0.9230769230769231, 0.0, 0.07692307692307693, 0.0, 0.0}, 5 * sizeof(double));
                            }
                        } else {
                            if (input[1] <= 2.04917049407959) {
                                memcpy(var0, (double[]){0.2857142857142857, 0.0, 0.07142857142857142, 0.6428571428571429, 0.0}, 5 * sizeof(double));
                            } else {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            }
                        }
                    }
                }
            } else {
                if (input[1] <= -0.19139520078897476) {
                    if (input[0] <= -0.13449690490961075) {
                        if (input[1] <= -0.42350345849990845) {
                            if (input[2] <= 1.9616778492927551) {
                                if (input[0] <= -0.8092309832572937) {
                                    if (input[0] <= -1.6604212522506714) {
                                        memcpy(var0, (double[]){0.0, 0.4, 0.6, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.5833333333333334, 0.25, 0.0, 0.16666666666666666}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[0] <= -0.35186271369457245) {
                                        if (input[2] <= 0.9167996942996979) {
                                            memcpy(var0, (double[]){0.0, 0.84, 0.0, 0.16, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.39215686274509803, 0.0, 0.3333333333333333, 0.27450980392156865}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= -0.6690313220024109) {
                                            memcpy(var0, (double[]){0.0, 0.9074074074074074, 0.0, 0.09259259259259259, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.7207207207207207, 0.0, 0.04504504504504504, 0.23423423423423423}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[0] <= -0.5861377716064453) {
                                    if (input[1] <= -0.5994485318660736) {
                                        if (input[1] <= -0.9694797694683075) {
                                            memcpy(var0, (double[]){0.0, 0.9, 0.0, 0.0, 0.1}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.5714285714285714, 0.0, 0.0, 0.42857142857142855}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.18181818181818182, 0.0, 0.0, 0.8181818181818182}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[1] <= -0.5532256662845612) {
                                        if (input[1] <= -0.6498960554599762) {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.9833333333333333, 0.0, 0.0, 0.016666666666666666}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 2.613614797592163) {
                                            memcpy(var0, (double[]){0.0, 0.8076923076923077, 0.0, 0.0, 0.19230769230769232}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[2] <= 1.6291632652282715) {
                                if (input[0] <= -0.7604122757911682) {
                                    memcpy(var0, (double[]){0.0, 0.15384615384615385, 0.3076923076923077, 0.0, 0.5384615384615384}, 5 * sizeof(double));
                                } else {
                                    if (input[2] <= 1.3005090951919556) {
                                        if (input[2] <= 1.0160000920295715) {
                                            memcpy(var0, (double[]){0.0, 0.21153846153846154, 0.0, 0.34615384615384615, 0.4423076923076923}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.7777777777777778, 0.2222222222222222}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.2866802513599396) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.4782608695652174, 0.5217391304347826}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.56, 0.0, 0.2, 0.24}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[0] <= -0.2719528377056122) {
                                    if (input[0] <= -0.639047384262085) {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[1] <= -0.32459650933742523) {
                                            memcpy(var0, (double[]){0.0, 0.4, 0.0, 0.1, 0.5}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.13793103448275862, 0.0, 0.0, 0.8620689655172413}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= -0.21031584590673447) {
                                        memcpy(var0, (double[]){0.0, 0.7, 0.0, 0.1, 0.2}, 5 * sizeof(double));
                                    } else {
                                        if (input[1] <= -0.3680857867002487) {
                                            memcpy(var0, (double[]){0.0, 0.9, 0.0, 0.0, 0.1}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[2] <= 1.3172943592071533) {
                            if (input[1] <= -0.49830494821071625) {
                                if (input[1] <= -1.0542216897010803) {
                                    memcpy(var0, (double[]){0.3, 0.7, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[1] <= -0.6416952311992645) {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[0] <= 0.24432506412267685) {
                                            memcpy(var0, (double[]){0.0, 0.8809523809523809, 0.0, 0.11904761904761904, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[0] <= 0.00868635205551982) {
                                    if (input[2] <= 0.768418550491333) {
                                        memcpy(var0, (double[]){0.0, 0.46153846153846156, 0.0, 0.07692307692307693, 0.46153846153846156}, 5 * sizeof(double));
                                    } else {
                                        if (input[0] <= -0.07858724892139435) {
                                            memcpy(var0, (double[]){0.0, 0.10526315789473684, 0.0, 0.6842105263157895, 0.21052631578947367}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.5, 0.0, 0.35, 0.15}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[0] <= 0.26477980613708496) {
                                        if (input[1] <= -0.21127599477767944) {
                                            memcpy(var0, (double[]){0.0, 0.6981132075471698, 0.0, 0.2830188679245283, 0.018867924528301886}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.6, 0.0, 0.1, 0.3}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.9285714285714286, 0.0, 0.0, 0.07142857142857142}, 5 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[1] <= -0.609885960817337) {
                                if (input[1] <= -0.7908011078834534) {
                                    if (input[1] <= -0.8067057430744171) {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.8181818181818182, 0.0, 0.0, 0.18181818181818182}, 5 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[2] <= 1.568568766117096) {
                                    if (input[2] <= 1.4131377339363098) {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        if (input[1] <= -0.22171341627836227) {
                                            memcpy(var0, (double[]){0.0, 0.9130434782608695, 0.0, 0.057971014492753624, 0.028985507246376812}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.8, 0.0, 0.0, 0.2}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 2.3655295372009277) {
                                        if (input[0] <= 0.3681444823741913) {
                                            memcpy(var0, (double[]){0.0, 0.9391534391534392, 0.0, 0.0, 0.06084656084656084}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[1] <= 0.6102977991104126) {
                        if (input[0] <= -0.23513426631689072) {
                            if (input[0] <= -0.5834104716777802) {
                                if (input[2] <= 0.9555734694004059) {
                                    if (input[1] <= 0.19652880728244781) {
                                        if (input[1] <= 0.009897836833260953) {
                                            memcpy(var0, (double[]){0.0, 0.2, 0.0, 0.1, 0.7}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.1, 0.1, 0.6, 0.2}, 5 * sizeof(double));
                                    }
                                } else {
                                    if (input[0] <= -0.6788659989833832) {
                                        if (input[2] <= 2.40665340423584) {
                                            memcpy(var0, (double[]){0.0, 0.0036101083032490976, 0.0, 0.0, 0.9963898916967509}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.18181818181818182, 0.0, 0.0, 0.8181818181818182}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 1.473732352256775) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.14, 0.86}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.02564102564102564, 0.0, 0.0, 0.9743589743589743}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[2] <= 1.6498091220855713) {
                                    if (input[2] <= 0.8046745657920837) {
                                        if (input[1] <= 0.20721474289894104) {
                                            memcpy(var0, (double[]){0.0, 0.14655172413793102, 0.0, 0.22413793103448276, 0.6293103448275862}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.05555555555555555, 0.7222222222222222, 0.2222222222222222}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.3504990488290787) {
                                            memcpy(var0, (double[]){0.0, 0.016025641025641024, 0.0, 0.4519230769230769, 0.532051282051282}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.7142857142857143, 0.2857142857142857}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 1.8574419021606445) {
                                        if (input[1] <= 0.2290836125612259) {
                                            memcpy(var0, (double[]){0.0, 0.038461538461538464, 0.0, 0.23076923076923078, 0.7307692307692307}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.41377241909503937) {
                                            memcpy(var0, (double[]){0.0, 0.08888888888888889, 0.0, 0.0, 0.9111111111111111}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[0] <= 0.11286921054124832) {
                                if (input[1] <= 0.16471952199935913) {
                                    if (input[2] <= 1.4817891716957092) {
                                        if (input[0] <= -0.014768429566174746) {
                                            memcpy(var0, (double[]){0.0, 0.14960629921259844, 0.0, 0.6889763779527559, 0.16141732283464566}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.4063926940639269, 0.0, 0.5159817351598174, 0.0776255707762557}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.035223182290792465) {
                                            memcpy(var0, (double[]){0.0, 0.26, 0.0, 0.32, 0.42}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.6842105263157895, 0.0, 0.0, 0.3157894736842105}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 1.556986927986145) {
                                        if (input[2] <= 1.0757553577423096) {
                                            memcpy(var0, (double[]){0.0, 0.06372549019607843, 0.03431372549019608, 0.8137254901960784, 0.08823529411764706}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.8804347826086957, 0.11956521739130435}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= -0.0005864687118446454) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.3333333333333333, 0.6666666666666666}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.15384615384615385, 0.0, 0.15384615384615385, 0.6923076923076923}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[1] <= 0.29593275487422943) {
                                    if (input[2] <= 0.7914142310619354) {
                                        if (input[0] <= 0.22087031602859497) {
                                            memcpy(var0, (double[]){0.0, 0.3125, 0.0, 0.625, 0.0625}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.4, 0.3, 0.3, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.3520534187555313) {
                                            memcpy(var0, (double[]){0.0, 0.7833333333333333, 0.0, 0.1375, 0.07916666666666666}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.967741935483871, 0.0, 0.03225806451612903, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 0.8656048476696014) {
                                        if (input[0] <= 0.3820537179708481) {
                                            memcpy(var0, (double[]){0.0, 0.07407407407407407, 0.1111111111111111, 0.8148148148148148, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.9, 0.1, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[2] <= 1.185698390007019) {
                                            memcpy(var0, (double[]){0.0, 0.8, 0.0, 0.2, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.15789473684210525, 0.0, 0.5526315789473685, 0.2894736842105263}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[1] <= 2.190821051597595) {
                            if (input[2] <= 1.6858972311019897) {
                                if (input[0] <= 0.2822345346212387) {
                                    if (input[0] <= -0.6998662054538727) {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.3, 0.3, 0.4}, 5 * sizeof(double));
                                    } else {
                                        if (input[1] <= 1.0223273634910583) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.027210884353741496, 0.9489795918367347, 0.023809523809523808}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.006329113924050633, 0.0, 0.0031645569620253164, 0.990506329113924, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[2] <= 0.7848680019378662) {
                                        if (input[1] <= 0.9271479845046997) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.8181818181818182, 0.18181818181818182, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.15789473684210525, 0.0, 0.21052631578947367, 0.631578947368421, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[1] <= 0.8781915009021759) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.18181818181818182, 0.8181818181818182, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.019230769230769232, 0.0, 0.0, 0.9807692307692307, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            } else {
                                if (input[1] <= 1.2559266090393066) {
                                    if (input[0] <= -0.30358950793743134) {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.0, 1.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.1, 0.0, 0.4, 0.5}, 5 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.0, 0.0, 0.0, 1.0, 0.0}, 5 * sizeof(double));
                                }
                            }
                        } else {
                            if (input[1] <= 2.3898775577545166) {
                                if (input[2] <= 0.9533914029598236) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[0] <= -0.034677715972065926) {
                                        memcpy(var0, (double[]){0.09090909090909091, 0.0, 0.0, 0.9090909090909091, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.36363636363636365, 0.0, 0.0, 0.6363636363636364, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            } else {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[0] <= 1.813613474369049) {
            if (input[1] <= -0.3494475185871124) {
                if (input[2] <= 0.33200351893901825) {
                    if (input[1] <= -0.6215659081935883) {
                        if (input[0] <= 0.9108771979808807) {
                            if (input[1] <= -1.65735524892807) {
                                if (input[2] <= -0.5962176322937012) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.9285714285714286, 0.0, 0.0, 0.07142857142857142, 0.0}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[0] <= 0.8110580146312714) {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[1] <= -1.409342348575592) {
                                        memcpy(var0, (double[]){0.7272727272727273, 0.0, 0.0, 0.2727272727272727, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.6, 0.0, 0.0, 0.4, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[1] <= -1.0353348851203918) {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                if (input[0] <= 1.1808800101280212) {
                                    memcpy(var0, (double[]){0.8, 0.0, 0.0, 0.1, 0.1}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            }
                        }
                    } else {
                        memcpy(var0, (double[]){0.5555555555555556, 0.2222222222222222, 0.1111111111111111, 0.0, 0.1111111111111111}, 5 * sizeof(double));
                    }
                } else {
                    if (input[2] <= 1.010796695947647) {
                        if (input[0] <= 1.1184247732162476) {
                            memcpy(var0, (double[]){0.07142857142857144, 0.8571428571428572, 0.0, 0.0, 0.07142857142857144}, 5 * sizeof(double));
                        } else {
                            if (input[1] <= -0.841497153043747) {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                memcpy(var0, (double[]){0.5294117647058824, 0.47058823529411764, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            }
                        }
                    } else {
                        memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                }
            } else {
                if (input[1] <= 1.5429555177688599) {
                    if (input[2] <= -0.7336884438991547) {
                        if (input[1] <= 0.6724252998828888) {
                            if (input[2] <= -1.5341743230819702) {
                                memcpy(var0, (double[]){0.16666666666666666, 0.5, 0.0, 0.0, 0.3333333333333333}, 5 * sizeof(double));
                            } else {
                                if (input[2] <= -1.1160550713539124) {
                                    memcpy(var0, (double[]){0.7142857142857143, 0.0, 0.0, 0.0, 0.2857142857142857}, 5 * sizeof(double));
                                } else {
                                    if (input[0] <= 0.9457866549491882) {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.0, 0.07142857142857142, 0.9285714285714286}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.4, 0.1, 0.0, 0.1, 0.4}, 5 * sizeof(double));
                                    }
                                }
                            }
                        } else {
                            if (input[0] <= 0.9534230828285217) {
                                memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                memcpy(var0, (double[]){0.0, 0.6, 0.0, 0.0, 0.4}, 5 * sizeof(double));
                            }
                        }
                    } else {
                        if (input[2] <= 0.646222323179245) {
                            if (input[1] <= 0.042701153084635735) {
                                if (input[2] <= 0.22155694663524628) {
                                    if (input[2] <= -0.1757485643029213) {
                                        memcpy(var0, (double[]){0.0, 0.18181818181818182, 0.6363636363636364, 0.0, 0.18181818181818182}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.9473684210526315, 0.05263157894736842, 0.0}, 5 * sizeof(double));
                                    }
                                } else {
                                    memcpy(var0, (double[]){0.5, 0.0, 0.2, 0.0, 0.3}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[0] <= 0.9793324172496796) {
                                    if (input[1] <= 0.5571167469024658) {
                                        if (input[0] <= 0.8004215359687805) {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.9830508474576272, 0.01694915254237288, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    } else {
                                        if (input[0] <= 0.8159671425819397) {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.0, 0.0, 0.9090909090909091, 0.09090909090909091, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                } else {
                                    if (input[1] <= 0.15751273185014725) {
                                        memcpy(var0, (double[]){0.0, 0.0, 0.9, 0.0, 0.1}, 5 * sizeof(double));
                                    } else {
                                        if (input[2] <= 0.4959948658943176) {
                                            memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                                        } else {
                                            memcpy(var0, (double[]){0.1, 0.0, 0.9, 0.0, 0.0}, 5 * sizeof(double));
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[2] <= 1.3394507765769958) {
                                if (input[0] <= 1.0966063737869263) {
                                    memcpy(var0, (double[]){0.0, 0.3076923076923077, 0.46153846153846156, 0.23076923076923078, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[1] <= 1.263381838798523) {
                                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.8, 0.0, 0.2, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            } else {
                                memcpy(var0, (double[]){0.0, 1.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            }
                        }
                    }
                } else {
                    if (input[1] <= 1.9679075479507446) {
                        if (input[2] <= 0.612651914358139) {
                            memcpy(var0, (double[]){0.5555555555555556, 0.0, 0.3333333333333333, 0.1111111111111111, 0.0}, 5 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                        }
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                }
            }
        } else {
            if (input[2] <= 0.389241024851799) {
                if (input[1] <= -0.07882020249962807) {
                    if (input[1] <= -0.3879665583372116) {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    } else {
                        if (input[2] <= 0.20510746538639069) {
                            memcpy(var0, (double[]){0.7, 0.0, 0.3, 0.0, 0.0}, 5 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                        }
                    }
                } else {
                    if (input[1] <= 1.9360982775688171) {
                        if (input[0] <= 2.7179863452911377) {
                            if (input[1] <= 0.3284875601530075) {
                                memcpy(var0, (double[]){0.6, 0.0, 0.4, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                if (input[2] <= 0.29289403557777405) {
                                    memcpy(var0, (double[]){0.0, 0.0, 1.0, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.1, 0.0, 0.9, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            }
                        } else {
                            if (input[1] <= 0.44379618763923645) {
                                if (input[0] <= 3.248445987701416) {
                                    memcpy(var0, (double[]){0.9166666666666666, 0.0, 0.08333333333333333, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    memcpy(var0, (double[]){0.7, 0.0, 0.3, 0.0, 0.0}, 5 * sizeof(double));
                                }
                            } else {
                                if (input[0] <= 3.1341720819473267) {
                                    memcpy(var0, (double[]){0.2, 0.0, 0.8, 0.0, 0.0}, 5 * sizeof(double));
                                } else {
                                    if (input[2] <= 0.14065231382846832) {
                                        memcpy(var0, (double[]){0.5, 0.0, 0.5, 0.0, 0.0}, 5 * sizeof(double));
                                    } else {
                                        memcpy(var0, (double[]){0.7333333333333333, 0.0, 0.26666666666666666, 0.0, 0.0}, 5 * sizeof(double));
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[1] <= 2.56507670879364) {
                            memcpy(var0, (double[]){0.9, 0.0, 0.1, 0.0, 0.0}, 5 * sizeof(double));
                        } else {
                            memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                        }
                    }
                }
            } else {
                if (input[2] <= 1.6271491050720215) {
                    if (input[1] <= -0.3327973484992981) {
                        if (input[2] <= 0.860737144947052) {
                            if (input[0] <= 1.957069456577301) {
                                memcpy(var0, (double[]){0.9, 0.1, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            } else {
                                memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                            }
                        } else {
                            memcpy(var0, (double[]){0.8, 0.2, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                        }
                    } else {
                        memcpy(var0, (double[]){1.0, 0.0, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                    }
                } else {
                    memcpy(var0, (double[]){0.2, 0.8, 0.0, 0.0, 0.0}, 5 * sizeof(double));
                }
            }
        }
    }
    memcpy(output, var0, 5 * sizeof(double));
}


// High-level prediction function with normalization
int predict_gesture(float omega_x, float omega_y, float omega_z) {
    float features[3] = {omega_x, omega_y, omega_z};
    float normalized[3];
    
    // Normalize input
    normalize_features(features, normalized);
    
    // Predict using decision tree
    int prediction = score(normalized);
    
    return prediction;
}

// Get gesture name from prediction
const char* get_gesture_name(int prediction) {
    if (prediction >= 0 && prediction < 5) {
        return GESTURE_NAMES[prediction];
    }
    return "unknown";
}

#endif // GESTURE_CLASSIFIER_H
