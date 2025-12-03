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
