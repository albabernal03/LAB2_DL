import numpy as np
import pickle

# Cargar datos
print("="*60)
print("ANÁLISIS DE DATOS - VERIFICACIÓN DE 100% ACCURACY")
print("="*60)

X_train = np.load('sequences_processed/X_train.npy')
y_train = np.load('sequences_processed/y_train.npy')

X_valid = np.load('sequences_processed/X_valid.npy')
y_valid = np.load('sequences_processed/y_valid.npy')

X_test = np.load('sequences_processed/X_test.npy')
y_test = np.load('sequences_processed/y_test.npy')

print(f"\n📊 TAMAÑOS DE DATOS:")
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

print(f"\n🔍 DISTRIBUCIÓN DE CLASES:")
print(f"\nTrain:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Clase {u}: {c} muestras")

print(f"\nValid:")
unique, counts = np.unique(y_valid, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Clase {u}: {c} muestras")

print(f"\nTest:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Clase {u}: {c} muestras")

print(f"\n⚠️  ANÁLISIS:")
total_test = len(y_test)
print(f"Total test samples: {total_test}")

if total_test < 50:
    print(f"\n🚨 PROBLEMA DETECTADO:")
    print(f"   El test set tiene solo {total_test} muestras!")
    print(f"   Con {len(np.unique(y_test))} clases, eso es ~{total_test/len(np.unique(y_test)):.1f} muestras por clase")
    print(f"   100% accuracy con tan pocas muestras NO es estadísticamente significativo")
    print(f"\n   Esto explica por qué los modelos alcanzan 100%:")
    print(f"   - Dataset muy pequeño")
    print(f"   - Fácil de memorizar")
    print(f"   - No representa bien la variabilidad real")
else:
    print(f"\n✅ El test set tiene {total_test} muestras, tamaño razonable")

print(f"\n📈 ESTADÍSTICAS DE FEATURES:")
print(f"Shape de cada secuencia: {X_train.shape[1:]} (timesteps, features)")
print(f"Rango de valores en train:")
print(f"  Min: {X_train.min():.4f}")
print(f"  Max: {X_train.max():.4f}")
print(f"  Mean: {X_train.mean():.4f}")
print(f"  Std: {X_train.std():.4f}")

# Verificar si hay datos idénticos
print(f"\n🔍 VERIFICACIÓN DE DUPLICADOS:")
print(f"¿Hay secuencias duplicadas en train? ", end="")
train_reshaped = X_train.reshape(X_train.shape[0], -1)
unique_train = np.unique(train_reshaped, axis=0)
print(f"{'SÍ' if len(unique_train) < len(X_train) else 'NO'}")
print(f"  Secuencias únicas: {len(unique_train)} de {len(X_train)}")

print(f"\n🎯 CONCLUSIÓN:")
if total_test < 50:
    print(f"⚠️  El 100% accuracy es ENGAÑOSO debido a:")
    print(f"   1. Test set extremadamente pequeño ({total_test} muestras)")
    print(f"   2. Solo ~1 muestra por clase en test")
    print(f"   3. Los modelos pueden estar memorizando en lugar de generalizar")
    print(f"\n💡 RECOMENDACIÓN:")
    print(f"   - Necesitas MÁS datos de test para validar correctamente")
    print(f"   - Idealmente 100-200 muestras por clase")
    print(f"   - Considera usar cross-validation en todo el dataset")
else:
    print(f"✅ El tamaño del test set es adecuado")
    print(f"   El 100% accuracy es más confiable")
