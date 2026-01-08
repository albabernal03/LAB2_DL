import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Configuración
BASE_DIR = "."

GESTURE_DIRS = [
    "clockwise_dataset",
    "horizontal_swipe_dataset",
    "forward_thrust_dataset",
    "vertical_updown_dataset",
    "wrist_twist_dataset",
]

GESTURE_LABELS = {
    "clockwise_dataset": 0,
    "horizontal_swipe_dataset": 1,
    "forward_thrust_dataset": 2,
    "vertical_updown_dataset": 3,
    "wrist_twist_dataset": 4,
}

LABEL_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

def load_all_pkl_files(gesture_dir, split):
    """
    Carga todos los archivos .pkl de una carpeta (train/valid/test)
    """
    split_path = os.path.join(BASE_DIR, gesture_dir, split)
    
    if not os.path.exists(split_path):
        print(f"⚠️  No existe: {split_path}")
        return None
    
    all_data = []
    pkl_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        file_path = os.path.join(split_path, pkl_file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)
    
    if len(all_data) == 0:
        return None
    
    # Concatenar todos los arrays
    return np.vstack(all_data)

def load_dataset_split(split):
    """
    Carga train, valid o test de TODOS los gestos
    """
    X_list = []
    y_list = []
    
    for gesture_dir in GESTURE_DIRS:
        data = load_all_pkl_files(gesture_dir, split)
        
        if data is not None:
            # Separar features (columnas 0-2) y label (columna 3)
            X = data[:, :-1]  # Todas menos la última
            y = data[:, -1]   # Última columna
            
            X_list.append(X)
            y_list.append(y)
            
            print(f"  ✓ {gesture_dir}/{split}: {X.shape[0]} muestras")
    
    if len(X_list) == 0:
        return None, None
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    return X, y

# ============================================
# CARGAR DATOS
# ============================================
print("📂 Cargando datos...")
print("\n--- TRAIN ---")
X_train, y_train = load_dataset_split('train')

print("\n--- VALID ---")
X_valid, y_valid = load_dataset_split('valid')

print("\n--- TEST ---")
X_test, y_test = load_dataset_split('test')

if X_train is None or X_valid is None or X_test is None:
    print("❌ Error: No se pudieron cargar todos los splits")
    exit(1)

print(f"\n📊 Tamaños finales:")
print(f"  Train: {X_train.shape}")
print(f"  Valid: {X_valid.shape}")
print(f"  Test:  {X_test.shape}")

# ============================================
# NORMALIZACIÓN
# ============================================
print("\n⚙️  Normalizando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# ============================================
# ENTRENAMIENTO RANDOM FOREST
# ============================================
print("\n🌳 Entrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,           # 200 árboles
    max_depth=15,               # Profundidad máxima
    min_samples_split=20,       # Mínimo para dividir nodo
    min_samples_leaf=10,        # Mínimo en hojas
    class_weight='balanced',    # Balancear clases
    random_state=42,
    n_jobs=-1,                  # Usar todos los cores
    verbose=1                   # Mostrar progreso
)

model.fit(X_train_scaled, y_train)

# ============================================
# EVALUACIÓN EN VALIDATION
# ============================================
print("\n" + "="*50)
print("📊 VALIDATION SET")
print("="*50)
y_pred_valid = model.predict(X_valid_scaled)
acc_valid = accuracy_score(y_valid, y_pred_valid)
print(f"Accuracy: {acc_valid:.4f} ({acc_valid*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_valid, y_pred_valid, target_names=LABEL_NAMES))

# ============================================
# EVALUACIÓN EN TEST
# ============================================
print("\n" + "="*50)
print("📊 TEST SET")
print("="*50)
y_pred_test = model.predict(X_test_scaled)
acc_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {acc_test:.4f} ({acc_test*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=LABEL_NAMES))

print("\n📋 Confusion Matrix (Test):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# ============================================
# ANÁLISIS DE IMPORTANCIA DE FEATURES
# ============================================
print("\n" + "="*50)
print("🔍 FEATURE IMPORTANCE")
print("="*50)
feature_names = ["ωx", "ωy", "ωz"]
importances = model.feature_importances_

for i, (name, importance) in enumerate(zip(feature_names, importances)):
    print(f"{name}: {importance:.4f}")

# ============================================
# GUARDAR MODELO Y SCALER
# ============================================
os.makedirs('models', exist_ok=True)

print("\n💾 Guardando modelo...")
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("✅ Modelo guardado en: models/random_forest_model.pkl")
print("✅ Scaler guardado en: models/scaler.pkl")

# ============================================
# VISUALIZACIÓN DE MATRIZ DE CONFUSIÓN
# ============================================
print("\n📈 Generando visualización...")
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Set')
plt.colorbar()
tick_marks = np.arange(len(LABEL_NAMES))
plt.xticks(tick_marks, LABEL_NAMES, rotation=45, ha='right')
plt.yticks(tick_marks, LABEL_NAMES)

# Añadir valores en cada celda
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Matriz guardada en: confusion_matrix.png")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*50)
print("📊 RESUMEN FINAL")
print("="*50)
print(f"Validation Accuracy: {acc_valid*100:.2f}%")
print(f"Test Accuracy:       {acc_test*100:.2f}%")
print(f"Diferencia (overfitting): {(acc_valid - acc_test)*100:.2f}%")
print("="*50)