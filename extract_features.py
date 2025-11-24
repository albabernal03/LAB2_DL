import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

LABEL_NAMES = [
    'clockwise',
    'horizontal_swipe',
    'forward_thrust',
    'vertical_updown',
    'wrist_twist'
]

def load_all_pkl_files(gesture_dir, split):
    """Carga todos los archivos .pkl de una carpeta"""
    split_path = os.path.join(BASE_DIR, gesture_dir, split)
    
    if not os.path.exists(split_path):
        print(f"⚠️  No existe: {split_path}")
        return None
    
    all_data = []
    pkl_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        file_path = os.path.join(split_path, pkl_file)
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            all_data.append(df)
    
    if len(all_data) == 0:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def load_dataset_split(split):
    """Carga train, valid o test de TODOS los gestos"""
    all_dfs = []
    
    for gesture_dir in GESTURE_DIRS:
        df = load_all_pkl_files(gesture_dir, split)
        
        if df is not None:
            all_dfs.append(df)
            print(f"  ✓ {gesture_dir}/{split}: {len(df)} muestras")
    
    if len(all_dfs) == 0:
        return None, None
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    X = combined_df[['ωx', 'ωy', 'ωz']].values
    y = combined_df['label'].values
    
    return X, y

def create_engineered_features(X):
    """
    Crea features adicionales a partir de ωx, ωy, ωz
    CON MANEJO ROBUSTO DE NaN e Inf
    """
    X_new = []
    
    # Features originales
    X_new.append(X)
    
    # 1. MAGNITUD (norma del vector)
    magnitude = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    X_new.append(magnitude)
    
    # 2. ÁNGULOS (dirección del vector) - con manejo seguro
    epsilon = 1e-10
    
    # Ángulo en plano XY
    angle_xy = np.arctan2(X[:, 1], X[:, 0] + epsilon).reshape(-1, 1)
    X_new.append(angle_xy)
    
    # Ángulo en plano XZ
    angle_xz = np.arctan2(X[:, 2], X[:, 0] + epsilon).reshape(-1, 1)
    X_new.append(angle_xz)
    
    # Ángulo en plano YZ
    angle_yz = np.arctan2(X[:, 2], X[:, 1] + epsilon).reshape(-1, 1)
    X_new.append(angle_yz)
    
    # 3. RATIOS (relaciones entre ejes) - con manejo robusto
    # Usar clip para evitar divisiones extremas
    X_clipped = np.clip(X, -1e6, 1e6)
    
    ratio_xy = np.divide(X_clipped[:, 0], X_clipped[:, 1] + epsilon)
    ratio_xz = np.divide(X_clipped[:, 0], X_clipped[:, 2] + epsilon)
    ratio_yz = np.divide(X_clipped[:, 1], X_clipped[:, 2] + epsilon)
    
    # Clip los ratios a rangos razonables
    ratio_xy = np.clip(ratio_xy, -100, 100).reshape(-1, 1)
    ratio_xz = np.clip(ratio_xz, -100, 100).reshape(-1, 1)
    ratio_yz = np.clip(ratio_yz, -100, 100).reshape(-1, 1)
    
    X_new.append(ratio_xy)
    X_new.append(ratio_xz)
    X_new.append(ratio_yz)
    
    # 4. PRODUCTOS (interacciones entre ejes)
    prod_xy = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    prod_xz = (X[:, 0] * X[:, 2]).reshape(-1, 1)
    prod_yz = (X[:, 1] * X[:, 2]).reshape(-1, 1)
    X_new.append(prod_xy)
    X_new.append(prod_xz)
    X_new.append(prod_yz)
    
    # 5. VALORES ABSOLUTOS (para detectar rotaciones simétricas)
    abs_values = np.abs(X)
    X_new.append(abs_values)
    
    # Concatenar todo
    X_engineered = np.hstack(X_new)
    
    # VERIFICACIÓN Y LIMPIEZA FINAL
    # Reemplazar cualquier NaN o Inf restante
    X_engineered = np.nan_to_num(X_engineered, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_engineered

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

print(f"\n📊 Tamaños originales:")
print(f"  Train: {X_train.shape}")
print(f"  Valid: {X_valid.shape}")
print(f"  Test:  {X_test.shape}")

# ============================================
# FEATURE ENGINEERING
# ============================================
print("\n🔧 Creando features adicionales...")
X_train_eng = create_engineered_features(X_train)
X_valid_eng = create_engineered_features(X_valid)
X_test_eng = create_engineered_features(X_test)

print(f"\n📊 Tamaños con features nuevas:")
print(f"  Train: {X_train_eng.shape} (antes: {X_train.shape})")
print(f"  Valid: {X_valid_eng.shape}")
print(f"  Test:  {X_test_eng.shape}")

# Verificación de que no hay NaN
print(f"\n🔍 Verificando datos limpios...")
print(f"  NaN en train: {np.isnan(X_train_eng).sum()}")
print(f"  Inf en train: {np.isinf(X_train_eng).sum()}")

# ============================================
# NORMALIZACIÓN
# ============================================
print("\n⚙️  Normalizando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_valid_scaled = scaler.transform(X_valid_eng)
X_test_scaled = scaler.transform(X_test_eng)

# ============================================
# MODELO: GRADIENT BOOSTING
# ============================================
print("\n🚀 Entrenando Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
    verbose=1
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
print("🔍 TOP 10 FEATURES MÁS IMPORTANTES")
print("="*50)
feature_names = [
    'ωx', 'ωy', 'ωz',           # Originales
    'magnitude',                 # Magnitud
    'angle_xy', 'angle_xz', 'angle_yz',  # Ángulos
    'ratio_xy', 'ratio_xz', 'ratio_yz',  # Ratios
    'prod_xy', 'prod_xz', 'prod_yz',     # Productos
    'abs_ωx', 'abs_ωy', 'abs_ωz'         # Valores absolutos
]

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(10, len(feature_names))):
    idx = indices[i]
    if idx < len(feature_names):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    else:
        print(f"{i+1}. Feature_{idx}: {importances[idx]:.4f}")

# ============================================
# GUARDAR MODELO Y SCALER
# ============================================
os.makedirs('models', exist_ok=True)

print("\n💾 Guardando modelo...")
with open('models/gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/scaler_engineered.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("✅ Modelo guardado en: models/gradient_boosting_model.pkl")
print("✅ Scaler guardado en: models/scaler_engineered.pkl")

# ============================================
# VISUALIZACIÓN
# ============================================
print("\n📈 Generando visualización...")
plt.figure(figsize=(12, 9))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - Test Set (Accuracy: {acc_test*100:.2f}%)', fontsize=14)
plt.colorbar()
tick_marks = np.arange(len(LABEL_NAMES))
plt.xticks(tick_marks, LABEL_NAMES, rotation=45, ha='right')
plt.yticks(tick_marks, LABEL_NAMES)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=10)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix_engineered.png', dpi=150, bbox_inches='tight')
print("✅ Matriz guardada en: confusion_matrix_engineered.png")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*50)
print("📊 RESUMEN FINAL")
print("="*50)
print(f"Baseline (Decision Tree):    66.95%")
print(f"Random Forest:               72.46%")
print(f"Gradient Boosting + Features: {acc_test*100:.2f}%")
print(f"\nValidation Accuracy: {acc_valid*100:.2f}%")
print(f"Test Accuracy:       {acc_test*100:.2f}%")
print(f"Diferencia:          {abs(acc_valid - acc_test)*100:.2f}%")
print("="*50)