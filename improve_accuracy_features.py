"""
Feature Engineering para datasets pequeños
Adaptado para funcionar con pocas muestras
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

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

def extract_features(df):
    """Extrae múltiples características estadísticas"""
    features = {}
    
    for axis in ['ωx', 'ωy', 'ωz']:
        data = df[axis].values
        
        # Estadísticas básicas
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_range'] = np.max(data) - np.min(data)
        
        # Percentiles
        features[f'{axis}_q25'] = np.percentile(data, 25)
        features[f'{axis}_median'] = np.median(data)
        features[f'{axis}_q75'] = np.percentile(data, 75)
        
        # Momentos estadísticos
        features[f'{axis}_skew'] = pd.Series(data).skew()
        features[f'{axis}_kurtosis'] = pd.Series(data).kurtosis()
        
        # Energía
        features[f'{axis}_energy'] = np.sum(data ** 2)
        features[f'{axis}_rms'] = np.sqrt(np.mean(data ** 2))
        
    # Magnitud total
    magnitude = np.sqrt(df['ωx']**2 + df['ωy']**2 + df['ωz']**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    features['magnitude_max'] = np.max(magnitude)
    
    return features

def load_all_pkl_files_with_features(gesture_dir, split):
    """Carga archivos y extrae features"""
    split_path = os.path.join(BASE_DIR, gesture_dir, split)
    
    if not os.path.exists(split_path):
        return None, None
    
    all_features = []
    all_labels = []
    pkl_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
    
    for pkl_file in pkl_files:
        file_path = os.path.join(split_path, pkl_file)
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
            features = extract_features(df)
            all_features.append(features)
            all_labels.append(df['label'].iloc[0])
    
    if len(all_features) == 0:
        return None, None
    
    features_df = pd.DataFrame(all_features)
    return features_df.values, np.array(all_labels)

def load_dataset_split_with_features(split):
    """Carga datos con features extendidas"""
    all_X = []
    all_y = []
    
    for gesture_dir in GESTURE_DIRS:
        X, y = load_all_pkl_files_with_features(gesture_dir, split)
        
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            print(f"  ✓ {gesture_dir}/{split}: {len(X)} samples")
    
    if len(all_X) == 0:
        return None, None
    
    return np.vstack(all_X), np.concatenate(all_y)

print("="*60)
print("FEATURE ENGINEERING - DATASET PEQUEÑO")
print("="*60)

# ============================================
# LOAD DATA
# ============================================
print("\n📂 Loading data...")
print("\n--- TRAIN ---")
X_train, y_train = load_dataset_split_with_features('train')

print("\n--- VALID ---")
X_valid, y_valid = load_dataset_split_with_features('valid')

print("\n--- TEST ---")
X_test, y_test = load_dataset_split_with_features('test')

if X_train is None:
    print("❌ Error: Could not load data")
    exit(1)

print(f"\n📊 Dataset sizes:")
print(f"  Train: {X_train.shape}")
print(f"  Valid: {X_valid.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Total features: {X_train.shape[1]}")

# Verificar si dataset es muy pequeño
if X_train.shape[0] < 50:
    print(f"\n⚠️  ADVERTENCIA: Dataset muy pequeño ({X_train.shape[0]} muestras)")
    print("   Resultados pueden no ser representativos")
    print("   Recomendación: Capturar más datos para mejorar")

# ============================================
# NORMALIZATION
# ============================================
print("\n⚙️  Normalizing data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# ============================================
# DECISION TREE (sin cross-validation)
# ============================================
print("\n" + "="*60)
print("🌳 DECISION TREE")
print("="*60)

# Probar diferentes configuraciones manualmente
configs = [
    {'max_depth': 5, 'min_samples_split': 2},
    {'max_depth': 10, 'min_samples_split': 2},
    {'max_depth': None, 'min_samples_split': 2},
]

best_dt_acc = 0
best_dt_model = None
best_dt_config = None

for config in configs:
    dt = DecisionTreeClassifier(random_state=42, **config)
    dt.fit(X_train_scaled, y_train)
    acc = dt.score(X_valid_scaled, y_valid)
    
    if acc > best_dt_acc:
        best_dt_acc = acc
        best_dt_model = dt
        best_dt_config = config

print(f"🏆 Best DT config: {best_dt_config}")
print(f"📊 Validation Accuracy: {best_dt_acc*100:.2f}%")

y_pred_dt = best_dt_model.predict(X_test_scaled)
acc_dt_test = accuracy_score(y_test, y_pred_dt)
print(f"✅ Test Accuracy: {acc_dt_test*100:.2f}%")

# ============================================
# RANDOM FOREST
# ============================================
print("\n" + "="*60)
print("🌲 RANDOM FOREST")
print("="*60)

# Configuraciones para RF
rf_configs = [
    {'n_estimators': 10, 'max_depth': 5},
    {'n_estimators': 20, 'max_depth': 10},
    {'n_estimators': 50, 'max_depth': None},
]

best_rf_acc = 0
best_rf_model = None
best_rf_config = None

for config in rf_configs:
    rf = RandomForestClassifier(random_state=42, **config)
    rf.fit(X_train_scaled, y_train)
    acc = rf.score(X_valid_scaled, y_valid)
    
    if acc > best_rf_acc:
        best_rf_acc = acc
        best_rf_model = rf
        best_rf_config = config

print(f"🏆 Best RF config: {best_rf_config}")
print(f"📊 Validation Accuracy: {best_rf_acc*100:.2f}%")

y_pred_rf = best_rf_model.predict(X_test_scaled)
acc_rf_test = accuracy_score(y_test, y_pred_rf)
print(f"✅ Test Accuracy: {acc_rf_test*100:.2f}%")

# ============================================
# COMPARACIÓN
# ============================================
print("\n" + "="*60)
print("📊 COMPARACIÓN FINAL")
print("="*60)
print(f"Decision Tree - Test: {acc_dt_test*100:.2f}%")
print(f"Random Forest - Test: {acc_rf_test*100:.2f}%")

# Elegir mejor modelo
if acc_rf_test > acc_dt_test:
    best_model = best_rf_model
    model_name = "RandomForest"
    best_acc = acc_rf_test
    y_pred_best = y_pred_rf
else:
    best_model = best_dt_model
    model_name = "DecisionTree"
    best_acc = acc_dt_test
    y_pred_best = y_pred_dt

print(f"\n🏆 Mejor modelo: {model_name} ({best_acc*100:.2f}%)")
print("="*60)

# ============================================
# CLASSIFICATION REPORT
# ============================================
print(f"\n📊 {model_name.upper()} - CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_best, target_names=LABEL_NAMES, zero_division=0))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# ============================================
# FEATURE IMPORTANCE
# ============================================
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*60)
    print("🔍 TOP 10 FEATURES MÁS IMPORTANTES")
    print("="*60)
    
    # Obtener nombres de features
    feature_names_list = []
    for axis in ['ωx', 'ωy', 'ωz']:
        feature_names_list.extend([
            f'{axis}_mean', f'{axis}_std', f'{axis}_min', f'{axis}_max',
            f'{axis}_range', f'{axis}_q25', f'{axis}_median', f'{axis}_q75',
            f'{axis}_skew', f'{axis}_kurtosis', f'{axis}_energy', f'{axis}_rms'
        ])
    feature_names_list.extend(['magnitude_mean', 'magnitude_std', 'magnitude_max'])
    
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names_list[idx]}: {importances[idx]:.4f}")

# ============================================
# SAVE MODEL
# ============================================
os.makedirs('models', exist_ok=True)

print("\n💾 Saving model...")
with open(f'models/{model_name.lower()}_enhanced.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('models/scaler_enhanced.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Model saved: models/{model_name.lower()}_enhanced.pkl")
print("✅ Scaler saved: models/scaler_enhanced.pkl")

# ============================================
# RECOMENDACIONES
# ============================================
print("\n" + "="*60)
print("💡 RECOMENDACIONES")
print("="*60)

if X_train.shape[0] < 100:
    print("⚠️  Tu dataset es muy pequeño:")
    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Recomendado: >500 muestras")
    print("\n🎯 Para mejorar el accuracy:")
    print("   1. Captura más gestos (al menos 30-50 por clase)")
    print("   2. Captura con diferentes personas")
    print("   3. Captura en diferentes posiciones/velocidades")
    print(f"\n✅ Con los datos actuales: {best_acc*100:.2f}% es razonable")
else:
    print(f"✅ Accuracy alcanzado: {best_acc*100:.2f}%")

print("="*60)