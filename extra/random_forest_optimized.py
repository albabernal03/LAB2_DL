import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
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
# BÚSQUEDA DE HIPERPARÁMETROS (Opcional - descomentar para buscar mejores params)
# ============================================
SEARCH_PARAMS = False  # Cambiar a True para buscar hiperparámetros (tarda ~15 min)

if SEARCH_PARAMS:
    print("\n🔍 Buscando mejores hiperparámetros (esto puede tardar)...")
    
    param_distributions = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [10, 20, 30, 40],
        'min_samples_leaf': [5, 10, 15, 20],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    random_search = RandomizedSearchCV(
        rf_base,
        param_distributions,
        n_iter=20,  # Número de combinaciones a probar
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ Mejores parámetros encontrados:")
    print(random_search.best_params_)
    print(f"Mejor score CV: {random_search.best_score_:.4f}")
    
    model = random_search.best_estimator_

else:
    # ============================================
    # MODELO OPTIMIZADO (basado en experimentación previa)
    # ============================================
    print("\n🌳 Entrenando Random Forest OPTIMIZADO...")
    model = RandomForestClassifier(
        n_estimators=400,           # Más árboles = mejor generalización
        max_depth=20,               # Limitar profundidad
        min_samples_split=30,       # Más conservador
        min_samples_leaf=15,        # Hojas más grandes
        max_features='sqrt',        # Usar sqrt de features
        class_weight='balanced',    # Balancear clases
        bootstrap=True,             # Usar bootstrap
        oob_score=True,            # Out-of-bag score para validar
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train_scaled, y_train)
    
    if hasattr(model, 'oob_score_'):
        print(f"\n📊 OOB Score (validación interna): {model.oob_score_:.4f}")

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
# ANÁLISIS DETALLADO POR CLASE
# ============================================
print("\n" + "="*50)
print("🔍 ANÁLISIS POR GESTO")
print("="*50)

for i, gesture in enumerate(LABEL_NAMES):
    true_positives = cm[i, i]
    total_true = cm[i, :].sum()
    recall = true_positives / total_true if total_true > 0 else 0
    
    # Principales confusiones
    confused_with = []
    for j, other_gesture in enumerate(LABEL_NAMES):
        if i != j and cm[i, j] > 50:  # Si hay más de 50 confusiones
            confused_with.append(f"{other_gesture} ({cm[i, j]})")
    
    print(f"\n{gesture}:")
    print(f"  Recall: {recall*100:.1f}%")
    if confused_with:
        print(f"  Se confunde con: {', '.join(confused_with)}")

# ============================================
# IMPORTANCIA DE FEATURES
# ============================================
print("\n" + "="*50)
print("🔍 FEATURE IMPORTANCE")
print("="*50)
feature_names = ["ωx", "ωy", "ωz"]
importances = model.feature_importances_

for i, (name, importance) in enumerate(zip(feature_names, importances)):
    print(f"{name}: {importance:.4f} ({'█' * int(importance * 50)})")

# ============================================
# GUARDAR MODELO
# ============================================
os.makedirs('models', exist_ok=True)

print("\n💾 Guardando modelo...")
with open('models/random_forest_optimized.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("✅ Modelo guardado en: models/random_forest_optimized.pkl")
print("✅ Scaler guardado en: models/scaler.pkl")

# ============================================
# VISUALIZACIÓN MEJORADA
# ============================================
print("\n📈 Generando visualización...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Matriz de confusión
im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax1.set_title(f'Confusion Matrix\nTest Accuracy: {acc_test*100:.2f}%', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax1)
tick_marks = np.arange(len(LABEL_NAMES))
ax1.set_xticks(tick_marks)
ax1.set_xticklabels(LABEL_NAMES, rotation=45, ha='right')
ax1.set_yticks(tick_marks)
ax1.set_yticklabels(LABEL_NAMES)

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    ax1.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=11, fontweight='bold')

ax1.set_ylabel('True label', fontsize=12)
ax1.set_xlabel('Predicted label', fontsize=12)

# Importancia de features
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax2.barh(feature_names, importances, color=colors)
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax2.set_xlim([0, max(importances) * 1.1])

for i, (bar, imp) in enumerate(zip(bars, importances)):
    ax2.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('results_optimized.png', dpi=150, bbox_inches='tight')
print("✅ Visualización guardada en: results_optimized.png")

# ============================================
# RESUMEN FINAL
# ============================================
print("\n" + "="*60)
print("📊 RESUMEN DE EVOLUCIÓN")
print("="*60)
print(f"Decision Tree (baseline):        66.95%")
print(f"Random Forest (básico):          72.46%")
print(f"Random Forest (optimizado):      {acc_test*100:.2f}%")
print(f"\nMejora total:                    +{(acc_test - 0.6695)*100:.2f}%")
print(f"Validation Accuracy:             {acc_valid*100:.2f}%")
print(f"Test Accuracy:                   {acc_test*100:.2f}%")
print(f"Diferencia (overfitting):        {abs(acc_valid - acc_test)*100:.2f}%")
print("="*60)

# Probabilidades de predicción (confianza del modelo)
y_pred_proba = model.predict_proba(X_test_scaled)
confidence = np.max(y_pred_proba, axis=1)
avg_confidence = np.mean(confidence)
print(f"\n🎯 Confianza promedio del modelo: {avg_confidence*100:.2f}%")
print(f"Predicciones con >90% confianza: {(confidence > 0.9).sum()} / {len(confidence)} ({(confidence > 0.9).sum()/len(confidence)*100:.1f}%)")