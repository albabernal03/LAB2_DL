"""
Script para convertir los archivos .pkl existentes (que contienen secuencias temporales)
en formato adecuado para Deep Learning con LSTM/CNN
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================
# CONFIGURACIÓN
# ============================================
BASE_DIR = "."
MAX_SEQUENCE_LENGTH = 150  # Longitud máxima de secuencia

GESTURE_DIRS = [
    "clockwise_dataset",
    "horizontal_swipe_dataset",
    "forward_thrust_dataset",
    "vertical_updown_dataset",
    "wrist_twist_dataset",
]

GESTURE_LABELS = {
    "clockwise": 0,
    "horizontal_swipe": 1,
    "forward_thrust": 2,
    "vertical_updown": 3,
    "wrist_twist": 4,
}

OUTPUT_DIR = "sequences_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# FUNCIONES
# ============================================

def load_sequence_from_pkl(pkl_path):
    """
    Carga un archivo .pkl que contiene un DataFrame con la secuencia temporal
    Returns: numpy array (n_timesteps, 3) con [ωx, ωy, ωz]
    """
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    
    # Extraer las columnas de velocidad angular (sin la columna label)
    sequence = df[['ωx', 'ωy', 'ωz']].values
    
    # Obtener el label (primera fila)
    label = df['label'].iloc[0]
    
    return sequence, label

def pad_or_truncate_sequence(sequence, max_length):
    """
    Ajusta la secuencia a una longitud fija
    """
    n_timesteps, n_features = sequence.shape
    
    if n_timesteps > max_length:
        # Truncar desde el centro (mantener inicio y fin)
        start = (n_timesteps - max_length) // 2
        return sequence[start:start + max_length, :]
    elif n_timesteps < max_length:
        # Padding con ceros al final
        padding = np.zeros((max_length - n_timesteps, n_features))
        return np.vstack([sequence, padding])
    else:
        return sequence

def load_all_sequences(base_dir, gesture_dirs, max_length=150):
    """
    Carga todas las secuencias de todos los splits (train/valid/test)
    """
    X_train_list = []
    y_train_list = []
    
    X_valid_list = []
    y_valid_list = []
    
    X_test_list = []
    y_test_list = []
    
    print("="*60)
    print("📂 CARGANDO SECUENCIAS TEMPORALES")
    print("="*60)
    
    for gesture_dir in gesture_dirs:
        gesture_path = os.path.join(base_dir, gesture_dir)
        
        if not os.path.exists(gesture_path):
            print(f"⚠️  {gesture_dir} no existe, saltando...")
            continue
        
        gesture_name = gesture_dir.replace('_dataset', '')
        label = GESTURE_LABELS.get(gesture_name, -1)
        
        print(f"\n📁 {gesture_name} (label={label})")
        
        # Procesar cada split
        for split, X_list, y_list in [
            ('train', X_train_list, y_train_list),
            ('valid', X_valid_list, y_valid_list),
            ('test', X_test_list, y_test_list)
        ]:
            split_path = os.path.join(gesture_path, split)
            
            if not os.path.exists(split_path):
                print(f"  ⚠️  {split} no existe")
                continue
            
            pkl_files = sorted([f for f in os.listdir(split_path) if f.endswith('.pkl')])
            
            for pkl_file in pkl_files:
                pkl_path = os.path.join(split_path, pkl_file)
                
                # Cargar secuencia
                sequence, seq_label = load_sequence_from_pkl(pkl_path)
                original_length = len(sequence)
                
                # Ajustar longitud
                sequence = pad_or_truncate_sequence(sequence, max_length)
                
                X_list.append(sequence)
                y_list.append(label)
                
                print(f"  ✓ {split}/{pkl_file}: {original_length} → {max_length} timesteps")
    
    # Convertir a numpy arrays
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    X_valid = np.array(X_valid_list)
    y_valid = np.array(y_valid_list)
    
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    print("\n" + "="*60)
    print("📊 RESUMEN DE DATOS CARGADOS")
    print("="*60)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Distribución de clases
    print("\n📊 Distribución de clases:")
    for split_name, y_data in [('Train', y_train), ('Valid', y_valid), ('Test', y_test)]:
        print(f"\n{split_name}:")
        unique, counts = np.unique(y_data, return_counts=True)
        for label_id, count in zip(unique, counts):
            label_name = [k for k, v in GESTURE_LABELS.items() if v == label_id][0]
            print(f"  {label_name} (label {label_id}): {count} samples")
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def normalize_sequences(X_train, X_valid, X_test):
    """
    Normaliza las secuencias usando StandardScaler
    """
    print("\n⚙️  Normalizando secuencias...")
    
    # Reshape para normalizar: (n_samples * n_timesteps, n_features)
    n_train_samples, n_timesteps, n_features = X_train.shape
    n_valid_samples = X_valid.shape[0]
    n_test_samples = X_test.shape[0]
    
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_valid_reshaped = X_valid.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    # Fit scaler en train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_valid_scaled = scaler.transform(X_valid_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape de vuelta a secuencias
    X_train_scaled = X_train_scaled.reshape(n_train_samples, n_timesteps, n_features)
    X_valid_scaled = X_valid_scaled.reshape(n_valid_samples, n_timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_test_samples, n_timesteps, n_features)
    
    print("✅ Normalización completa")
    
    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    
    # Cargar todas las secuencias
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_all_sequences(
        BASE_DIR, 
        GESTURE_DIRS,
        max_length=MAX_SEQUENCE_LENGTH
    )
    
    # Verificar que tenemos datos
    if len(X_train) == 0:
        print("\n❌ ERROR: No se cargaron datos. Verifica que los directorios existen.")
        exit(1)
    
    # Normalizar
    X_train_scaled, X_valid_scaled, X_test_scaled, scaler = normalize_sequences(
        X_train, X_valid, X_test
    )
    
    # Guardar datos procesados
    print(f"\n💾 Guardando datos procesados en {OUTPUT_DIR}/...")
    
    # Guardar cada split
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_valid.npy'), X_valid_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_valid.npy'), y_valid)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)
    
    # Guardar scaler
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Datos guardados:")
    print(f"   - X_train.npy: {X_train_scaled.shape}")
    print(f"   - X_valid.npy: {X_valid_scaled.shape}")
    print(f"   - X_test.npy: {X_test_scaled.shape}")
    print(f"   - scaler.pkl")
    
    # Estadísticas finales
    print("\n📊 Estadísticas de las secuencias normalizadas:")
    print(f"   Media: {X_train_scaled.mean():.4f}")
    print(f"   Std: {X_train_scaled.std():.4f}")
    print(f"   Min: {X_train_scaled.min():.4f}")
    print(f"   Max: {X_train_scaled.max():.4f}")
    
    print("\n" + "="*60)
    print("✅ ¡LISTO! Ahora puedes entrenar modelos de Deep Learning")
    print("="*60)
    print("\nPróximo paso: ejecutar train_deep_learning.py")