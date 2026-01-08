import os
import pickle

print("="*60)
print("CONTEO DE ARCHIVOS ORIGINALES")
print("="*60)

GESTURE_DIRS = [
    "clockwise_dataset",
    "horizontal_swipe_dataset",
    "forward_thrust_dataset",
    "vertical_updown_dataset",
    "wrist_twist_dataset",
]

total_files = 0
for gesture_dir in GESTURE_DIRS:
    print(f"\n📁 {gesture_dir}:")
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(gesture_dir, split)
        if os.path.exists(split_path):
            pkl_files = [f for f in os.listdir(split_path) if f.endswith('.pkl')]
            num_files = len(pkl_files)
            total_files += num_files
            print(f"  {split}: {num_files} archivos .pkl")
            
            # Contar muestras en cada archivo
            if num_files > 0:
                first_file = os.path.join(split_path, pkl_files[0])
                with open(first_file, 'rb') as f:
                    df = pickle.load(f)
                    print(f"    → Cada archivo tiene ~{len(df)} timesteps")

print(f"\n{'='*60}")
print(f"📊 TOTAL: {total_files} archivos .pkl disponibles")
print(f"{'='*60}")

print(f"\n🤔 PERO el script de secuencias solo usó:")
print(f"   - Train: 20 secuencias (4 por gesto)")
print(f"   - Valid: 5 secuencias (1 por gesto)")
print(f"   - Test: 5 secuencias (1 por gesto)")

print(f"\n💡 PROBLEMA:")
print(f"   El script 'prepare_sequences_from_pkl.py' probablemente:")
print(f"   1. Solo tomó 1 archivo por gesto por split")
print(f"   2. O solo extrajo 1 secuencia por archivo")
print(f"   3. Necesitas revisar ese script para usar TODOS los datos")
