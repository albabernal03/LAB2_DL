# LAB2_DELIVERABLES - Instrucciones de Ejecución

## ⚠️ IMPORTANTE: Cómo ejecutar los scripts

### Estructura del repositorio

```
LAB2_DL/
├── LAB2_DELIVERABLES/          ← Carpeta de entrega
│   ├── README.md
│   ├── INDEX.md
│   ├── QUICK_START.md
│   ├── reports/
│   │   └── COMPREHENSIVE_REPORT.md  ⭐ LEER PRIMERO
│   ├── optimized_models/
│   │   ├── train_neural_network_optimized.py
│   │   ├── train_lstm_optimized.py
│   │   ├── train_cnn_optimized.py
│   │   ├── train_hybrid_optimized.py
│   │   └── compare_all_models.py
│   └── baseline_models/
│       ├── train_lstm_cnn.py
│       ├── train_decision_tree_optimized.py
│       └── train_random_forest.py
│
├── clockwise_dataset/          ← Datos necesarios
├── horizontal_swipe_dataset/
├── forward_thrust_dataset/
├── vertical_updown_dataset/
├── wrist_twist_dataset/
└── extra/                      ← Archivos auxiliares (no necesarios)
```

---

## 📖 Para Evaluar (SIN ejecutar código)

**Leer estos documentos en orden:**

1. **`README.md`** (5 min) - Vista general
2. **`reports/COMPREHENSIVE_REPORT.md`** (20-30 min) ⭐ **DOCUMENTO PRINCIPAL**
3. **`INDEX.md`** (opcional) - Índice de archivos

El **COMPREHENSIVE_REPORT.md** contiene:
- ✅ Explicación de las 4 arquitecturas (FNN, LSTM, CNN, Hybrid)
- ✅ Motivación de cada elección
- ✅ Estrategia de búsqueda de hiperparámetros (1,533 combinaciones)
- ✅ Justificación científica de RandomizedSearch (Bergstra & Bengio, 2012)
- ✅ 7 técnicas anti-overfitting implementadas
- ✅ Análisis completo de resultados

---

## 🚀 Para Ejecutar los Scripts

### Requisitos Previos

```bash
# 1. Clonar el repositorio
git clone https://github.com/albabernal03/LAB2_DL.git
cd LAB2_DL

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno (Windows)
venv\Scripts\activate

# 4. Instalar dependencias
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

### Ejecución

**IMPORTANTE:** Los scripts deben ejecutarse desde la carpeta `optimized_models`:

```bash
cd LAB2_DELIVERABLES/optimized_models

# Ejecutar modelos (uno a uno)
python train_neural_network_optimized.py  # FNN: 30-60 min
python train_lstm_optimized.py            # LSTM: 20-30 min
python train_cnn_optimized.py             # CNN: 30-45 min
python train_hybrid_optimized.py          # Hybrid: 2-4 horas

# Comparar resultados (después de entrenar todos)
python compare_all_models.py
```

### Dónde se guardan los resultados

Los scripts crean automáticamente estas carpetas:

```
LAB2_DELIVERABLES/optimized_models/
├── models/                    ← Modelos entrenados (.keras, .pkl)
└── (los scripts buscan datos en ../../*_dataset/)
```

Y también usan:
```
LAB2_DL/
├── models_dl/                 ← Resultados de LSTM/CNN/Hybrid
└── figures/                   ← Visualizaciones
```

---

## ✅ Verificación Rápida

Para verificar que todo funciona sin entrenar (test rápido):

```bash
cd LAB2_DELIVERABLES/optimized_models
python -c "import tensorflow as tf; import sklearn; print('✅ Dependencias OK')"
```

---

## 📊 Resultados Esperados

Después de ejecutar todos los scripts:

### Modelos guardados
- `models/neural_network_optimized.keras`
- `../../models_dl/LSTM_optimized_best.keras`
- `../../models_dl/CNN1D_optimized_best.keras`
- `../../models_dl/Hybrid_optimized_best.keras`

### Resultados de búsqueda
- `models/nn_hyperparameter_results.pkl` (FNN: 576 combinaciones)
- `../../models_dl/lstm_hyperparameter_results.pkl` (LSTM: 50 combinaciones)
- `../../models_dl/cnn_hyperparameter_results.pkl` (CNN: 75 combinaciones)
- `../../models_dl/hybrid_hyperparameter_results.pkl` (Hybrid: 512 combinaciones)

### Visualizaciones
- `../../figures/model_comparison_summary.png`
- `../../figures/l2_regularization_analysis.png`
- `../../figures/overfitting_analysis_all_models.png`

---

## ⏱️ Tiempos de Ejecución

| Script | Combinaciones | Tiempo (CPU) | Tiempo (GPU) |
|--------|---------------|--------------|--------------|
| FNN | 576 | 30-60 min | 10-20 min |
| LSTM | 50 | 20-30 min | 5-10 min |
| CNN | 75 | 30-45 min | 10-15 min |
| Hybrid | 512 | 2-4 hours | 45-90 min |
| **TOTAL** | **1,533** | **4-6 hours** | **1.5-2.5 hours** |

---

## 🆘 Solución de Problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: "No such file or directory: '../../clockwise_dataset'"
**Solución:** Asegúrate de ejecutar desde `LAB2_DELIVERABLES/optimized_models/`

### Error: "No module named 'seaborn'"
```bash
pip install seaborn
```

---

## 📧 Contacto

**Autor:** Alba Bernal  
**Repositorio:** https://github.com/albabernal03/LAB2_DL  
**Fecha:** Enero 2026

---

## 🎯 Summary for Evaluators

**To evaluate WITHOUT executing:**
- Read `reports/COMPREHENSIVE_REPORT.md` (main document)
- Review code in `optimized_models/` (well-commented)

**To execute (optional):**
- Follow "Execution" instructions above
- Total time: 4-6 hours (CPU) or 1.5-2.5 hours (GPU)

**All lab requirements met:**
- ✅ 4 types of neural networks (FNN, LSTM, CNN, Hybrid)
- ✅ Motivated choice (see Section 3 of report)
- ✅ Exhaustive search: 1,533 combinations (Grid + RandomizedSearch)
- ✅ Overfitting prevention: L2, Dropout, Early Stopping, etc.
