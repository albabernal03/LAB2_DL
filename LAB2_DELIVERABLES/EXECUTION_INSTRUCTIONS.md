# LAB2_DELIVERABLES - Execution Instructions

## ⚠️ IMPORTANT: How to Execute the Scripts

### Repository Structure

```
LAB2_DL/
├── LAB2_DELIVERABLES/          ← Deliverables folder
│   ├── README.md
│   ├── INDEX.md
│   ├── QUICK_START.md
│   ├── reports/
│   │   └── COMPREHENSIVE_REPORT.md  ⭐ READ FIRST
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
├── clockwise_dataset/          ← Required data
├── horizontal_swipe_dataset/
├── forward_thrust_dataset/
├── vertical_updown_dataset/
├── wrist_twist_dataset/
└── extra/                      ← Auxiliary files (not required)
```

---

## 📖 For Evaluation (WITHOUT executing code)

**Read these documents in order:**

1. **`README.md`** (5 min) - Overview
2. **`reports/COMPREHENSIVE_REPORT.md`** (20-30 min) ⭐ **MAIN DOCUMENT**
3. **`INDEX.md`** (optional) - File index

The **COMPREHENSIVE_REPORT.md** contains:
- ✅ Explanation of the 4 architectures (FNN, LSTM, CNN, Hybrid)
- ✅ Motivation for each choice
- ✅ Hyperparameter search strategy (1,533 combinations)
- ✅ Scientific justification for RandomizedSearch (Bergstra & Bengio, 2012)
- ✅ 7 anti-overfitting techniques implemented
- ✅ Complete results analysis

---

## 🚀 To Execute the Scripts

### Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/albabernal03/LAB2_DL.git
cd LAB2_DL

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

### Execution

**IMPORTANT:** Scripts must be executed from the `optimized_models` folder:

```bash
cd LAB2_DELIVERABLES/optimized_models

# Run models (one by one)
python train_neural_network_optimized.py  # FNN: 30-60 min
python train_lstm_optimized.py            # LSTM: 20-30 min
python train_cnn_optimized.py             # CNN: 30-45 min
python train_hybrid_optimized.py          # Hybrid: 2-4 hours

# Compare results (after training all)
python compare_all_models.py
```

### Where Results Are Saved

Scripts automatically create these folders:

```
LAB2_DELIVERABLES/optimized_models/
├── models/                    ← Trained models (.keras, .pkl)
└── (scripts look for data in ../../*_dataset/)
```

And also use:
```
LAB2_DL/
├── models_dl/                 ← LSTM/CNN/Hybrid results
└── figures/                   ← Visualizations
```

---

## ✅ Quick Verification

To verify everything works without training (quick test):

```bash
cd LAB2_DELIVERABLES/optimized_models
python -c "import tensorflow as tf; import sklearn; print('✅ Dependencies OK')"
```

---

## 📊 Expected Results

After executing all scripts:

### Saved Models
- `models/neural_network_optimized.keras`
- `../../models_dl/LSTM_optimized_best.keras`
- `../../models_dl/CNN1D_optimized_best.keras`
- `../../models_dl/Hybrid_optimized_best.keras`

### Search Results
- `models/nn_hyperparameter_results.pkl` (FNN: 576 combinations)
- `../../models_dl/lstm_hyperparameter_results.pkl` (LSTM: 50 combinations)
- `../../models_dl/cnn_hyperparameter_results.pkl` (CNN: 75 combinations)
- `../../models_dl/hybrid_hyperparameter_results.pkl` (Hybrid: 512 combinations)

### Visualizations
- `../../figures/model_comparison_summary.png`
- `../../figures/l2_regularization_analysis.png`
- `../../figures/overfitting_analysis_all_models.png`

---

## ⏱️ Execution Times

| Script | Combinations | Time (CPU) | Time (GPU) |
|--------|--------------|------------|------------|
| FNN | 576 | 30-60 min | 10-20 min |
| LSTM | 50 | 20-30 min | 5-10 min |
| CNN | 75 | 30-45 min | 10-15 min |
| Hybrid | 512 | 2-4 hours | 45-90 min |
| **TOTAL** | **1,533** | **4-6 hours** | **1.5-2.5 hours** |

---

## 🆘 Troubleshooting

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: "No such file or directory: '../../clockwise_dataset'"
**Solution:** Make sure to run from `LAB2_DELIVERABLES/optimized_models/`

### Error: "No module named 'seaborn'"
```bash
pip install seaborn
```

---

## 📧 Contact

**Author:** Alba Bernal  
**Repository:** https://github.com/albabernal03/LAB2_DL  
**Date:** January 2026

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
