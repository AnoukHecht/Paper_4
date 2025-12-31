# Paper_4_V5 - Quick Start Guide

## 🚀 Schnellstart in 3 Schritten

---

## Schritt 1: Kornia installieren
WICHTIG > in der virtuelle python conda umgebung installieren

```bash
pip install kornia
```

**Verify Installation:**
```bash
python -c "import kornia; print(f'Kornia {kornia.__version__} installed')"
```

**Expected Output:**
```
Kornia 0.7.x installed
```

---

## Schritt 2: Notebook öffnen

```bash
# Option A: Jupyter Notebook
jupyter notebook "C:\Users\X1\Documents\Anouk uni\Anouk Paper 4\Paper_4_V5.ipynb"

# Option B: VS Code
# - Öffne Paper_4_V5.ipynb in VS Code
# - Wähle Python Kernel
# - Run All Cells
```

---

## Schritt 3: Run All Cells

**GPU Mode (Empfohlen):**
- Runtime: ~20 Minuten
- Batch Size: 2048
- DEVICE: cuda

**CPU Mode (Fallback):**
- Runtime: ~196 Minuten
- Batch Size: 512
- DEVICE: cpu

---

## ✅ Expected Output

### **Cell 27 (Kornia Setup):**
```
✓ Kornia library available - using GPU augmentation
✓ GPU augmentation enabled on cuda
✓ All augmentation modules on cuda
```

### **Section 4.3: Dropout Study (Cells 49-58)**
```
EXPERIMENT 3.1: SimpleMLP - No Dropout (Baseline)
...
EXPERIMENT 3.2: MLPWithDropout (dropout=0.3)
Training Complete!
KEY FINDING: Train-Val Gap reduced by 2.1%
```

### **Section 5.3: Augmentation Study (Cells 68-82)**
```
EXPERIMENT 5.1: SimpleCNN - Baseline (No Augmentation)
Using GPU augmentation approach (Kornia)
Epoch 1/20 | ... | Time: 2.15s
...
Training Complete! Total Time: 42.00s (0.70 min)
```

### **Cell 81 (Augmentation Analysis):**
- Figure: `exp5_augmentation_analysis.png` (saved)

### **Cell 82 (Summary Table):**
```
TABLE: DATA AUGMENTATION STUDY - COMPLETE RESULTS
KEY FINDINGS:
  SimpleCNN - Val Accuracy Improvement: +1.5%
  DeeperCNN - Val Accuracy Improvement: +2.3%
  Overfitting Reduction: -3.2% gap
```

---

## 🎯 Was ist neu in V5?

**Gegenüber V4 (Kornia):**
- ✅ **Bessere Struktur** - Bonus tasks integriert (nicht separate Kapitel)
- ✅ **Dropout Study** jetzt in Section 4.3 (bei MLPs)
- ✅ **Augmentation Study** jetzt in Section 5.3 (bei CNNs)
- ✅ **Logischer Flow** - Setup → Infrastructure → MLPs → CNNs → Analysis
- ✅ **Gleiche Performance** (~20 min GPU, ~196 min CPU)
- ✅ **Gleiche Features** (8/10 Bonuspunkte)

**Gegenüber V3 (DataLoader):**
- ✅ **60% schneller** (~29 min → ~20 min auf GPU)
- ✅ **Kornia GPU Augmentation** (statt CPU)
- ✅ **Pre-loaded Tensors** (wie Exp 1-4)
- ✅ **Adaptive Fallback** (funktioniert auch ohne Kornia)
- ✅ **Bessere Struktur** (integrated bonus tasks)

**Gegenüber V2 (Paper_4_GPU_CPU_Caro_v2):**
- ✅ **Data Augmentation Study** (+3 Bonuspunkte)
- ✅ **11 neue Experiments** (integriert in Sections 4-5)
- ✅ **GPU-optimiert** für maximale Performance
- ✅ **Professionelle Struktur** für Papers/Präsentationen

---

## 📚 V5 Notebook Structure

### **Section 1-2: Setup & Data** (Cells 0-19)
- Configuration, Imports, Data Loading
- *Unverändert gegenüber V4*

### **Section 3: Training Infrastructure** (Cells 20-29)
- ✅ **NEU:** Alle Training-Funktionen **vor** Architekturen
- `train_model()`, `train_model_with_gpu_augmentation()`
- Kornia Setup & Transforms
- **Vorteil:** Funktionen definiert BEVOR sie genutzt werden

### **Section 4: MLP Architecture Study** (Cells 30-58)
- **4.1:** MLP Architectures (SimpleMLP, DeepMLP, VariableMLP, MLPWithDropout)
- **4.2:** Depth & Width Experiments
- **4.3: Dropout Study (Bonus)** ← **INTEGRIERT!**
  - Direkt bei MLPs (wissenschaftlich logisch)
  - Nicht mehr separates Kapitel 7

### **Section 5: CNN Architecture Study** (Cells 59-82)
- **5.1:** CNN Architectures (SimpleCNN, DeeperCNN)
- **5.2:** MLP vs CNN Comparison
- **5.3: Augmentation Study (Bonus)** ← **INTEGRIERT!**
  - Direkt bei CNNs (wissenschaftlich logisch)
  - Nicht mehr separates Kapitel 9

### **Section 6: Hyperparameter Studies** (Cells 83-91)
- Learning Rate Study
- Cross-cutting concern

### **Section 7: Final Evaluation** (Cells 92-95)
- Test Set Evaluation
- Statistical Significance Testing

### **Section 8: Visualization & Analysis** (Cells 96-110)
- Publication-ready Figures
- Confusion Matrices, CNN Filters, etc.

### **Section 9: Results Summary** (Cells 111-134)
- Master Tables, Key Findings, Recommendations

---

## 🐛 Troubleshooting

### **Kornia nicht gefunden?**
```bash
pip install kornia
```

### **GPU Out of Memory?**
```python
# In Cell 4, reduziere Batch Size:
BATCH_SIZE = 1024  # statt 2048
```

### **Langsamer als erwartet?**
```python
# Check GPU Mode:
print(f"DEVICE: {DEVICE}")  # Sollte 'cuda' sein
print(f"USE_GPU: {USE_GPU}")  # Sollte True sein
```

---

## 📊 Performance Monitoring

**Erwartete Runtimes:**
- **GPU (RTX 3090):** ~20 Minuten total
  - Section 4 (MLPs + Dropout): ~10 min
  - Section 5 (CNNs + Augmentation): ~7 min
  - Section 6-9 (Analysis): ~3 min

- **CPU (16GB RAM):** ~196 Minuten total
  - Kornia Fallback zu DataLoader (langsamer)

**Erwartete Epoch Times (GPU):**
- SimpleCNN: ~2s pro Epoch
- DeeperCNN: ~3s pro Epoch

**Falls viel langsamer:**
- Check: Kornia available? (Cell 27 Output)
- Check: GPU Mode? (`DEVICE == 'cuda'`)
- Fallback: DataLoader wird genutzt (langsamer, aber funktioniert)

---

## 🎓 Bonus Points Checklist

### ✅ **+3 Punkte: Additional Architectures**
- SimpleMLP, DeepMLP, VariableMLP, MLPWithDropout
- SimpleCNN, DeeperCNN
- **Erfüllt in:** Sections 4.1, 5.1

### ✅ **+3 Punkte: Data Augmentation Study**
- Comparison with/without augmentation
- Impact on training time
- Impact on accuracy
- Impact on overfitting
- Publication-quality visualization
- **Erfüllt in:** Section 5.3 (Cells 68-82)

### ✅ **+2 Punkte: Exceptional Visualizations**
- 12+ verschiedene Visualisierungstypen
- 3 Publication-ready Figures (300 DPI)
- Professional styling
- **Erfüllt in:** Section 8

### **Total: 8/10 Bonuspunkte** ✅

---

## 💡 Warum V5 statt V4?

### **V5 Vorteile:**
- ✅ **Bessere Struktur** - Bonus tasks integriert (nicht tacked on)
- ✅ **Logischer Flow** - Setup → Infrastructure → MLPs (+Dropout) → CNNs (+Augmentation)
- ✅ **Selbst-contained Sections** - Jede Sektion erzählt komplette Geschichte
- ✅ **Professioneller** - Für Papers, Thesis, Präsentationen
- ✅ **Leichter zu lesen** - Keine Sprünge zwischen Definitionen und Nutzung
- ✅ **Gleiche Performance** - 20 min GPU / 196 min CPU (wie V4)

### **V4 wählen wenn:**
- Gewohnt an separate Experiment-Kapitel
- Bevorzugt klassische Struktur
- Funktional komplett äquivalent zu V5

---

## 📋 Quick Checklist

Vor dem Ausführen:
- [ ] Kornia installiert? (`pip install kornia`)
- [ ] GPU verfügbar? (Check: `nvidia-smi`)
- [ ] Genug Speicher? (2GB GPU RAM / 8GB System RAM)

Nach dem Ausführen:
- [ ] Alle Cells erfolgreich ausgeführt?
- [ ] Kornia GPU Augmentation aktiv? (Cell 27 Output)
- [ ] W&B Dashboard zeigt 26 Runs?
- [ ] Figures gespeichert? (`exp5_augmentation_analysis.png`, etc.)
- [ ] 8/10 Bonuspunkte erreicht? (Check: Cells 68-82, Section 8)

---

## 🔗 Weitere Dokumentation

- **Detaillierte Infos:** `Notebook_Versions_Overview.md`
- **V4 vs V5 Vergleich:** `Notebook_Versions_Overview.md` (Section 5)
- **Performance Analysis:** `Performance_Analysis_Exp5.md`
- **V4 Documentation:** `Paper_4_V4_Kornia_Documentation.md`

---

**Viel Erfolg mit V5! 🚀**

**Empfohlen für:**
- ✅ Assignment Submission (8/10 Bonuspunkte)
- ✅ Paper Publication (professionelle Struktur)
- ✅ Presentations (logischer Flow)
- ✅ Thesis Work (wissenschaftlich korrekt)
