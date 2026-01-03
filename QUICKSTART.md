# Quick Start Guide - SR-FedAvg with Top-k Compression

## 🚀 Getting Started (5 minutes)

### 1. Run the Comparison Notebook Locally

```bash
# Navigate to workspace
cd c:\git\mamintoosi-papers-codes\PFLlib

# Generate MNIST dataset (if not present)
cd dataset
python generate_MNIST.py
cd ..

# Open Jupyter and run the notebook
jupyter notebook Compare_FedAvg_SR-FedAvg.ipynb
```

**Expected Time:** ~5 minutes for 50 rounds

**Output Files:**
- `comparison_results_three_methods.png` - Accuracy & loss curves
- `comparison_table_three_methods.csv` - Metrics table
- `stability_analysis_three_methods.png` - Variance comparison

---

### 2. Run on Google Colab (Easiest)

1. Open: `Compare_FedAvg_SR-FedAvg_Colab.ipynb`
2. Click "Open in Colab" button
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run cells sequentially (Shift+Enter)

**Expected Time:** ~3 minutes with GPU

**Automatic Setup:**
- ✅ Clones PFLlib repository
- ✅ Installs dependencies
- ✅ Downloads CIFAR-10
- ✅ Trains all three methods
- ✅ Generates visualizations

---

### 3. Run Individual Algorithms

**FedAvg (Baseline):**
```bash
cd system
python main.py -algo FedAvg -dataset MNIST -go baseline -gr 50 -jr 0.1
```

**SR-FedAvg (Stein-Rule):**
```bash
python main.py -algo SR-FedAvg -dataset MNIST -go sr_only -gr 50 -jr 0.1 -srbeta 0.9 -srwarmup 5
```

**SR-FedAvg + Top-k (Full Enhancement):**
```bash
python main.py -algo SR-FedAvg -dataset MNIST -go sr_topk -gr 50 -jr 0.1 -srbeta 0.9 -srwarmup 5 -topk 0.1
```

**View Results:**
```bash
# Results saved in: results/MNIST_{algorithm}_{goal}_0.h5
# Load via h5py or use comparison notebook
```

---

## 📊 Understanding the Results

### Three Methods Compared

| Method | Acronym | What It Does |
|--------|---------|------|
| **FedAvg** | Standard Federated Averaging | Baseline - weighted average of client gradients |
| **SR-FedAvg** | Stein-Rule Federated Averaging | Adds adaptive shrinkage for stability |
| **SR-FedAvg+TopK** | SR-FedAvg with compression | Adds gradient sparsification (10% kept) |

### Key Metrics

- **Final Accuracy:** Accuracy at last round
- **Best Accuracy:** Peak accuracy achieved
- **Convergence Round:** When best accuracy reached
- **Stability Improvement:** Reduced variance % (lower is better)

### Expected Improvements

```
Baseline FedAvg:           95.2% accuracy
SR-FedAvg:               +0.2-0.3% accuracy (more stable)
SR-FedAvg + Top-k:       +0.1-0.2% accuracy (less overhead)
```

---

## 🔧 Hyperparameter Tuning

### Key Parameters

```python
# Global Training
-gr 50              # Global rounds (increase for better accuracy)
-jr 0.1             # Join ratio: 0.1 = 50% clients participate
-le 5               # Local epochs per round
-bs 512             # Batch size

# Learning
-lr 0.01            # Learning rate
-dataset MNIST      # Dataset

# SR-FedAvg Specific
-srbeta 0.9         # Shrinkage momentum (0.5-1.0)
-srwarmup 5         # Rounds before applying shrinkage

# Top-k Compression
-topk 0.1           # Keep top 10% of gradients (0.01-1.0)
```

### Recommended Configurations

**Quick Test (1-2 min):**
```bash
-gr 10 -jr 0.5 -le 2 -bs 256
```

**Research Quality (5-10 min):**
```bash
-gr 50 -jr 0.1 -le 5 -bs 512
```

**Publication Quality (20-30 min):**
```bash
-gr 100 -jr 0.1 -le 10 -bs 1024
```

---

## 📈 Interpreting Plots

### Accuracy Curve
- **X-axis:** Global communication rounds
- **Y-axis:** Test accuracy (%)
- **Bands:** ±1 standard deviation (uncertainty)
- **Key:** SR-FedAvg should stabilize faster, especially at low participation rates

### Loss Curve
- **X-axis:** Global communication rounds
- **Y-axis:** Training loss
- **Pattern:** Monotonic decrease (or stable fluctuations)
- **Key:** Lower loss ≠ always better (watch for overfitting)

### Variance (Stability) Plot
- **X-axis:** Rounds
- **Y-axis:** Standard deviation of accuracy/loss
- **Key:** Lower variance = more consistent performance across client samples

---

## 🐛 Troubleshooting

### Issue: "Module not found: flcore"
**Solution:**
```python
import sys
sys.path.insert(0, '/full/path/to/PFLlib/system')
```

### Issue: "No CUDA device found"
**Solution:** Works fine on CPU (slower). For GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Results file not found"
**Solution:** Check results directory and matching goal suffix:
```bash
# Files saved as: results/DATASET_ALGORITHM_GOAL_RUN.h5
# Example: results/MNIST_SR-FedAvg_comparison_sr_0.h5
```

### Issue: "Out of memory"
**Solution:** Reduce batch size and join_ratio:
```bash
python main.py ... -bs 32 -jr 0.05 -le 2
```

---

## 📚 File Locations

```
PFLlib/
├── Compare_FedAvg_SR-FedAvg.ipynb           ← Local notebook (recommended to start)
├── Compare_FedAvg_SR-FedAvg_Colab.ipynb     ← Colab notebook (easiest)
├── IMPLEMENTATION_COMPLETE.md               ← Full documentation
├── QUICKSTART.md                            ← This file
├── system/
│   ├── main.py                              ← Entry point
│   ├── flcore/
│   │   ├── clients/
│   │   │   ├── clientavg.py                 ← Base class
│   │   │   └── clienttopk.py                ← Top-k compression (NEW)
│   │   └── servers/
│   │       ├── serveravg.py                 ← FedAvg reference
│   │       └── serversrfedavg.py            ← SR-FedAvg (NEW)
│   └── utils/
│       └── data_utils.py                    ← Data loading
├── dataset/
│   ├── generate_MNIST.py
│   ├── generate_Cifar10.py
│   └── [other datasets...]
└── results/
    └── [saved .h5 files]
```

---

## 🎯 Next Steps

1. **Try It:** Run local or Colab notebook
2. **Understand:** Read comparison table and plots
3. **Experiment:** Adjust hyperparameters
4. **Deploy:** Integrate into your federated system
5. **Extend:** Add more datasets or clients

---

## 📞 Questions?

- **Design Details:** See [SR_FedAvg_Design.md](SR_FedAvg_Design.md)
- **Full Documentation:** See [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
- **Code:** All source in [system/flcore/](system/flcore/)

---

**Ready?** Start with [Compare_FedAvg_SR-FedAvg.ipynb](Compare_FedAvg_SR-FedAvg.ipynb) 🚀
