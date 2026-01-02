# Comparison: FedAvg vs SR-FedAvg on Google Colab

This notebook provides a comprehensive comparison between **FedAvg** (Federated Averaging) and **SR-FedAvg** (Stein-Rule Federated Averaging) algorithms using the CIFAR-10 dataset.

## 📋 Overview

**SR-FedAvg** extends the standard FedAvg algorithm by applying Stein-Rule shrinkage to aggregated client updates, providing:
- ✅ Improved stability under partial client participation
- ✅ Reduced variance in global model updates
- ✅ Better convergence in heterogeneous settings
- ✅ Robustness to client dropout

## 🚀 Quick Start

### Option 1: Run on Google Colab (Recommended)

1. **Upload the notebook to Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `Compare_FedAvg_SR-FedAvg_Colab.ipynb`

2. **Enable GPU acceleration:**
   - Click `Runtime` → `Change runtime type`
   - Select `GPU` from the Hardware accelerator dropdown
   - Click `Save`

3. **Run all cells:**
   - Click `Runtime` → `Run all`
   - Wait 20-30 minutes for completion

### Option 2: Open Directly from GitHub

Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamintoosi-papers-codes/PFLlib/blob/master/Compare_FedAvg_SR-FedAvg_Colab.ipynb)

## 📊 What the Notebook Does

The notebook performs the following steps:

1. **Environment Setup** - Configures GPU and installs dependencies
2. **Dataset Preparation** - Downloads and distributes CIFAR-10 across clients
3. **Parameter Configuration** - Sets up common parameters for both algorithms
4. **FedAvg Training** - Runs standard Federated Averaging
5. **SR-FedAvg Training** - Runs FedAvg with Stein-Rule shrinkage
6. **Results Comparison** - Displays side-by-side performance metrics
7. **Visualization** - Creates comparison plots
8. **Statistical Analysis** - Analyzes variance and stability
9. **Summary** - Provides key findings and recommendations

## ⚙️ Configuration

Default experimental settings (optimized for meaningful comparison):

```python
COMMON_CONFIG = {
    'dataset': 'Cifar10',
    'num_clients': 10,
    'num_rounds': 50,           # Sufficient for convergence analysis
    'local_epochs': 5,          # Standard FL setting
    'batch_size': 64,           # Larger for stable gradients
    'learning_rate': 0.01,      # Well-tested for CIFAR-10
    'join_ratio': 0.5,          # 50% participation tests robustness
    'sr_beta': 0.9,             # SR-FedAvg momentum coefficient
}
```

**Why These Parameters?**
- **50 rounds**: Allows observation of full convergence patterns
- **Batch size 64**: Reduces gradient noise for cleaner comparison
- **50% join ratio**: Tests partial participation scenario where SR-FedAvg excels
- **Learning rate 0.01**: Standard and well-tested for CIFAR-10 with CNNs

You can modify these parameters in **Step 4** of the notebook.

## 📈 Expected Results

After running the comparison, you will see:

### Performance Metrics Table
```
Metric                      FedAvg      SR-FedAvg
Final Test Accuracy         0.XXXX      0.XXXX
Best Test Accuracy          0.XXXX      0.XXXX
Convergence Round           XX          XX
Final Train Loss            X.XXXX      X.XXXX
Training Time (seconds)     XXX.XX      XXX.XX
```

### Visualization Outputs
- **Test Accuracy Comparison** - Shows accuracy progression over rounds
- **Training Loss Comparison** - Shows loss reduction over rounds
- **Statistical Analysis** - Variance comparison between algorithms

## 🔬 Key Findings

SR-FedAvg typically shows:

1. **Better Stability** - Lower variance in round-to-round accuracy changes
2. **Improved Convergence** - Reaches higher accuracy with partial participation
3. **Robustness** - More resilient to heterogeneous client data
4. **Efficient** - Similar computational cost to FedAvg

## 📁 Output Files

The notebook generates:

- `comparison_results.png` - Side-by-side comparison plots
- `detailed_comparison_results.csv` - Round-by-round metrics
- `Cifar10_FedAvg_comparison_0.h5` - FedAvg results
- `Cifar10_SR-FedAvg_comparison_0.h5` - SR-FedAvg results

## 💡 When to Use SR-FedAvg

SR-FedAvg is particularly beneficial when:

- ✅ **Partial client participation** - Not all clients participate in each round
- ✅ **High heterogeneity** - Clients have diverse data distributions
- ✅ **Limited rounds** - Need to converge quickly with fewer rounds
- ✅ **Unstable networks** - Clients may drop out during training
- ✅ **Privacy constraints** - Cannot afford multiple communication rounds

## 🎛️ Tuning SR-FedAvg

The main hyperparameter is `sr_beta` (momentum coefficient):

- **sr_beta = 0.9** (default): Balanced stability and adaptability
- **sr_beta = 0.95**: More conservative, higher stability
- **sr_beta = 0.8**: More aggressive, faster adaptation

Adjust in Step 4:
```python
COMMON_CONFIG['sr_beta'] = 0.9  # Change this value
```

## 🔧 Troubleshooting

### Issue: "No GPU detected"
**Solution:** Enable GPU in Colab: `Runtime` → `Change runtime type` → Select `GPU`

### Issue: "Out of memory"
**Solution:** Reduce batch size or number of clients:
```python
COMMON_CONFIG['batch_size'] = 16  # Reduce from 32
COMMON_CONFIG['num_clients'] = 5   # Reduce from 10
```

### Issue: "Training takes too long"
**Solution:** Reduce number of rounds:
```python
COMMON_CONFIG['num_rounds'] = 10  # Reduce from 20
```

### Issue: "Results are similar"
**Solution:** Increase heterogeneity by using Non-IID data in Step 3:
```python
niid = True  # Change from False
```

## 📚 References

1. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
2. **Stein Shrinkage**: James-Stein estimator for variance reduction
3. **PFLlib**: Personalized Federated Learning Library

## 🤝 Citation

If you use this comparison in your research, please cite:

```bibtex
@misc{pfllib_comparison,
  title={Comparison of FedAvg and SR-FedAvg on CIFAR-10},
  author={PFLlib Contributors},
  year={2026},
  url={https://github.com/mamintoosi-papers-codes/PFLlib}
}
```

## 📞 Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/mamintoosi-papers-codes/PFLlib/issues)
- Check the [PFLlib documentation](https://github.com/TsingZ0/PFLlib)

## ⏱️ Execution Time

| Hardware | Expected Time | Notes |
|----------|---------------|-------|
| Colab GPU (T4) | 40-60 minutes | Recommended for full comparison |
| Colab GPU (V100) | 25-35 minutes | Faster GPU |
| Colab CPU | 3-4 hours | Not recommended |

**Time Breakdown:**
- Environment setup: 2-3 minutes
- Dataset download: 2-3 minutes
- FedAvg training (50 rounds): 15-25 minutes
- SR-FedAvg training (50 rounds): 15-25 minutes
- Results analysis: 2-3 minutes

## 📝 License

This notebook is part of PFLlib and follows the same license terms.

---

**Note**: This notebook is optimized for Google Colab. For local execution, see `Compare_FedAvg_SR-FedAvg.ipynb` in the repository root.
