

I am working with **PFLlib** (TsingZ0) and want to implement a **minimal, stable SR-FedAvg variant with Top-k compression**, fully consistent with the existing codebase and dependencies.

### Context

* Client implementation file:
  `system/flcore/clients/clientavg.py` (class `clientAVG`)
* Server-side FedAvg implementation is already present (e.g., `serveravg.py`)
* The current `clientAVG.train()` method **does not return model updates (delta)** explicitly.
* I want to **minimally extend** the existing logic without breaking any caller or inheritance structure.

---

### Task 1 — Client-side (Top-k compression)

Please modify **`clientavg.py`** as follows:

1. **Before local training**, create a deep copy of the initial model parameters.
2. **After local training**, compute the model update
   [
   \Delta = w_{\text{local}} - w_{\text{initial}}
   ]
   as a list of tensors, consistent with how FedAvg aggregation expects updates.
3. Implement a **simple Top-k compressor**:

   * Operates tensor-wise
   * Keeps the top `k_ratio` fraction of entries by absolute value
   * Zeroes out the rest
   * No momentum, no error feedback, no randomness
4. Apply Top-k compression to the computed delta.
5. Return the compressed delta from `train()`.

Constraints:

* Do **not** change the training loop logic itself.
* Use `self.args.topk_ratio` as the compression hyperparameter.
* Follow the existing coding style and imports in PFLlib.
* Do not introduce new dependencies.

---

### Task 2 — Server-side (SR-FedAvg aggregation)

Please modify the **server-side FedAvg aggregation** (e.g., in `serveravg.py`) to add a **new class named exactly `SR_FedAvg`** with the following properties:

1. The class name must remain **`SR_FedAvg`** (I will replace an existing one).
2. It aggregates **client deltas** (already Top-k compressed).
3. Compute the standard FedAvg mean update.
4. After a warmup period (`self.args.sr_warmup` rounds), apply **Stein-type shrinkage**:

   * Estimate variance across client updates
   * Compute shrinkage factor
     [
     c = 1 - \frac{(n-2),\mathrm{Var}}{|\bar{\Delta}|^2 + \varepsilon}
     ]
   * Clamp (c) to ([0, 1])
   * Multiply the mean update by (c)
5. **No momentum, no EMA, no history buffers**.

Constraints:

* Warmup rounds use **plain FedAvg**
* Code must match PFLlib aggregation style
* Use PyTorch only

---

### Configuration assumptions

* `topk_ratio` is defined in args / config
* `sr_warmup` is defined in args / config
* Partial participation may or may not be enabled, but SR must **not depend on it**

---

### Goal

The final result should be:

* A clean **SR + compression** baseline
* Stable after warmup (no freezing)
* Easy to justify in an academic paper (JMLR-style)

Please implement the required changes carefully, respecting all existing method calls and dependencies.

