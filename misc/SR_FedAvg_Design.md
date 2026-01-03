
### 📄 SR-FedAvg — Design Specification for PFLlib

#### Goal

Extend the existing **FedAvg server update** in PFLlib by adding a **Stein-Rule (SR) shrinkage step** on the aggregated client updates.

No new optimizer should be reimplemented from scratch.

---

### Where to Modify

Server logic is located in:

```
system/flcore/servers/
```

Identify the server class that:

* aggregates client updates
* updates global model using FedAvg

Create a new server:

```
system/flcore/servers/sr_fedavg.py
```

This server must be identical to FedAvg **except for the SR step**.

---

### Algorithm (Server Side)

Let:

* `delta_t`: aggregated client update (FedAvg result)
* `m_prev`: exponential moving average of past global updates
* `beta`: momentum coefficient (e.g. 0.9)

---

#### Step 1: Maintain Server Momentum

Initialize once:

```python
m = zeros_like(model_parameters)
```

Update each round:

```python
m = beta * m + (1 - beta) * delta_t
```

---

#### Step 2: Estimate Variance (Per Layer)

For each parameter tensor (l):

```python
sigma2_l = mean((delta_l - m_l) ** 2)
```

---

#### Step 3: Compute Stein Shrinkage Factor

For each layer:

```python
p_l = number_of_elements(delta_l)
D_l = sum((delta_l - m_l) ** 2)

c_l = max(0.0, 1.0 - ((p_l - 2) * sigma2_l) / (D_l + eps))
```

---

#### Step 4: Apply SR Correction

```python
delta_l_sr = m_l + c_l * (delta_l - m_l)
```

---

#### Step 5: Global Model Update

Replace FedAvg update:

```python
theta = theta - lr * delta_sr
```

---

### Important Constraints

* SR must be applied **on the server**
* SR must be **per-layer**
* Shrinkage factor must be non-negative
* When `c_l = 1`, behavior must reduce to FedAvg

---

### Naming

Register optimizer/server name as:

```yaml
server: sr_fedavg
```

---

### Expected Outcome

SR-FedAvg should:

* behave exactly like FedAvg on IID data
* be more stable under partial participation
* reduce variance on CIFAR-10 with CNNs