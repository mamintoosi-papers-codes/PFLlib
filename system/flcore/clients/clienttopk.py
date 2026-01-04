import torch
import numpy as np
from flcore.clients.clientavg import clientAVG


class clientTopK(clientAVG):
    """
    FedAvg client with:
    - Stein-rule shrinkage (ONLY on convolutional layers)
    - Optional Top-k compression
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.topk_ratio = getattr(args, 'topk_ratio', 0.1)
        self.use_sr = getattr(args, 'use_sr', True)
        self.sr_epsilon = 1e-12

        self.initial_params = None
        self.compressed_delta = None

    def train(self):
        # ---- Step 1: save initial params ----
        self.initial_params = [p.data.clone() for p in self.model.parameters()]

        # ---- Step 2: standard local training ----
        super().train()

        # ---- Step 3: compute local delta ----
        delta = [
            p.data.clone() - p0
            for p, p0 in zip(self.model.parameters(), self.initial_params)
        ]

        # ---- Step 4: Stein shrinkage ONLY on Conv layers ----
        if self.use_sr:
            with torch.no_grad():
                for i, (name, p) in enumerate(self.model.named_parameters()):
                    if ("conv" in name) and (delta[i].numel() > 1):
                        d = delta[i]
                        p_l = d.numel()
                        D_l = torch.sum(d ** 2)

                        sigma2_l = torch.var(d, unbiased=False)
                        c_l = 1.0 - ((p_l - 2.0) * sigma2_l) / (D_l + self.sr_epsilon)
                        c_l = torch.clamp(c_l, min=0.5, max=1.0)

                        delta[i] = c_l * d
                    # FC & bias untouched

        # ---- Step 5: Top-k compression (layer-wise) ----
        self.compressed_delta = self._apply_topk(delta)

    def _apply_topk(self, delta):
        if self.topk_ratio >= 1.0:
            return delta

        compressed = []
        for d in delta:
            if d.numel() == 0:
                compressed.append(d)
                continue

            flat = d.view(-1)
            k = max(1, int(flat.numel() * self.topk_ratio))

            _, idx = torch.topk(torch.abs(flat), k)
            mask = torch.zeros_like(flat)
            mask[idx] = 1.0

            compressed.append((flat * mask).view(d.shape))

        return compressed

    def get_compressed_delta(self):
        return self.compressed_delta
