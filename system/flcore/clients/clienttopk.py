import copy
import torch
import numpy as np
import time
from flcore.clients.clientavg import clientAVG


class clientTopK(clientAVG):
    """
    FedAvg client with Top-k gradient compression.
    
    Implements:
    1. Track initial model parameters before training
    2. Compute model update (delta) after training
    3. Apply Top-k compression to the update
    4. Store compressed updates for server aggregation
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Top-k compression ratio (e.g., 0.1 = keep top 10%)
        self.topk_ratio = getattr(args, 'topk_ratio', 0.1)
        
        # Store for tracking updates
        self.initial_params = None
        self.compressed_delta = None

    def train(self):
        """
        Enhanced training with Top-k compression tracking.
        """
        # ---- Step 1: Save initial parameters ----
        self.initial_params = [p.data.clone() for p in self.model.parameters()]
        
        # ---- Step 2: Standard training (inherited from clientAVG) ----
        super().train()
        
        # ---- Step 3: Compute model update (delta) ----
        delta = []
        for current_param, initial_param in zip(self.model.parameters(), 
                                               self.initial_params):
            delta.append(current_param.data.clone() - initial_param.clone())
        
        # ---- Step 4: Apply Top-k compression ----
        self.compressed_delta = self._apply_topk_compression(delta)

    def _apply_topk_compression(self, delta):
        """
        Apply Top-k sparsification to model updates.
        
        For each layer:
        - Keep top-k% of gradient values by absolute magnitude
        - Zero out the rest
        
        Args:
            delta: List of parameter update tensors
            
        Returns:
            delta_compressed: List of sparsified tensors
        """
        delta_compressed = []
        
        for d in delta:
            # Flatten the tensor for easier manipulation
            d_flat = d.view(-1)
            k = max(1, int(d_flat.numel() * self.topk_ratio))
            
            # Find top-k indices by absolute value
            _, top_indices = torch.topk(torch.abs(d_flat), k, largest=True)
            
            # Create mask: 1 for top-k, 0 for others
            mask = torch.zeros_like(d_flat)
            mask[top_indices] = 1.0
            
            # Apply mask to compress
            d_compressed_flat = d_flat * mask
            
            # Reshape back to original shape
            d_compressed = d_compressed_flat.view(d.shape)
            delta_compressed.append(d_compressed)
        
        return delta_compressed

    def get_compressed_delta(self):
        """
        Retrieve the most recent compressed model update.
        
        Returns:
            List of compressed parameter updates, or None if not yet computed
        """
        return self.compressed_delta
