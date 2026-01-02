import time
import torch
import copy
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread


class SR_FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # SR-FedAvg specific parameters
        self.sr_beta = args.sr_beta if hasattr(args, 'sr_beta') else 0.9
        self.epsilon = 1e-10  # small constant to avoid division by zero
        
        # Initialize server momentum to None (will be initialized after first aggregation)
        self.server_momentum = None

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"SR-FedAvg momentum coefficient (beta): {self.sr_beta}")
        print("Finished creating server and clients.")

        self.Budget = []

    def aggregate_parameters_sr(self):
        """
        Aggregate client updates using Stein-Rule shrinkage.
        
        Algorithm:
        1. Standard FedAvg aggregation to get delta_t (client update)
        2. Update server momentum: m = beta * m + (1 - beta) * delta_t
        3. For each layer, compute variance and shrinkage factor
        4. Apply SR correction: delta_sr = m + c * (delta - m)
        5. Update global model with delta_sr
        """
        assert (len(self.uploaded_models) > 0)

        # Save old global model parameters
        old_global_params = [param.data.clone() for param in self.global_model.parameters()]
        
        # Step 1: Compute standard FedAvg aggregation
        aggregated_model = copy.deepcopy(self.uploaded_models[0])
        for param in aggregated_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for agg_param, client_param in zip(aggregated_model.parameters(), client_model.parameters()):
                agg_param.data += client_param.data.clone() * w
        
        # Compute delta_t (aggregated update): delta_t = aggregated_model - old_model
        delta_t = [agg_param.data.clone() - old_param 
                   for agg_param, old_param in zip(aggregated_model.parameters(), old_global_params)]
        
        # Step 2: Initialize or update server momentum
        if self.server_momentum is None:
            # First round: initialize momentum with delta_t
            self.server_momentum = [d.clone() for d in delta_t]
        else:
            # Update momentum: m = beta * m + (1 - beta) * delta_t
            self.server_momentum = [self.sr_beta * m + (1 - self.sr_beta) * d 
                                   for m, d in zip(self.server_momentum, delta_t)]
        
        # Step 3-4: Compute SR shrinkage factor per layer and apply correction
        delta_sr = []
        for layer_idx, (delta_l, m_l) in enumerate(zip(delta_t, self.server_momentum)):
            # Compute difference from momentum
            diff = delta_l - m_l
            
            # Estimate variance (per layer)
            sigma2_l = torch.mean(diff ** 2)
            
            # Number of parameters in this layer
            p_l = float(diff.numel())
            
            # Compute squared distance
            D_l = torch.sum(diff ** 2)
            
            # Compute shrinkage factor
            # c_l = max(0, 1 - ((p_l - 2) * sigma2_l) / (D_l + eps))
            numerator = (p_l - 2) * sigma2_l
            denominator = D_l + self.epsilon
            c_l = torch.clamp(1.0 - numerator / denominator, min=0.0, max=1.0)
            
            # Apply SR correction: delta_l_sr = m_l + c_l * (delta_l - m_l)
            delta_l_sr = m_l + c_l * diff
            delta_sr.append(delta_l_sr)
        
        # Step 5: Update global model with SR-corrected delta
        # global_model = old_global_model + delta_sr
        for param, old_param, d_sr in zip(self.global_model.parameters(), old_global_params, delta_sr):
            param.data = old_param + d_sr

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            
            # Use SR-FedAvg aggregation instead of standard aggregation
            self.aggregate_parameters_sr()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
