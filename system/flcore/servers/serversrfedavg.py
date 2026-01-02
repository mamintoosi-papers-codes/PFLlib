import time
import torch
import copy
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server


class SR_FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # ---- SR-FedAvg parameters ----
        self.warmup_rounds = getattr(args, "sr_warmup_rounds", 10)
        self.epsilon = 1e-10

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"SR-FedAvg warmup rounds: {self.warmup_rounds}")
        print("Finished creating server and clients.")

        self.Budget = []

    def aggregate_parameters_sr(self, round_idx):
        """
        Minimal SR-FedAvg aggregation (no momentum, with warmup).

        After warmup:
            delta_l_sr = c_l * delta_l
        """
        assert len(self.uploaded_models) > 0

        # ---- Save old global parameters ----
        old_global_params = [p.data.clone() for p in self.global_model.parameters()]

        # ---- Standard FedAvg aggregation ----
        aggregated_model = copy.deepcopy(self.uploaded_models[0])
        for p in aggregated_model.parameters():
            p.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for agg_p, client_p in zip(aggregated_model.parameters(),
                                       client_model.parameters()):
                agg_p.data += w * client_p.data

        # ---- Compute delta_t ----
        delta_t = [
            agg_p.data.clone() - old_p
            for agg_p, old_p in zip(aggregated_model.parameters(),
                                    old_global_params)
        ]

        # ---- Warmup: behave exactly like FedAvg ----
        if round_idx < self.warmup_rounds:
            for p, old_p, d in zip(self.global_model.parameters(),
                                   old_global_params,
                                   delta_t):
                p.data = old_p + d
            return

        # ---- Apply Stein Rule shrinkage (layer-wise) ----
        delta_sr = []
        for d_l in delta_t:
            p_l = float(d_l.numel())

            # squared norm
            D_l = torch.sum(d_l ** 2)

            # variance estimate (scalar)
            sigma2_l = torch.mean(d_l ** 2)

            # Stein shrinkage coefficient
            c_l = 1.0 - ((p_l - 2.0) * sigma2_l) / (D_l + self.epsilon)
            c_l = torch.clamp(c_l, min=0.0, max=1.0)

            delta_sr.append(c_l * d_l)

        # ---- Update global model ----
        for p, old_p, d_sr in zip(self.global_model.parameters(),
                                  old_global_params,
                                  delta_sr):
            p.data = old_p + d_sr

    def train(self):
        for i in range(self.global_rounds + 1):
            start_time = time.time()

            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n------------- Round {i} -------------")
                print("Evaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # ---- SR-FedAvg aggregation ----
            self.aggregate_parameters_sr(i)

            self.Budget.append(time.time() - start_time)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc],
                top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy:")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round:")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print("\n------------- Fine tuning round -------------")
            print("Evaluate new clients")
            self.evaluate()
