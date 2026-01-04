import time
from flcore.clients.clienttopk import clientTopK
from flcore.servers.serverbase import Server


class SR_FedAvg(Server):
    """
    Server-side implementation for SR-FedAvg (CORRECT version):

    - No Stein Rule on the server
    - No shrinkage after aggregation
    - Standard FedAvg aggregation
    - SR + Top-k are applied ONLY on client-side updates
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # ---- bookkeeping / logging only ----
        self.topk_ratio = getattr(args, "topk_ratio", 1.0)
        self.use_sr = getattr(args, "use_sr", True)

        # select slow clients
        self.set_slow_clients()

        # IMPORTANT: use clientTopK (which includes SR + Top-k)
        self.set_clients(clientTopK)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"Client-side SR enabled: {self.use_sr}")
        print(f"Client-side Top-k ratio: {self.topk_ratio:.2%}")
        print("Server aggregation: Standard FedAvg")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            start_time = time.time()

            # ---- client selection & broadcast ----
            self.selected_clients = self.select_clients()
            self.send_models()

            # ---- evaluation ----
            if i % self.eval_gap == 0:
                print(f"\n------------- Round {i} -------------")
                print("Evaluate global model")
                self.evaluate()

            # ---- local training ----
            for client in self.selected_clients:
                client.train()

            # ---- receive client updates ----
            self.receive_models()

            # ---- optional DLG ----
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # ---- STANDARD FedAvg aggregation ----
            self.aggregate_parameters()

            # ---- bookkeeping ----
            self.Budget.append(time.time() - start_time)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # ---- early stopping ----
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

        # ---- fine-tuning on new clients (if any) ----
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientTopK)
            print("\n------------- Fine tuning round -------------")
            print("Evaluate new clients")
            self.evaluate()
