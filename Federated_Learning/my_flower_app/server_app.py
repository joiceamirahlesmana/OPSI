from typing import List, Tuple, Optional
import os
import torch
from flwr.common import Context, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedProx
from my_flower_app.task import load_model

# Custom Strategy to log extra metrics and save model (PyTorch)
class CustomFedProx(FedProx):
    def __init__(self, total_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self.latest_parameters = None
        base_dir = os.path.dirname(__file__)
        self.model_save_path = os.path.join(base_dir, "model", "mosquito_classification_model.pt")

    def aggregate_fit(self, rnd: int, results: List[Tuple], failures: List):
        """Aggregate client model updates."""
        agg = super().aggregate_fit(rnd, results, failures)
        if agg is None:
            return agg
        if isinstance(agg, tuple):
            aggregated_parameters = agg[0]
        else:
            aggregated_parameters = agg
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

        # Save model after the final round
        if rnd >= self.total_rounds and self.latest_parameters is not None:
            try:
                final_weights = parameters_to_ndarrays(self.latest_parameters)
                model = load_model()  # Assuming `load_model` gives a PyTorch model
                
                # Convert ndarrays to torch tensors and load into the model
                state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in final_weights]))
                model.load_state_dict(state_dict, strict=True)
                
                # Save the global model to the specified path
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                torch.save(model.state_dict(), self.model_save_path)
                print(f"[✔] Saved global model to '{self.model_save_path}'")
            except Exception as e:
                print(f"[WARN] Failed to save global model: {e}")
        return agg

    def aggregate_evaluate(self, rnd: int, results: List[Tuple], failures: List) -> Optional[float]:
        """Aggregate evaluation results from clients."""
        aggregated_loss = super().aggregate_evaluate(rnd, results, failures)
        
        metrics = [res.metrics for _, res in results if res.metrics is not None]
        if metrics:
            accs = [m.get("accuracy", 0.0) for m in metrics]
            precs = [m.get("precision", 0.0) for m in metrics]
            f1s = [m.get("f1_score", 0.0) for m in metrics]
            
            avg_acc = sum(accs) / len(accs)
            avg_prec = sum(precs) / len(precs)
            avg_f1 = sum(f1s) / len(f1s)
            
            print(f"[ROUND {rnd}] Accuracy avg: {avg_acc:.4f}, "
                  f"Precision avg: {avg_prec:.4f}, F1 avg: {avg_f1:.4f}")
        return aggregated_loss

def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    
    # Load initial parameters from PyTorch model
    model = load_model()  # Assuming `load_model` loads a PyTorch model
    parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
    
    strategy = CustomFedProx(
        total_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        prox_mu=0.1,
        initial_parameters=parameters,
    )
    
    # Server configuration
    config = ServerConfig(num_rounds=num_rounds)
    
    return ServerAppComponents(strategy=strategy, config=config)

# Flower ServerApp
app = ServerApp(server_fn=server_fn)
