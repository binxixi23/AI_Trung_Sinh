# utils/checkpoint.py
import os
import torch

class ManifoldCheckpointManager:
    def __init__(self, storage_dir="checkpoints"):
        """
        Initializes the Checkpoint Manager and ensures a dedicated physical 
        directory exists on disk to protect generational state assets.
        """
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            print(f"[CHECKPOINT INIT] Created storage directory: {os.path.abspath(self.storage_dir)}")

    def save_generational_state(self, generation: int, step: int, weights: torch.Tensor, core_seed: torch.Tensor, metric_g: torch.Tensor):
        """
        Serializes and dumps the active geometric states of the model into a unified PyTorch binary block.
        """
        state_dict = {
            "generation": generation,
            "step": step,
            "manifold_weights": weights.detach().cpu(),
            "core_knowledge_seed": core_seed.detach().cpu(),
            "metric_tensor_g": metric_g.detach().cpu()
        }
        
        filename = f"ai_trung_sinh_gen_{generation}_step_{step}.pt"
        full_path = os.path.join(self.storage_dir, filename)
        
        # Binary serialization block
        torch.save(state_dict, full_path)
        print(f" -> [CHECKPOINT SAVE] Generational assets safely persisted to disk: {filename}")

    def load_generational_state(self, file_name: str):
        """
        Loads a legacy state file from disk and parses the underlying tensors back into runtime memory arrays.
        """
        full_path = os.path.join(self.storage_dir, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"[CHECKPOINT ERROR] Targeted snapshot file not found: {full_path}")
            
        print(f"[CHECKPOINT LOAD] Hydrating system tensors from historical file: {file_name}")
        state_dict = torch.load(full_path, map_location=torch.device('cpu'))
        return state_dict
