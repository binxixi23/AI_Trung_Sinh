# core/governor.py
import torch
import zlib

class ShannonEntropyGovernor:
    def __init__(self, alpha=1.0, beta=0.5, h_max=5.95, complexity_threshold=0.85):
        """
        Initializes the Entropy Governor with a Dual-Verification Security Firewall.
        
        Args:
            alpha (float): Scaling anchor parameter for the core layer.
            beta (float): Sensitivity multiplier for the vaccine layer.
            h_max (float): Terminal entropy limit before metric collapse.
            complexity_threshold (float): Maximum normalized Kolmogorov complexity permitted
                                           before flagging semantic mimicry attacks.
        """
        self.alpha = alpha
        self.beta = beta
        self.h_max = h_max
        self.comp_threshold = complexity_threshold # Protection boundary for Question 25

    def compute_entropy(self, batch_tensor: torch.Tensor) -> float:
        """Computes real-time Shannon Entropy H(X) over token distributions"""
        probabilities = torch.softmax(batch_tensor, dim=-1)
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-9))
        return torch.clamp(entropy, min=0.0, max=10.0).item()

    def estimate_kolmogorov_complexity(self, batch_tensor: torch.Tensor) -> float:
        """Estimates algorithmic information density using zlib compression as a proxy"""
        # Convert tensor data to byte strings to parse compression metrics
        raw_bytes = batch_tensor.detach().cpu().numpy().tobytes()
        compressed_bytes = zlib.compress(raw_bytes)
        
        # Calculate the Normalized Compression Distance (NCD Ratio)
        # Closer to 1.0 means highly complex/uncompressible (Adversarial Mimicry)
        # Closer to 0.0 means highly ordered/structured (Pure Knowledge)
        ncd_ratio = len(compressed_bytes) / (len(raw_bytes) + 1e-9)
        return ncd_ratio

    def get_triadic_scale(self, current_entropy: float, batch_tensor: torch.Tensor):
        """Dynamic Governor: Uses Softmax and Kolmogorov validation to defeat mimicry exploits"""
        base_logits = torch.tensor([
            torch.log(torch.tensor(0.01)), 
            torch.log(torch.tensor(0.03)), 
            torch.log(torch.tensor(0.01))
        ])
        
        gated_logits = base_logits.clone()
        
        # 1. Evaluate Kolmogorov Complexity to audit the Shannon Governor (Resolving Question 25)
        k_complexity = self.estimate_kolmogorov_complexity(batch_tensor)
        
        if k_complexity > self.comp_threshold and current_entropy < 3.0:
            print(f" -> [GOVERNOR WALL] Semantic Mimicry Detected! Low Entropy ({current_entropy:.2f}) paired with High Complexity ({k_complexity:.2f}).")
            # Override logits: Force maximum vaccine deployment to quarantine the masquerading payload
            gated_logits[1] += 50.0  # Force Vaccine layer to saturate to protect the Core matrix
        else:
            # Standard Dynamic Scaling Path
            gated_logits[1] += current_entropy * 0.15     # Adjust vaccine scale based on randomness
            gated_logits[0] -= current_entropy * 0.02     # Compress core scale when noisy
        
        # Enforce strict sum-to-one constraint via Softmax
        normalized_scales = torch.softmax(gated_logits, dim=-1)
        
        L_H = normalized_scales[0].item()
        R_H = normalized_scales[1].item()
        V_H = normalized_scales[2].item()
        
        # Trigger metric collapse if terminal entropy threshold is breached
        force_collapse = current_entropy >= self.h_max
        return L_H, R_H, V_H, force_collapse
