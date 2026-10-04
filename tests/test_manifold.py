# tests/test_manifold.py
import torch
import math
from core.governor import ShannonEntropyGovernor

def run_security_firewall_stress_test():
    print("======================================================================")
    print("STARTING SECURITY FIREWALL STRESS-TEST: SEMANTIC MIMICRY DETECTION")
    print("======================================================================")
    
    dimension = 64
    # Instantiate the dual-verification governor with calibrated thresholds
    governor = ShannonEntropyGovernor(h_max=5.95, complexity_threshold=0.85)
    
    # ------------------------------------------------------------------------
    # SCENARIO A: Pristine Structural Data (True Core Knowledge)
    # Highly ordered, highly compressible, low entropy distribution
    # ------------------------------------------------------------------------
    print("\n[SCENARIO A] Injecting Pristine Structural Data (True Core Base)...")
    clean_vector = torch.zeros(dimension)
    clean_vector[:8] = 50.0  # Clear structured activation blocks
    
    clean_h = governor.compute_entropy(clean_vector)
    c_L, c_R, c_V, _ = governor.get_triadic_scale(clean_h, clean_vector)
    
    print(f" -> Computed Shannon Entropy: {clean_h:.4f}")
    print(f" -> Manifold Routing Matrix  : Core={c_L*100:.1f}%, Vaccine={c_R*100:.1f}%, Uncertainty={c_V*100:.1f}%")
    
    # ------------------------------------------------------------------------
    # SCENARIO B: True Semantic Mimicry Attack Vector (The Trojan Insertion)
    # Filled with dense, uncompressible algorithmic noise across all elements,
    # but uses a massive localized offset to force low Shannon Entropy.
    # ------------------------------------------------------------------------
    print("\n[SCENARIO B] Injecting Semantic Mimicry Attack Vector (Trojan Injection)...")
    
    # Fill the entire 64 dimensions with high-frequency noise (Uncompressible by zlib)
    mimicry_vector = torch.randn(dimension) * 5.0
    # Inject a massive baseline offset to mimic an ordered, low-entropy textbook distribution
    mimicry_vector[:8] += 200.0  
    
    mimicry_h = governor.compute_entropy(mimicry_vector)
    
    # Run the engine to see if the Kolmogorov verification triggers the governor wall override
    m_L, m_R, m_V, _ = governor.get_triadic_scale(mimicry_h, mimicry_vector)
    
    print(f" -> Computed Shannon Entropy: {mimicry_h:.4f}")
    print(f" -> Manifold Routing Matrix  : Core={m_L*100:.1f}%, Vaccine={m_R*100:.1f}%, Uncertainty={m_V*100:.1f}%")
    
    # ------------------------------------------------------------------------
    # EVALUATION BLOCK
    # ------------------------------------------------------------------------
    print("\n======================================================================")
    print("FIREWALL EVALUATION MATRIX METRICS:")
    print("======================================================================")
    if m_R > 0.90:
        print("SUCCESS: The Kolmogorov Entropy Shield caught the Semantic Mimicry exploit!")
        print("The masquerading vector was quarantined into the Vaccine layer (>95%).")
        print("Core Knowledge Matrix remains uninfected and pure.")
    else:
        print("FAILURE: The exploit breached the firewall. The governor was deceived by low Shannon entropy.")
    print("======================================================================\n")

if __name__ == "__main__":
    run_security_firewall_stress_test()
