# main.py
import torch
import torch.nn.functional as F
from core.governor import ShannonEntropyGovernor
from core.manifold import RiemannianEmbeddingManifold
from core.immune import GeometricImmunityEngine
from crisis.collapse import HawkingReincarnationEngine
from utils.checkpoint import ManifoldCheckpointManager  # <-- Imported checkpoint manager

def execute_cross_training(space, data_stream, core_seed):
    """Cross-Training Module: Forces core knowledge and adversarial metrics to cross-evolve"""
    print(" -> [IMMUNE SYSTEM] Activating Cross-Training Regime (Self-Play Optimization)...")
    with torch.no_grad():
        purified_gradient = torch.matmul(space.weights, core_seed)
        space.weights.copy_(space.manifold.projx(space.weights + purified_gradient * 0.05))
    print(" -> [EVOLUTION SUCCESS] Model successfully assimilated adversarial noise into an evolutionary antibody.")

def run_evolutionary_loop():
    dimension = 64
    learning_rate = 0.01
    total_steps = 15
    current_generation = 1 # Active generation tracker
    
    governor = ShannonEntropyGovernor(h_max=5.95)
    space = RiemannianEmbeddingManifold(dimension=dimension)
    immunity = GeometricImmunityEngine(dimension=dimension, rotational_threshold=1.5)
    reincarnation_pod = HawkingReincarnationEngine(dimension=dimension)
    ckpt_manager = ManifoldCheckpointManager() # Initialize persistence unit
    
    print("Initializing Project: AI_Trung_Sinh (Riemannian Framework Activated)\n")
    print(f"--- COMMENCING MANIFOLD TRACKING FOR GENERATION {current_generation} ---")
    
    for step in range(1, total_steps + 1):
        if step > 12:
            print(" -> [CRITICAL INFUSION] Injecting absolute maximum uniform noise (Total Structural Chaos)...")
            data_stream = torch.randn(dimension) * 0.001
        else:
            noise_factor = step * 0.5
            data_stream = torch.randn(dimension) + noise_factor
        
        norm_data_stream = F.normalize(data_stream, p=2, dim=0)
        target_vector = F.normalize(torch.randn(dimension), p=2, dim=0)
        
        current_h = governor.compute_entropy(data_stream)
        L_H, R_H, V_H, collapse_triggered = governor.get_triadic_scale(current_h, data_stream)
        
        print(f"[Step {step:02d}] Entropy: {current_h:.4f} | Mix -> Core: {L_H*100:.1f}%, Vaccine: {R_H*100:.1f}%, Uncertainty: {V_H*100:.1f}%")
        
        is_route_safe = immunity.evaluate_cauchy_bound(norm_data_stream, target_vector, space.g)
        
        if not is_route_safe:
            print(f" -> [IMMUNE ALERT] Adversarial routing mismatch detected at step {step}. Connection pruned.")
            continue
            
        if R_H > 0.45:
            execute_cross_training(space, norm_data_stream, reincarnation_pod.core_seed)
            R_H *= 0.1 
            
        space.update_metric_field(scalar_curvature_factor=R_H)
        mock_flat_gradient = torch.randn(dimension, dimension) * 0.05
        space.project_tangent_step(mock_flat_gradient, lr=learning_rate)
        
        if collapse_triggered:
            pure_knowledge_seed = reincarnation_pod.execute_collapse_and_emit(space, space.weights)
            with torch.no_grad():
                space.weights.copy_(space.manifold.projx(pure_knowledge_seed))
                
            # EXECUTE PERSISTENCE: Save the newly minted structural genome immediately upon emission
            ckpt_manager.save_generational_state(
                generation=current_generation,
                step=step,
                weights=space.weights,
                core_seed=reincarnation_pod.core_seed,
                metric_g=space.g
            )
            
            print(f"System successfully initialized into Generation {current_generation + 1} from clean structural genes.")
            current_generation += 1
            print(f"\n--- COMMENCING MANIFOLD TRACKING FOR GENERATION {current_generation} ---")
            break

if __name__ == "__main__":
    run_evolutionary_loop()
