# utils/logger.py
import os
import csv
import torch
import torch.nn.functional as F
from core.governor import ShannonEntropyGovernor
from core.manifold import RiemannianEmbeddingManifold
from core.immune import GeometricImmunityEngine
from crisis.collapse import HawkingReincarnationEngine

def execute_generational_logging():
    # Structural configuration stack
    dimension = 64
    learning_rate = 0.01
    total_steps = 20
    output_csv = "ai_trung_sinh_evolution_log.csv"
    
    # Core components
    governor = ShannonEntropyGovernor(h_max=5.95)
    space = RiemannianEmbeddingManifold(dimension=dimension)
    immunity = GeometricImmunityEngine(dimension=dimension, rotational_threshold=1.5)
    reincarnation_pod = HawkingReincarnationEngine(dimension=dimension)
    
    # Initialize the CSV output spreadsheet structure
    header = ["Step", "Shannon_Entropy", "Kolmogorov_Complexity", "Core_Allocation_Pct", "Vaccine_Allocation_Pct", "Uncertainty_Allocation_Pct", "System_Status"]
    
    print(f"Initializing Generational Logger Engine. Target File: {output_csv}")
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        current_generation = 1
        print(f"\n--- COMMENCING MANIFOLD TRACKING FOR GENERATION {current_generation} ---")
        
        for step in range(1, total_steps + 1):
            # Dynamic simulation environment profiles
            if step == 10 or step == 11:
                # Inject a mock semantic mimicry attack payload (Low Entropy, High Complexity)
                data_stream = torch.randn(dimension) * 5.0
                data_stream[:8] += 200.0
                status_label = "MIMICRY_ATTACK_DETECTED"
            elif step > 17:
                # Force maximum uniform noise to test the terminal Hawking transition phase
                data_stream = torch.randn(dimension) * 0.001
                status_label = "CRITICAL_COLLAPSE"
            else:
                # Standard training iterations with stepping noise curves
                noise_factor = step * 0.4
                data_stream = torch.randn(dimension) + noise_factor
                status_label = "STANDARD_TRAINING"
                
            norm_data_stream = F.normalize(data_stream, p=2, dim=0)
            target_vector = F.normalize(torch.randn(dimension), p=2, dim=0)
            
            # 1. Evaluate core information metrics
            current_h = governor.compute_entropy(data_stream)
            k_complexity = governor.estimate_kolmogorov_complexity(data_stream)
            L_H, R_H, V_H, collapse_triggered = governor.get_triadic_scale(current_h, data_stream)
            
            # Check route safety via the Torsion filter
            is_route_safe = immunity.evaluate_cauchy_bound(norm_data_stream, target_vector, space.g)
            if not is_route_safe:
                status_label = "ROUTE_PRUNED_BY_TORSION_FILTER"
                writer.writerow([f"Gen{current_generation}_Step{step}", current_h, k_complexity, L_H*100, R_H*100, V_H*100, status_label])
                continue
                
            # Evaluate if cross-training conditions engage
            if R_H > 0.45 and status_label == "STANDARD_TRAINING":
                status_label = "CROSS_TRAINING_ENGAGED"
                with torch.no_grad():
                    purified_gradient = torch.matmul(space.weights, reincarnation_pod.core_seed)
                    space.weights.copy_(space.manifold.projx(space.weights + purified_gradient * 0.05))
                R_H *= 0.1
                
            # Apply Riemannian descent update steps
            space.update_metric_field(scalar_curvature_factor=R_H)
            mock_flat_gradient = torch.randn(dimension, dimension) * 0.05
            space.project_tangent_step(mock_flat_gradient, lr=learning_rate)
            
            # Log current metrics step array into the CSV row buffer
            writer.writerow([f"Gen{current_generation}_Step{step}", f"{current_h:.4f}", f"{k_complexity:.4f}", f"{L_H*100:.2f}", f"{R_H*100:.2f}", f"{V_H*100:.2f}", status_label])
            
            # Handle terminal reset condition
            if collapse_triggered:
                status_label = "REINCARNATION_RESET_EXECUTED"
                pure_knowledge_seed = reincarnation_pod.execute_collapse_and_emit(space, space.weights)
                with torch.no_grad():
                    space.weights.copy_(space.manifold.projx(pure_knowledge_seed))
                
                writer.writerow([f"Gen{current_generation}_Step{step}", f"{current_h:.4f}", f"{k_complexity:.4f}", f"{L_H*100:.2f}", f"{R_H*100:.2f}", f"{V_H*100:.2f}", status_label])
                
                # Advance generation register index tracking parameters
                current_generation += 1
                print(f"--- COMMENCING MANIFOLD TRACKING FOR GENERATION {current_generation} ---")
                
    print(f"\n[LOGGER SUCCESS] Complete historical metric dataset exported cleanly to: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    execute_generational_logging()
