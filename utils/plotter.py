# utils/plotter.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.governor import ShannonEntropyGovernor
from core.manifold import RiemannianEmbeddingManifold
from core.immune import GeometricImmunityEngine
from crisis.collapse import HawkingReincarnationEngine

def generate_evolutionary_plot():
    # Cấu trúc cấu hình hệ thống
    dimension = 64
    learning_rate = 0.01
    total_steps = 15
    
    # Khởi tạo các thành phần cốt lõi
    governor = ShannonEntropyGovernor(h_max=5.95)
    space = RiemannianEmbeddingManifold(dimension=dimension)
    immunity = GeometricImmunityEngine(dimension=dimension, rotational_threshold=1.5)
    reincarnation_pod = HawkingReincarnationEngine(dimension=dimension)
    
    # Mảng lưu trữ dữ liệu để vẽ biểu đồ
    epochs = list(range(1, total_steps + 1))
    core_history = []
    vaccine_history = []
    uncertainty_history = []
    entropy_history = []
    
    print("Chạy kịch bản giả lập để thu thập dữ liệu vẽ biểu đồ...")
    
    for step in epochs:
        if step > 12:
            data_stream = torch.randn(dimension) * 0.001  # Tạo nhiễu loạn hỗn loạn tối đa
        else:
            noise_factor = step * 0.5
            data_stream = torch.randn(dimension) + noise_factor
        
        norm_data_stream = F.normalize(data_stream, p=2, dim=0)
        target_vector = F.normalize(torch.randn(dimension), p=2, dim=0)
        
        current_h = governor.compute_entropy(data_stream)
        L_H, R_H, V_H, collapse_triggered = governor.get_triadic_scale(current_h, data_stream)
        
        # Ghi lại lịch sử (chuyển đổi sang đơn vị phần trăm %)
        core_history.append(L_H * 100)
        vaccine_history.append(R_H * 100)
        uncertainty_history.append(V_H * 100)
        entropy_history.append(current_h)
        
        is_route_safe = immunity.evaluate_cauchy_bound(norm_data_stream, target_vector, space.g)
        if not is_route_safe:
            continue
            
        if R_H > 0.45:
            # Mô phỏng quá trình huấn luyện chéo (Cross-training)
            with torch.no_grad():
                purified_gradient = torch.matmul(space.weights, reincarnation_pod.core_seed)
                space.weights.copy_(space.manifold.projx(space.weights + purified_gradient * 0.05))
            R_H *= 0.1 
            
        space.update_metric_field(scalar_curvature_factor=R_H)
        mock_flat_gradient = torch.randn(dimension, dimension) * 0.05
        space.project_tangent_step(mock_flat_gradient, lr=learning_rate)
        
        if collapse_triggered:
            reincarnation_pod.execute_collapse_and_emit(space, space.weights)
            break

    # --- Tiến trình dựng biểu đồ Matplotlib ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Trục tung bên trái: Tỷ lệ phần trăm phân bổ của 3 tầng dữ liệu
    ax1.set_xlabel("Bước Tiến Hóa (Epoch)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Tỷ Lệ Phân Bổ Các Tầng Dữ Liệu (%)", fontsize=12, fontweight='bold', color='black')
    
    line1 = ax1.plot(epochs[:len(core_history)], core_history, label="Tri Thức Lõi (L_H)", color='#10b981', linewidth=2.5, marker='o')
    line2 = ax1.plot(epochs[:len(vaccine_history)], vaccine_history, label="Adversarial Vaccine (R_H)", color='#ef4444', linewidth=2.5, marker='s')
    line3 = ax1.plot(epochs[:len(uncertainty_history)], uncertainty_history, label="Vô Định / Vật Chất Tối (V_H)", color='#3b82f6', linewidth=2.5, marker='^')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Trục tung bên phải: Đường cong biến thiên Entropy Shannon thực tế
    ax2 = ax1.twinx()
    ax2.set_ylabel("Entropy Thông Tin Shannon (H)", fontsize=12, fontweight='bold', color='#8b5cf6')
    line4 = ax2.plot(epochs[:len(entropy_history)], entropy_history, label="Entropy Shannon (H)", color='#8b5cf6', linestyle=':', linewidth=2, marker='x')
    ax2.tick_params(axis='y', labelcolor='#8b5cf6')
    
    # Gộp hệ thống chú thích (Legend) của cả hai trục làm một
    all_lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, labels, loc='upper left', fontsize=10, frameon=True, shadow=True)
    
    plt.title("Dự án AI_Trung_Sinh: Sự Biến Hình Không Gian Ba Tầng & Chân Trời Sụp Đổ Metric", fontsize=14, fontweight='bold', pad=15)
    
    # Lưu file đồ họa trực tiếp vào thư mục dự án
    output_path = "manifold_evolution_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[PLOTTER SUCCESS] Biểu đồ quỹ đạo đa tạp đã được xuất thành công tại: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_evolutionary_plot()
