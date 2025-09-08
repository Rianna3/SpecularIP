class MeasurementValidator:
    def __init__(self):
        self.ground_truth_data = self.load_ground_truth()
    
    def validate_measurements(self, predicted_measurements, ground_truth_id):
        """验证测量精度"""
        if ground_truth_id not in self.ground_truth_data:
            return None
        
        gt = self.ground_truth_data[ground_truth_id]
        
        errors = {}
        for key in ['chest', 'waist', 'hip']:
            if key in predicted_measurements and key in gt:
                error = abs(predicted_measurements[key] - gt[key])
                relative_error = error / gt[key] * 100
                errors[key] = {
                    'absolute_error': error,
                    'relative_error': relative_error
                }
        
        return errors
    
    def visualize_measurements(self, vertices, measurements):
        """可视化测量结果"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D人体模型
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.5)
        ax1.set_title('3D Human Model')
        
        # 测量位置标记
        for region, height_ratio in [('chest', 0.35), ('waist', 0.55), ('hip', 0.75)]:
            y_pos = vertices[:, 1].min() + (vertices[:, 1].max() - vertices[:, 1].min()) * height_ratio
            ax1.axhline(y=y_pos, color='red', alpha=0.7, linestyle='--')
        
        # 测量结果
        ax2 = fig.add_subplot(132)
        regions = list(measurements.keys())
        values = list(measurements.values())
        ax2.bar(regions, values)
        ax2.set_title('Body Measurements')
        ax2.set_ylabel('Circumference (cm)')
        
        # 测量历史
        if hasattr(self, 'measurements_history'):
            ax3 = fig.add_subplot(133)
            steps = [m['step'] for m in self.measurements_history]
            chest_values = [m['measurements']['chest'] for m in self.measurements_history]
            ax3.plot(steps, chest_values, label='Chest')
            ax3.set_title('Measurement History')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Circumference (cm)')
            ax3.legend()
        
        plt.tight_layout()
        plt.savefig('measurements_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
