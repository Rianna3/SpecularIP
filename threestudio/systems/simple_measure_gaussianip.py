#!/usr/bin/env python3
"""
简化版GaussianIP模型测量脚本
直接从保存的顶点数据测量身体尺寸，不依赖复杂的SMPL-Anthropometry库
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, Any, Optional, List
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
import threestudio
from threestudio.systems.GaussianIP import GaussianIP
try:
    from threestudio.utils.body_measurements import GaussianBodyMeasurer
except ImportError:
    print("Warning: GaussianBodyMeasurer not found, using simple measurement only")
from datetime import datetime


class SimpleBodyMeasurer:
    """简化的身体测量器"""
    
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.gender = 'neutral'
        self.measurements = {}
        
    def load_model_data(self, file_path: str) -> bool:
        """加载模型数据（顶点文件或参数文件）"""
        try:
            print(f"加载文件: {file_path}")
            
            # 处理PLY文件
            if file_path.endswith('.ply'):
                return self.load_ply_file(file_path)
            
            # 处理PKL文件            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            
            # 检查是否为顶点文件
            if 'vertices' in data:
                print("检测到顶点文件")
                vertices = data['vertices']
                if isinstance(vertices, np.ndarray):
                    self.vertices = vertices
                else:
                    self.vertices = np.array(vertices)
                
                self.faces = data.get('faces', None)
                self.gender = data.get('gender', 'neutral')
                
                print(f"  - 性别: {self.gender}")
                print(f"  - 顶点数量: {self.vertices.shape[0]}")
                print(f"  - 参数修改次数: {data.get('modification_count', 'unknown')}")
                print(f"  - 训练步数: {data.get('training_steps', 'unknown')}")
                
            # 检查是否为参数文件
            elif 'betas' in data:
                print("检测到参数文件")
                self.gender = data.get('gender', 'neutral')
                print(f"  - 性别: {self.gender}")
                print(f"  - 参数: {list(data.keys())}")
                
                # 尝试从参数重建顶点
                success = self.reconstruct_from_params(data)
                if not success:
                    print("从参数重建顶点失败，将使用默认顶点")
                    self.vertices = self.create_default_vertices()
            
            else:
                print("未知的文件格式")
                return False
            
            return True
            
        except Exception as e:
            print(f"加载文件失败: {e}")
            return False
    
    def load_ply_file(self, file_path: str) -> bool:
        """加载PLY文件中的顶点数据"""
        try:
            # 先尝试文本模式读取头部信息
            vertex_count = 0
            is_binary = False
            header_end_pos = 0
            
            with open(file_path, 'rb') as f:
                # 读取头部
                header = b""
                while True:
                    line = f.readline()
                    header += line
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    
                    if line_str.startswith('element vertex'):
                        vertex_count = int(line_str.split()[2])
                    elif line_str.startswith('format'):
                        if 'binary' in line_str:
                            is_binary = True
                    elif line_str.startswith('end_header'):
                        header_end_pos = f.tell()
                        break
            
            print(f"PLY文件包含 {vertex_count} 个顶点, 格式: {'二进制' if is_binary else '文本'}")
            
            if is_binary:
                return self.load_binary_ply(file_path, vertex_count, header_end_pos)
            else:
                return self.load_text_ply(file_path, vertex_count, header_end_pos)
                
        except Exception as e:
            print(f"PLY文件加载失败: {e}")
            return False
    
    def load_binary_ply(self, file_path: str, vertex_count: int, header_end_pos: int) -> bool:
        """加载二进制PLY文件"""
        try:
            import struct
            vertices = []
            
            with open(file_path, 'rb') as f:
                f.seek(header_end_pos)
                
                # 假设每个顶点包含3个float（x, y, z）加上其他数据
                # 根据PLY格式，通常还包含颜色信息等
                for i in range(vertex_count):
                    # 读取位置信息（3个float = 12字节）
                    pos_data = f.read(12)
                    if len(pos_data) == 12:
                        x, y, z = struct.unpack('fff', pos_data)
                        vertices.append([x, y, z])
                        
                        # 跳过其余数据（根据实际PLY格式调整）
                        # 通常PLY文件还包含法向量、颜色等，这里假设跳过剩余数据
                        f.read(24)  # 跳过可能的法向量(3*4)和颜色(3*4)数据
                    else:
                        break
            
            if len(vertices) > 0:
                self.vertices = np.array(vertices)
                print(f"✓ 从二进制PLY文件加载顶点数据: {self.vertices.shape}")
                return True
            else:
                print("二进制PLY文件中没有找到顶点数据")
                return False
                
        except Exception as e:
            print(f"二进制PLY文件加载失败: {e}")
            # 尝试使用open3d作为备选方案
            try:
                import open3d as o3d
                mesh = o3d.io.read_point_cloud(file_path)
                if len(mesh.points) == 0:
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    if len(mesh.vertices) > 0:
                        self.vertices = np.asarray(mesh.vertices)
                    else:
                        return False
                else:
                    self.vertices = np.asarray(mesh.points)
                
                print(f"✓ 使用open3d加载PLY文件: {self.vertices.shape}")
                return True
            except ImportError:
                print("无法导入open3d库，无法加载二进制PLY文件")
                return False
            except Exception as e2:
                print(f"open3d加载也失败: {e2}")
                return False
    
    def load_text_ply(self, file_path: str, vertex_count: int, header_end_pos: int) -> bool:
        """加载文本PLY文件"""
        try:            
            vertices = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 查找vertex count
            header_lines = 0
            for i, line in enumerate(lines):
                if line.strip() == 'end_header':
                    header_lines = i + 1
                    break
                        
            # 读取顶点数据
            for i in range(header_lines, header_lines + vertex_count):
                if i < len(lines):
                    parts = lines[i].strip().split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        vertices.append([x, y, z])
            
            if len(vertices) > 0:
                self.vertices = np.array(vertices)
                print(f"✓ 从文本PLY文件加载顶点数据: {self.vertices.shape}")
                return True
            else:
                print("文本PLY文件中没有找到顶点数据")
                return False
                
        except Exception as e:
            print(f"文本PLY文件加载失败: {e}")
            return False

    def reconstruct_from_params(self, params_data: Dict[str, Any]) -> bool:
        """从参数重建顶点（如果可能的话）"""
        try:
            # 尝试使用SMPL-X重建
            import smplx
            
            model_path = "models_smplx_v1_1/models"
            if not os.path.exists(model_path):
                print(f"SMPL-X模型路径不存在: {model_path}")
                return False
                
            gender = params_data.get('gender', 'neutral')
            
            model = smplx.create(
                model_path,
                model_type='smplx', 
                gender=gender,
                ext='npz',
                num_betas=10,
                use_face_contour=False
            )
            
            # 准备参数
            betas = torch.tensor(params_data.get('betas', np.zeros(10)), dtype=torch.float32).unsqueeze(0)
            body_pose = torch.tensor(params_data.get('body_pose', np.zeros((21, 3))), dtype=torch.float32).unsqueeze(0)
            global_orient = torch.tensor(params_data.get('global_orient', np.zeros((1, 3))), dtype=torch.float32).unsqueeze(0)
            
            output = model(betas=betas, body_pose=body_pose, global_orient=global_orient)
            self.vertices = output.vertices.squeeze().detach().numpy()
            self.faces = model.faces
            
            print(f"从参数成功重建顶点: {self.vertices.shape}")
            return True
            
        except Exception as e:
            print(f"从参数重建失败: {e}")
            return False
    
    def create_default_vertices(self) -> np.ndarray:
        """创建默认的SMPL-X顶点（用于测试）"""
        # 创建一个简化的人体模型顶点
        vertices = np.random.randn(10475, 3) * 0.1
        
        # 设置基本的人体比例
        # 头部区域
        head_indices = range(0, 1000)
        for i in head_indices:
            vertices[i, 1] += 1.6  # 头部高度
            
        # 躯干区域  
        torso_indices = range(1000, 5000)
        for i in torso_indices:
            vertices[i, 1] += 0.8  # 躯干高度
            
        # 腿部区域
        leg_indices = range(5000, 10475)
        for i in leg_indices:
            vertices[i, 1] -= 0.4  # 腿部位置
        
        print("使用默认顶点进行测量")
        return vertices
    
    def measure_body(self) -> Dict[str, float]:
        """测量身体尺寸"""
        if self.vertices is None:
            print("错误: 没有顶点数据")
            return {}
        
        print("开始测量身体尺寸...")
        
        measurements = {}
        
        try:
            # 1. 身高
            height = self.measure_height()
            measurements['height'] = height
            print(f"  身高: {height:.2f} cm")
            
            # 2. 胸围
            chest = self.measure_chest_circumference()
            measurements['chest_circumference'] = chest
            print(f"  胸围: {chest:.2f} cm")
            
            # 3. 腰围
            waist = self.measure_waist_circumference()
            measurements['waist_circumference'] = waist
            print(f"  腰围: {waist:.2f} cm")
            
            # 4. 臀围
            hip = self.measure_hip_circumference()
            measurements['hip_circumference'] = hip
            print(f"  臀围: {hip:.2f} cm")
            
            # 5. 肩宽
            shoulder_width = self.measure_shoulder_width()
            measurements['shoulder_width'] = shoulder_width
            print(f"  肩宽: {shoulder_width:.2f} cm")
            
            # 6. 腿长
            leg_length = self.measure_leg_length()
            measurements['leg_length'] = leg_length
            print(f"  腿长: {leg_length:.2f} cm")
            
            self.measurements = measurements
            return measurements
            
        except Exception as e:
            print(f"测量过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def measure_height(self) -> float:
        """测量身高"""
        y_coords = self.vertices[:, 1]
        height = (y_coords.max() - y_coords.min()) * 100  # 转换为厘米
        return height
    
    def measure_chest_circumference(self) -> float:
        """测量胸围"""
        return self.measure_circumference_at_height(0.35, tolerance=0.05, name="胸部")
    
    def measure_waist_circumference(self) -> float:
        """测量腰围"""
        return self.measure_circumference_at_height(0.55, tolerance=0.05, name="腰部")
    
    def measure_hip_circumference(self) -> float:
        """测量臀围"""
        return self.measure_circumference_at_height(0.75, tolerance=0.05, name="臀部")
    
    def measure_circumference_at_height(self, height_ratio: float, tolerance: float = 0.05, name: str = "body part") -> float:
        """在指定高度测量周长"""
        try:
            # 获取Y坐标范围
            y_coords = self.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            total_height = y_max - y_min
            
            # 计算目标高度
            target_y = y_min + total_height * height_ratio
            tolerance_abs = total_height * tolerance
            
            # 筛选在目标高度附近的顶点
            mask = (y_coords >= target_y - tolerance_abs) & (y_coords <= target_y + tolerance_abs)
            selected_vertices = self.vertices[mask]
            
            if len(selected_vertices) < 3:
                print(f"  警告: {name}区域顶点太少({len(selected_vertices)})，使用估算值")
                return 80.0  # 返回估算值
            
            # 投影到XZ平面
            points_2d = selected_vertices[:, [0, 2]]
            
            # 计算凸包
            try:
                hull = ConvexHull(points_2d)
                hull_points = points_2d[hull.vertices]
                
                # 计算周长
                circumference = 0.0
                for i in range(len(hull_points)):
                    p1 = hull_points[i]
                    p2 = hull_points[(i + 1) % len(hull_points)]
                    circumference += euclidean(p1, p2)
                
                # 转换为厘米
                circumference_cm = circumference * 100
                
                return circumference_cm
                
            except Exception as e:
                print(f"  警告: {name}凸包计算失败({e})，使用边界框估算")
                # 使用边界框作为后备方案
                x_range = points_2d[:, 0].max() - points_2d[:, 0].min()
                z_range = points_2d[:, 1].max() - points_2d[:, 1].min()
                estimated_circumference = 2 * (x_range + z_range) * 100
                return estimated_circumference
                
        except Exception as e:
            print(f"  错误: 测量{name}周长失败({e})")
            return 0.0
    
    def measure_shoulder_width(self) -> float:
        """测量肩宽"""
        try:
            # 寻找肩部区域（通常在身体上部）
            y_coords = self.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            total_height = y_max - y_min
            
            # 肩部通常在身体上部约20%的位置
            shoulder_y = y_min + total_height * 0.2
            tolerance = total_height * 0.1
            
            # 筛选肩部区域的顶点
            mask = (y_coords >= shoulder_y - tolerance) & (y_coords <= shoulder_y + tolerance)
            shoulder_vertices = self.vertices[mask]
            
            if len(shoulder_vertices) < 3:
                return 40.0  # 估算值
            
            # 找到X坐标的最大范围（左右肩膀）
            x_coords = shoulder_vertices[:, 0]
            shoulder_width = (x_coords.max() - x_coords.min()) * 100  # 转换为厘米
            
            return shoulder_width
            
        except Exception as e:
            print(f"  错误: 测量肩宽失败({e})")
            return 0.0
    
    def measure_leg_length(self) -> float:
        """测量腿长"""
        try:
            # 假设腿部从臀部到脚踝
            y_coords = self.vertices[:, 1]
            y_min, y_max = y_coords.min(), y_coords.max()
            total_height = y_max - y_min
            
            # 腿长大约是总身高的一半
            leg_length = total_height * 0.5 * 100  # 转换为厘米
            
            return leg_length
            
        except Exception as e:
            print(f"  错误: 测量腿长失败({e})")
            return 0.0
    
    def save_measurements(self, output_path: str):
        """保存测量结果"""
        try:
            result = {
                'measurements': self.measurements,
                'gender': self.gender,
                'vertex_count': len(self.vertices) if self.vertices is not None else 0,
                'measurement_time': str(np.datetime64('now')),
                'note': 'Measured using simplified GaussianIP measurement tool'
            }
            
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"✓ 测量结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"保存结果失败: {e}")


def find_latest_model_files(logs_dir: str = "logs"):
    """查找最新的模型文件"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"日志目录不存在: {logs_dir}")
        return []
    
    # 查找最新的时间戳目录
    timestamp_dirs = [d for d in logs_path.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        print(f"在 {logs_dir} 中没有找到时间戳目录")
        return []
    
    latest_dir = max(timestamp_dirs, key=lambda x: x.name)
    print(f"找到最新的日志目录: {latest_dir}")
    
    # 查找模型文件
    model_files = []
    
    # 查找SMPL-X模型目录
    smplx_dir = latest_dir / "smplx_model"
    if smplx_dir.exists():
        vertices_files = list(smplx_dir.glob("*_vertices.pkl"))
        params_files = list(smplx_dir.glob("*_parameters.pkl"))
        updated_files = list(smplx_dir.glob("*_updated.pkl"))  # 查找updated文件
        all_pkl_files = list(smplx_dir.glob("*.pkl"))  # 查找所有pkl文件
        
        model_files.extend(vertices_files)
        model_files.extend(params_files)
        model_files.extend(updated_files)
        model_files.extend(all_pkl_files)
    
    # 查找其他可能的模型文件
    other_pkl_files = list(latest_dir.glob("*.pkl"))
    other_ply_files = list(latest_dir.glob("*.ply"))
    model_files.extend(other_pkl_files)
    model_files.extend(other_ply_files)
    
    # 去重
    model_files = list(set(model_files))
    
    return model_files


@threestudio.register("simple-measure-gaussianip-system")
class SimpleMeasureGaussianIPSystem(GaussianIP):
    """
    专门用于身体测量的GaussianIP系统
    """
    
    def configure(self):
        super().configure()
        
        # 初始化测量器
        from threestudio.utils.body_measurements import GaussianBodyMeasurer
        self.body_measurer = GaussianBodyMeasurer(
            model_type=self.cfg.get('measurement_config', {}).get('model_type', 'smplx'),
            measurement_types=self.cfg.get('measurement_config', {}).get('measurement_types', ['bust', 'waist', 'hip', 'height']),
            gender=self.cfg.get('measurement_config', {}).get('gender', 'neutral')
        )
        
        # 测量历史记录
        self.measurement_history = []

    @torch.no_grad()
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
        # 执行身体测量
        if self.cfg.get('measurement_config', {}).get('save_measurements', True):
            measurements = self.measure_body()
            self.save_measurement_results(measurements)

    @torch.no_grad()
    def measure_body(self):
        """
        执行身体测量
        """
        # 穿衣测量
        clothed_measurements = self.body_measurer.measure_from_gaussian_model(
            self.gaussian_model, 
            remove_clothing=False
        )
        
        # 脱衣测量（如果启用）
        naked_measurements = None
        if self.cfg.get('measurement_config', {}).get('remove_clothing_for_measurement', True):
            naked_measurements = self.body_measurer.measure_from_gaussian_model(
                self.gaussian_model, 
                remove_clothing=True
            )
        
        # 组合结果
        results = {
            'step': self.global_step,
            'clothed': clothed_measurements,
            'naked': naked_measurements,
            'comparison': None
        }
        
        # 计算差异
        if naked_measurements is not None:
            results['comparison'] = self.body_measurer.compare_measurements(
                naked_measurements, clothed_measurements
            )
        
        return results

    def save_measurement_results(self, measurements):
        """
        保存测量结果
        """
        # 保存到实验目录
        save_dir = os.path.join(self.get_save_dir(), "measurements")
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(save_dir, f"measurements_step_{self.global_step}_{timestamp}.json")
        
        self.body_measurer.save_measurements(measurements, filepath)
        
        # 记录到训练日志
        if measurements.get('naked'):
            for key, value in measurements['naked'].items():
                if isinstance(value, (int, float)):
                    self.log(f"measurement_naked/{key}", value)
        
        if measurements.get('clothed'):
            for key, value in measurements['clothed'].items():
                if isinstance(value, (int, float)):
                    self.log(f"measurement_clothed/{key}", value)

    @torch.no_grad()
    def export_final_measurements(self):
        """
        导出最终的测量结果
        """
        print("Exporting final body measurements...")
        
        final_measurements = self.measure_body()
        
        # 保存到主目录
        save_path = os.path.join(self.get_save_dir(), "final_body_measurements.json")
        self.body_measurer.save_measurements(final_measurements, save_path)
        
        # 打印结果
        self.print_measurement_summary(final_measurements)
        
        return final_measurements

    def print_measurement_summary(self, measurements):
        """
        打印测量结果摘要
        """
        print("\n" + "="*50)
        print("BODY MEASUREMENT SUMMARY")
        print("="*50)
        
        if measurements.get('naked'):
            print("\nNaked Body Measurements:")
            for key, value in measurements['naked'].items():
                if isinstance(value, (int, float)) and key != 'height':
                    print(f"  {key.capitalize()}: {value:.1f} cm")
                elif key == 'height':
                    print(f"  {key.capitalize()}: {value:.1f} cm")
        
        if measurements.get('comparison'):
            print("\nClothing Effect (Clothed - Naked):")
            for key, comp in measurements['comparison'].items():
                if isinstance(comp, dict) and 'absolute_diff' in comp:
                    print(f"  {key.capitalize()}: +{comp['absolute_diff']:.1f} cm ({comp['percent_diff']:.1f}%)")
        
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='简化版GaussianIP模型测量工具')
    parser.add_argument('--model_file', type=str, help='模型文件路径（顶点或参数文件）')
    parser.add_argument('--logs_dir', type=str, default='logs', help='日志目录路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--auto', action='store_true', help='自动查找最新的模型文件')
    
    args = parser.parse_args()
    
    # 创建测量器
    measurer = SimpleBodyMeasurer()
    
    # 确定要使用的文件
    model_file = args.model_file
    
    if args.auto or not model_file:
        print("自动查找最新的模型文件...")
        model_files = find_latest_model_files(args.logs_dir)
        
        if not model_files:
            print("没有找到模型文件")
            return 1
        
        ply_files = [f for f in model_files if f.name.endswith('.ply')]
        vertices_files = [f for f in model_files if 'vertices' in f.name]
        params_files = [f for f in model_files if 'parameters' in f.name]
        
        if ply_files:
            model_file = str(ply_files[0])
            print(f"选择PLY文件: {model_file}")
        elif vertices_files:
            model_file = str(vertices_files[0])
        else:
            model_file = str(model_files[0])
        
        print(f"选择文件: {model_file}")
    
    if not model_file or not Path(model_file).exists():
        print("错误: 没有找到有效的模型文件")
        print("请指定 --model_file 或使用 --auto 自动查找")
        return 1
    
    # 加载模型数据
    success = measurer.load_model_data(model_file)
    if not success:
        print("加载模型失败")
        return 1
    
    # 执行测量
    print("\n开始身体测量...")
    measurements = measurer.measure_body()
    
    if not measurements:
        print("测量失败")
        return 1
    
    # 打印结果摘要
    print(f"\n=== 测量结果摘要 ===")
    print(f"身高: {measurements.get('height', 0):.1f} cm")
    print(f"胸围: {measurements.get('chest_circumference', 0):.1f} cm")
    print(f"腰围: {measurements.get('waist_circumference', 0):.1f} cm") 
    print(f"臀围: {measurements.get('hip_circumference', 0):.1f} cm")
    print(f"肩宽: {measurements.get('shoulder_width', 0):.1f} cm")
    print(f"腿长: {measurements.get('leg_length', 0):.1f} cm")
    
    # 保存结果
    if args.output:
        measurer.save_measurements(args.output)
    else:
        # 自动生成输出文件名
        timestamp = str(np.datetime64('now')).replace(':', '-').replace('.', '-')
        output_file = f"gaussianip_simple_measurements_{timestamp}.json"
        measurer.save_measurements(output_file)
    
    print("\n🎉 测量完成!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())