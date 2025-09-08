import torch
import numpy as np
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import trimesh

try:
    from SMPL_Anthropometry import BodyMeasurements
    SMPL_ANTHROPOMETRY_AVAILABLE = True
except ImportError:
    print("Warning: SMPL-Anthropometry not found. Using fallback ellipse-based measurement")
    SMPL_ANTHROPOMETRY_AVAILABLE = False

class BodyMeasurementCalculator:
    def __init__(self):
        # SMPL-X测量点定义（基于SMPL-X标准）
        self.measurement_regions = {
            'chest': {
                'height_ratio': 0.35,  # 胸部高度比例（从颈部到腰部）
                'tolerance': 0.02      # 截面厚度容差
            },
            'waist': {
                'height_ratio': 0.55,  # 腰部高度比例
                'tolerance': 0.02
            },
            'hip': {
                'height_ratio': 0.75,  # 臀部高度比例
                'tolerance': 0.02
            }
        }

    def calculate_chest_circumference(self, vertices):
        """计算胸围"""
        # 获取胸部高度范围的顶点
        chest_vertices = self.extract_region_vertices(
            vertices, 
            height_ratio=self.measurement_regions['chest']['height_ratio'],
            tolerance=self.measurement_regions['chest']['tolerance']
        )
        
        # 计算胸部截面周长
        circumference = self.calculate_circumference_2d(chest_vertices)
        return circumference
    
    def calculate_waist_circumference(self, vertices):
        """计算腰围"""
        waist_vertices = self.extract_region_vertices(
            vertices,
            height_ratio=self.measurement_regions['waist']['height_ratio'],
            tolerance=self.measurement_regions['waist']['tolerance']
        )
        
        circumference = self.calculate_circumference_2d(waist_vertices)
        return circumference

    def calculate_hip_circumference(self, vertices):
        """计算臀围"""
        hip_vertices = self.extract_region_vertices(
            vertices,
            height_ratio=self.measurement_regions['hip']['height_ratio'],
            tolerance=self.measurement_regions['hip']['tolerance']
        )
        
        circumference = self.calculate_circumference_2d(hip_vertices)
        return circumference
    
    def extract_region_vertices(self, vertices, height_ratio, tolerance):
        """提取指定高度区域的顶点"""
        # 计算人体高度范围
        min_y, max_y = vertices[:, 1].min(), vertices[:, 1].max()
        height = max_y - min_y
        
        # 计算目标高度
        target_height = min_y + height * height_ratio
        
        # 提取高度范围内的顶点
        mask = (vertices[:, 1] >= target_height - tolerance * height) & \
               (vertices[:, 1] <= target_height + tolerance * height)
        
        return vertices[mask]

    def calculate_circumference_2d(self, vertices_3d):
        """计算2D截面周长"""
        # 投影到XZ平面（去除Y轴高度信息）
        vertices_2d = vertices_3d[:, [0, 2]]  # 取X和Z坐标
        
        if len(vertices_2d) < 3:
            return 0.0
        
        # 使用凸包算法计算周长
        try:
            hull = ConvexHull(vertices_2d)
            circumference = hull.area  # 这里需要计算周长，不是面积
            
            # 计算凸包周长
            hull_points = vertices_2d[hull.vertices]
            circumference = 0.0
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                circumference += np.linalg.norm(p2 - p1)
            
            return circumference
        except:
            # 如果凸包失败，使用简单的边界框估计
            return self.estimate_circumference_from_bounds(vertices_2d)
    
    def estimate_circumference_from_bounds(self, vertices_2d):
        """从边界框估计周长"""
        min_x, max_x = vertices_2d[:, 0].min(), vertices_2d[:, 0].max()
        min_z, max_z = vertices_2d[:, 1].min(), vertices_2d[:, 1].max()
        
        width = max_x - min_x
        depth = max_z - min_z
        
        # 使用椭圆周长近似公式
        a, b = width / 2, depth / 2
        circumference = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        
        return circumference

    def calculate_measurements(self, betas, body_pose, smpl_model=None):
        """
        根据SMPL(-X)参数计算胸围、腰围、臀围。
        Args:
            betas: SMPL shape参数 (Tensor or ndarray)
            body_pose: SMPL pose参数 (Tensor or ndarray)
            smpl_model: 可选，外部传入的SMPL/SMPL-X模型实例，需有forward方法
        Returns:
            dict: {'chest': float, 'waist': float, 'hip': float}
        """
        if smpl_model is None:
            # 无法生成mesh，返回0
            return {'chest': 0.0, 'waist': 0.0, 'hip': 0.0}
        # 生成mesh顶点
        with torch.no_grad():
            output = smpl_model(betas=betas.unsqueeze(0), body_pose=body_pose.unsqueeze(0))
            vertices = output.vertices[0].cpu().numpy()
        chest = self.calculate_chest_circumference(vertices)
        waist = self.calculate_waist_circumference(vertices)
        hip = self.calculate_hip_circumference(vertices)
        return {'chest': chest, 'waist': waist, 'hip': hip}

class GaussianBodyMeasurer:
    """
    从GaussianIP模型中提取身体测量数据的工具类
    """
    
    def __init__(self, 
                 model_type='smplx',
                 measurement_types=['bust', 'waist', 'hip', 'height'],
                 gender='neutral'):
        self.model_type = model_type
        self.measurement_types = measurement_types
        self.gender = gender
        
        if SMPL_ANTHROPOMETRY_AVAILABLE:
            self.measurer = BodyMeasurements(
                model_type=model_type,
                measurement_types=measurement_types
            )
        else:
            self.measurer = None
            print("Using fallback ellipse-based measurement")

    def measure_from_gaussian_model(self, gaussian_model, remove_clothing=True):
        """
        从高斯模型中提取并测量身体尺寸
        
        Args:
            gaussian_model: 训练好的高斯模型
            remove_clothing: 是否移除衣物进行测量
        
        Returns:
            Dict: 包含各项测量结果的字典
        """
        # Step 1: 提取点云
        if remove_clothing and hasattr(gaussian_model, 'get_body_mask'):
            body_mask = gaussian_model.get_body_mask()
            vertices = gaussian_model.get_xyz[body_mask].detach().cpu().numpy()
        else:
            vertices = gaussian_model.get_xyz.detach().cpu().numpy()
        
        # Step 2: 点云预处理
        vertices = self.preprocess_pointcloud(vertices)
        
        # Step 3: 执行测量
        if self.measurer is not None:
            measurements = self.precise_measurement(vertices)
        else:
            measurements = self.fallback_measurement(vertices)
        
        # Step 4: 添加元数据
        measurements['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'clothing_removed': remove_clothing,
            'total_points': len(vertices),
            'method': 'smpl_anthropometry' if self.measurer else 'ellipse_approximation'
        }
        
        return measurements

    def preprocess_pointcloud(self, vertices):
        """
        点云预处理：去噪、平滑
        """
        # 移除离群点
        vertices = self.remove_outliers(vertices)
        return vertices

    def remove_outliers(self, vertices, std_threshold=2.0):
        """
        使用统计方法移除离群点
        """
        if len(vertices) < 100:
            return vertices
            
        # 计算每个点到质心的距离
        centroid = np.mean(vertices, axis=0)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        
        # 移除超过阈值的点
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        valid_mask = distances < (mean_dist + std_threshold * std_dist)
        
        return vertices[valid_mask]

    def precise_measurement(self, vertices):
        """
        使用SMPL-Anthropometry进行精确测量
        """
        try:
            # 创建简单的面片用于测量
            faces = self.create_simple_faces(vertices)
            
            measurements = self.measurer.measure_all(
                vertices=vertices,
                faces=faces,
                gender=self.gender
            )
            
            # 转换单位（通常从米转换为厘米）
            for key in measurements:
                if isinstance(measurements[key], (int, float)):
                    measurements[key] = measurements[key] * 100  # m to cm
            
            return measurements
        except Exception as e:
            print(f"SMPL measurement failed: {e}, falling back to ellipse method")
            return self.fallback_measurement(vertices)

    def create_simple_faces(self, vertices):
        """
        为点云创建简单的三角面片
        """
        try:
            import trimesh
            # 创建凸包mesh作为近似
            hull = trimesh.convex.convex_hull(vertices)
            return hull.faces
        except ImportError:
            # Fallback: 返回虚拟面片
            n_points = len(vertices)
            if n_points >= 3:
                return np.array([[0, 1, 2]])
            else:
                return np.array([[0, 0, 0]])

    def fit_ellipse_circumference(self, points_2d, remove_outliers=True, clothing_compensation=0.9):
        """
        改进的椭圆拟合并计算周长
        
        Args:
            points_2d: 2D截面点 [N, 2]
            remove_outliers: 是否移除离群点
            clothing_compensation: 衣物厚度补偿系数 (0.8-0.95)
        """
        if len(points_2d) < 5:
            return 0.0
            
        try:
            # 1. 移除离群点（基于距离中心的统计距离）
            if remove_outliers and len(points_2d) > 20:
                points_2d = self._remove_cross_section_outliers(points_2d)
                
            if len(points_2d) < 5:
                return 0.0
            
            # 2. 改进的椭圆拟合
            x = points_2d[:, 0]
            z = points_2d[:, 1]
            
            # 计算椭圆参数（使用凸包边界而非简单最值）
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points_2d)
                hull_points = points_2d[hull.vertices]
                
                # 基于凸包计算更精确的椭圆轴
                hull_x = hull_points[:, 0]
                hull_z = hull_points[:, 1]
                
                # 计算主轴和次轴
                a = (np.max(hull_x) - np.min(hull_x)) / 2  # 半长轴
                b = (np.max(hull_z) - np.min(hull_z)) / 2  # 半短轴
                
            except (ImportError, Exception):
                # Fallback: 使用简单边界
                a = (np.max(x) - np.min(x)) / 2
                b = (np.max(z) - np.min(z)) / 2
            
            # 3. 椭圆周长计算（Ramanujan公式）
            if a <= 0 or b <= 0:
                return 0.0
                
            h = ((a - b) / (a + b)) ** 2
            circumference = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
            
            # 4. 应用衣物厚度补偿
            compensated_circumference = circumference * clothing_compensation
            
            return compensated_circumference * 2  # 转换为直径周长
            
        except Exception as e:
            print(f"Ellipse fitting failed: {e}")
            # 最终fallback：简单凸包周长
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(points_2d)
                perimeter = 0
                hull_points = points_2d[hull.vertices]
                for i in range(len(hull_points)):
                    p1 = hull_points[i]
                    p2 = hull_points[(i + 1) % len(hull_points)]
                    perimeter += np.linalg.norm(p2 - p1)
                return perimeter * clothing_compensation
            except:
                return np.sum(np.linalg.norm(np.diff(points_2d, axis=0), axis=1))

    def _remove_cross_section_outliers(self, points_2d, std_threshold=1.5):
        """
        移除截面中的离群点
        """
        if len(points_2d) < 10:
            return points_2d
            
        # 计算到质心的距离
        centroid = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - centroid, axis=1)
        
        # 移除距离异常的点
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        valid_mask = distances < (mean_dist + std_threshold * std_dist)
        
        # 至少保留50%的点
        if np.sum(valid_mask) < len(points_2d) * 0.5:
            # 如果过滤太严格，保留距离最近的50%点
            sorted_indices = np.argsort(distances)
            keep_count = max(5, len(points_2d) // 2)
            valid_mask = np.zeros(len(points_2d), dtype=bool)
            valid_mask[sorted_indices[:keep_count]] = True
        
        return points_2d[valid_mask]

    def fallback_measurement(self, vertices, apply_conservative_filtering=True):
        """
        重写的保守测量方法：基于核心身体点云的精确截面分析
        
        Args:
            vertices: 顶点数组 [N, 3] (已修正坐标系，Y轴为身高)
            apply_conservative_filtering: 是否应用保守过滤
        """
        measurements = {}
        
        # 1. 基本尺寸分析
        model_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        measurements['height'] = model_height
        
        if len(vertices) < 100:
            print("Warning: Too few vertices for reliable measurement")
            return {'height': model_height, 'chest': 0, 'waist': 0, 'hip': 0, 'bust': 0}
        
        print(f"Processing {len(vertices)} vertices, model height: {model_height:.4f} units")
        
        # 2. 核心身体识别（移除离群点和衣物）
        if apply_conservative_filtering:
            core_vertices = self._extract_core_body_points(vertices)
            print(f"Extracted {len(core_vertices)} core body points")
        else:
            core_vertices = vertices
            
        # 3. 重新计算截面位置（基于核心身体）
        base_y = np.min(core_vertices[:, 1])
        height_range = np.max(core_vertices[:, 1]) - base_y
        
        # 优化的人体测量比例（避免重叠）
        measurement_levels = {
            'hip': 0.40,     # 臀围：40%身高处（最宽处）
            'waist': 0.58,   # 腰围：58%身高处（最细处）
            'chest': 0.75    # 胸围：75%身高处（胸部最突出处）
        }
        
        print(f"Height range: {height_range:.4f}, Base Y: {base_y:.4f}")
        
        # 4. 逐个测量各部位（从下到上）
        results = {}
        for part, ratio in measurement_levels.items():
            y_level = base_y + ratio * height_range
            circumference = self._measure_circumference_at_level(
                core_vertices, y_level, height_range, part
            )
            results[part] = circumference
            print(f"{part.upper()} at {ratio:.0%} height (Y={y_level:.3f}): {circumference:.4f} model units")
        
        # 5. 人体比例合理性修正
        results = self._apply_anatomical_constraints(results)
        
        # 6. 返回结果
        measurements.update(results)
        measurements['bust'] = results['chest']  # 胸围别名
        
        return measurements

    def _apply_anatomical_constraints(self, raw_measurements):
        """
        应用解剖学约束，确保人体比例合理
        """
        chest = raw_measurements.get('chest', 0)
        waist = raw_measurements.get('waist', 0) 
        hip = raw_measurements.get('hip', 0)
        
        print(f"Raw measurements before constraints - Chest: {chest:.2f}, Waist: {waist:.2f}, Hip: {hip:.2f}")
        
        if chest <= 0 or waist <= 0 or hip <= 0:
            return raw_measurements
        
        # 应用人体比例约束
        corrected = {}
        
        # 1. 腰围应该是最小的
        min_measurement = min(chest, waist, hip)
        if waist != min_measurement:
            print(f"Adjusting waist: {waist:.2f} -> {min_measurement:.2f}")
            corrected['waist'] = min_measurement
        else:
            corrected['waist'] = waist
            
        # 2. 胸围约束：应该比腰围大5-25%
        min_chest = corrected['waist'] * 1.05
        max_chest = corrected['waist'] * 1.25
        if chest < min_chest:
            corrected['chest'] = min_chest
            print(f"Chest too small, adjusting: {chest:.2f} -> {min_chest:.2f}")
        elif chest > max_chest:
            corrected['chest'] = max_chest  
            print(f"Chest too large, adjusting: {chest:.2f} -> {max_chest:.2f}")
        else:
            corrected['chest'] = chest
            
        # 3. 臀围约束：应该比腰围大10-30%
        min_hip = corrected['waist'] * 1.10
        max_hip = corrected['waist'] * 1.30
        if hip < min_hip:
            corrected['hip'] = min_hip
            print(f"Hip too small, adjusting: {hip:.2f} -> {min_hip:.2f}")
        elif hip > max_hip:
            corrected['hip'] = max_hip
            print(f"Hip too large, adjusting: {hip:.2f} -> {max_hip:.2f}")
        else:
            corrected['hip'] = hip
            
        print(f"Corrected measurements - Chest: {corrected['chest']:.2f}, Waist: {corrected['waist']:.2f}, Hip: {corrected['hip']:.2f}")
        
        return corrected

    def _measure_circumference_at_level(self, vertices, y_level, height_range, body_part, tolerance_ratio=0.03):
        """
        在指定高度测量周长（优化版）
        
        Args:
            vertices: 顶点数组
            y_level: 测量高度
            height_range: 总身高范围
            body_part: 身体部位名称（用于调试）
            tolerance_ratio: 容差比例
        """
        # 根据身体部位调整容差
        part_tolerances = {
            'waist': 0.02,    # 腰部容差小（寻找最细处）
            'chest': 0.03,    # 胸部中等容差
            'hip': 0.04       # 臀部容差大（寻找最宽处）
        }
        
        tolerance = height_range * part_tolerances.get(body_part, tolerance_ratio)
        
        # 提取该高度附近的点
        level_mask = np.abs(vertices[:, 1] - y_level) < tolerance
        level_points = vertices[level_mask]
        
        print(f"  {body_part} level points: {len(level_points)} (tolerance: {tolerance:.4f})")
        
        if len(level_points) < 8:
            # 如果点太少，尝试扩大搜索范围
            tolerance *= 2
            level_mask = np.abs(vertices[:, 1] - y_level) < tolerance
            level_points = vertices[level_mask]
            print(f"  Extended search: {len(level_points)} points")
            
        if len(level_points) < 5:
            print(f"  Warning: Insufficient points for {body_part} at level {y_level:.3f}")
            return 0.0
        
        # 投影到XZ平面
        xz_points = level_points[:, [0, 2]]
        
        # 根据身体部位应用不同的过滤策略
        if body_part == 'waist':
            # 腰部：寻找最小轮廓
            xz_points = self._filter_for_minimum_contour(xz_points)
        elif body_part == 'hip':
            # 臀部：寻找最大轮廓
            xz_points = self._filter_for_maximum_contour(xz_points)
        else:
            # 胸部：标准过滤
            xz_points = self._remove_extreme_outliers(xz_points)
        
        if len(xz_points) < 4:
            return 0.0
        
        # 计算周长
        circumference = self._compute_robust_circumference(xz_points)
        
        return circumference

    def _filter_for_minimum_contour(self, points_2d):
        """
        过滤以获得最小轮廓（用于腰围）
        """
        if len(points_2d) < 8:
            return points_2d
            
        # 计算到质心的距离
        center = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - center, axis=1)
        
        # 保留距离较小的点（内部轮廓）
        threshold = np.percentile(distances, 70)  # 保留70%的内部点
        inner_mask = distances <= threshold
        
        return points_2d[inner_mask]

    def _filter_for_maximum_contour(self, points_2d):
        """
        过滤以获得最大轮廓（用于臀围）
        """
        if len(points_2d) < 8:
            return points_2d
            
        # 计算到质心的距离
        center = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - center, axis=1)
        
        # 保留距离较大的点（外部轮廓）
        threshold = np.percentile(distances, 30)  # 保留30%的外部点
        outer_mask = distances >= threshold
        
        return points_2d[outer_mask]

    def _extract_core_body_points(self, vertices, density_percentile=60):
        """
        提取核心身体点，移除衣物和离群点
        
        Args:
            vertices: 输入顶点 [N, 3]
            density_percentile: 保留密度最高的百分比
        """
        if len(vertices) < 1000:
            return vertices
            
        # 1. 基于高度分层进行密度分析
        y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        height_layers = 20  # 分为20层
        
        core_indices = []
        
        for i in range(height_layers):
            # 当前层的Y范围
            y_start = y_min + i * (y_max - y_min) / height_layers
            y_end = y_min + (i + 1) * (y_max - y_min) / height_layers
            
            # 提取当前层的点
            layer_mask = (vertices[:, 1] >= y_start) & (vertices[:, 1] < y_end)
            layer_indices = np.where(layer_mask)[0]
            
            if len(layer_indices) < 10:
                continue
                
            layer_vertices = vertices[layer_indices]
            
            # 计算到该层中心的距离
            layer_center = np.mean(layer_vertices, axis=0)
            distances = np.linalg.norm(layer_vertices - layer_center, axis=1)
            
            # 保留距离中心最近的点（去除衣物外围）
            keep_count = max(5, int(len(layer_indices) * density_percentile / 100))
            closest_indices = np.argsort(distances)[:keep_count]
            
            core_indices.extend(layer_indices[closest_indices])
        
        return vertices[core_indices] if core_indices else vertices[:len(vertices)//3]

    def _remove_extreme_outliers(self, points_2d, percentile_range=(10, 90)):
        """
        移除极端离群点
        """
        if len(points_2d) < 8:
            return points_2d
            
        # 计算到质心的距离
        center = np.mean(points_2d, axis=0)
        distances = np.linalg.norm(points_2d - center, axis=1)
        
        # 使用百分位数过滤
        lower_bound = np.percentile(distances, percentile_range[0])
        upper_bound = np.percentile(distances, percentile_range[1])
        
        valid_mask = (distances >= lower_bound) & (distances <= upper_bound)
        
        # 至少保留30%的点
        if np.sum(valid_mask) < len(points_2d) * 0.3:
            sorted_indices = np.argsort(distances)
            keep_count = max(5, len(points_2d) // 3)
            valid_mask = np.zeros(len(points_2d), dtype=bool)
            valid_mask[sorted_indices[:keep_count]] = True
            
        return points_2d[valid_mask]

    def _compute_robust_circumference(self, points_2d):
        """
        计算鲁棒的周长估计
        
        策略：结合凸包周长和椭圆拟合周长，取较小值（更保守）
        """
        if len(points_2d) < 4:
            return 0.0
            
        try:
            # 方法1：凸包周长
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
            
            # 计算凸包周长
            hull_perimeter = 0
            for i in range(len(hull_points)):
                p1 = hull_points[i]
                p2 = hull_points[(i + 1) % len(hull_points)]
                hull_perimeter += np.linalg.norm(p2 - p1)
            
            # 方法2：椭圆拟合周长
            center = np.mean(points_2d, axis=0)
            # 计算主轴方向
            centered_points = points_2d - center
            
            # 简化的椭圆拟合：计算各方向的最大距离
            angles = np.linspace(0, 2*np.pi, 36)  # 36个方向
            max_distances = []
            
            for angle in angles:
                direction = np.array([np.cos(angle), np.sin(angle)])
                projections = np.dot(centered_points, direction)
                max_distances.append(np.max(np.abs(projections)))
            
            # 使用平均半径估算周长
            avg_radius = np.mean(max_distances)
            ellipse_circumference = 2 * np.pi * avg_radius
            
            # 取较保守的估计（通常是较小值）
            conservative_circumference = min(hull_perimeter, ellipse_circumference)
            
            # 额外的合理性检查
            max_span = np.max(np.max(points_2d, axis=0) - np.min(points_2d, axis=0))
            if conservative_circumference > max_span * 4:  # 周长不应超过最大跨度的4倍
                conservative_circumference = max_span * np.pi  # 使用圆形近似
            
            return conservative_circumference
            
        except Exception as e:
            print(f"Circumference calculation failed: {e}")
            # 最后的fallback：简单边界盒周长
            bbox_size = np.max(points_2d, axis=0) - np.min(points_2d, axis=0)
            return 2 * (bbox_size[0] + bbox_size[1])  # 矩形周长近似

    def detect_and_fix_coordinate_system(self, vertices):
        """
        智能检测并修正坐标系方向，确保Y轴是身高方向
        
        Args:
            vertices: 原始顶点数组 [N, 3]
            
        Returns:
            corrected_vertices: 修正后的顶点数组 [N, 3]
            transformation_info: 变换信息字典
        """
        # 计算各轴的范围
        ranges = [
            np.max(vertices[:, 0]) - np.min(vertices[:, 0]),  # X
            np.max(vertices[:, 1]) - np.min(vertices[:, 1]),  # Y  
            np.max(vertices[:, 2]) - np.min(vertices[:, 2])   # Z
        ]
        
        axis_names = ['X', 'Y', 'Z']
        max_axis = np.argmax(ranges)
        
        transformation_info = {
            'original_ranges': {
                'X': ranges[0],
                'Y': ranges[1], 
                'Z': ranges[2]
            },
            'height_axis_detected': axis_names[max_axis],
            'transformation_applied': None
        }
        
        print(f"Coordinate system analysis:")
        print(f"  X range: {ranges[0]:.3f}, Y range: {ranges[1]:.3f}, Z range: {ranges[2]:.3f}")
        print(f"  Detected height axis: {axis_names[max_axis]}")
        
        if max_axis == 1:  # Y轴已经是最大，无需变换
            print("  ✓ Y-axis is already the height direction")
            transformation_info['transformation_applied'] = 'none'
            return vertices, transformation_info
        
        elif max_axis == 0:  # X轴是最大，需要 X->Y 变换
            print("  🔄 Swapping X and Y axes (X->Y)")
            corrected_vertices = vertices[:, [1, 0, 2]]  # [Y, X, Z]
            transformation_info['transformation_applied'] = 'swap_xy'
            
        elif max_axis == 2:  # Z轴是最大，需要 Z->Y 变换  
            print("  🔄 Swapping Y and Z axes (Z->Y)")
            corrected_vertices = vertices[:, [0, 2, 1]]  # [X, Z, Y]
            transformation_info['transformation_applied'] = 'swap_yz'
        
        # 验证变换结果
        new_height = np.max(corrected_vertices[:, 1]) - np.min(corrected_vertices[:, 1])
        print(f"  ✓ Corrected height (Y-axis): {new_height:.3f}")
        
        return corrected_vertices, transformation_info

    def measure_from_ply(self, ply_path: str, target_height_cm: float = 170.0, enable_semantic=True):
        """
        从PLY文件测量身体三围（集成语义分析）
        
        Args:
            ply_path: PLY文件路径
            target_height_cm: 目标身高（cm）
            enable_semantic: 是否启用语义分析
        
        Returns:
            Dict: 包含测量结果的字典
        """
        try:
            # 加载点云
            mesh = trimesh.load(ply_path)
            if hasattr(mesh, 'vertices'):
                vertices = np.array(mesh.vertices)
            else:
                vertices = np.array(mesh)
            
            print(f"Loaded {len(vertices)} vertices from {ply_path}")
            
            # 首先修正坐标系方向
            vertices, coord_transform_info = self.detect_and_fix_coordinate_system(vertices)
            
            # 数据清理
            vertices = self.preprocess_pointcloud(vertices)
            print(f"After preprocessing: {len(vertices)} vertices")
            
            # 重新分析点云特征（现在Y轴应该是身高）
            model_height_units = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
            x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
            z_range = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
            
            print(f"Corrected dimensions - Height(Y): {model_height_units:.4f}, Width(X): {x_range:.4f}, Depth(Z): {z_range:.4f}")
            
            # 验证修正效果：身高应该是最大尺寸
            if model_height_units < max(x_range, z_range):
                print("Warning: Height still not the largest dimension after correction")
                return self._handle_abnormal_pointcloud(vertices, target_height_cm)
            
            # 语义分析（如果启用）
            clothing_corrections = None
            if enable_semantic:
                try:
                    from threestudio.utils.semantic_utils import SemanticAnalyzer, ClothingRemover
                    analyzer = SemanticAnalyzer()
                    semantic_result = analyzer.analyze_point_cloud(ply_path)
                    
                    if 'error' not in semantic_result:
                        remover = ClothingRemover(analyzer)
                        clothing_corrections = semantic_result.get('clothing_analysis', {})
                        print(f"Semantic analysis completed: {semantic_result.get('semantic_distribution', {})}")
                    else:
                        print(f"Semantic analysis failed: {semantic_result['error']}")
                        
                except Exception as e:
                    print(f"Semantic analysis not available: {e}")
            
            # 执行基础测量
            measurements = self.fallback_measurement(vertices)
            
            # --- 智能尺度换算逻辑 ---
            if model_height_units > 0:
                # 计算从模型单位到厘米的换算因子
                scale_factor = target_height_cm / model_height_units
                
                print(f"Scale factor: {scale_factor:.2f} (model_height: {model_height_units:.4f} -> target: {target_height_cm}cm)")
                
                # 验证尺度因子合理性（正常人体模型的尺度因子应该在1-100之间）
                if scale_factor > 200:  # 放宽阈值，因为修正坐标后可能仍需要较大缩放
                    print(f"Warning: Large scale factor {scale_factor:.2f}, using proportional method")
                    return self._proportional_measurement(vertices, target_height_cm, measurements)
                
                # 统一换算所有测量值到厘米
                for key in ['height', 'chest', 'waist', 'hip', 'bust']:
                    if key in measurements:
                        measurements[key] *= scale_factor
            else:
                scale_factor = 1.0
            
            # 添加元数据
            measurements['metadata'] = {
                'source_ply': ply_path,
                'target_height_cm': target_height_cm,
                'model_height_units': float(model_height_units),
                'scale_factor': float(scale_factor),
                'coordinate_transform': coord_transform_info,
                'semantic_enabled': enable_semantic,
                'clothing_corrections': clothing_corrections,
                'total_vertices': len(vertices),
                'timestamp': datetime.now().isoformat()
            }
            
            return measurements
            
        except Exception as e:
            print(f"Error measuring from PLY: {e}")
            return {
                'chest': 0.0,
                'waist': 0.0, 
                'hip': 0.0,
                'error': str(e)
            }

    def _handle_abnormal_pointcloud(self, vertices, target_height_cm):
        """
        处理异常点云数据的特殊方法
        """
        print("Using abnormal pointcloud handler...")
        
        # 尝试找到实际的人体部分（通过密度分析）
        # 计算点云密度
        from scipy.spatial.distance import cdist
        
        # 采样部分点进行密度计算（避免计算过重）
        sample_size = min(10000, len(vertices))
        sample_indices = np.random.choice(len(vertices), sample_size, replace=False)
        sample_vertices = vertices[sample_indices]
        
        # 计算每个点到最近10个点的平均距离
        distances = cdist(sample_vertices, sample_vertices)
        np.fill_diagonal(distances, np.inf)
        
        # 取最近10个点的平均距离作为密度指标
        k = min(10, len(sample_vertices) - 1)
        nearest_distances = np.partition(distances, k, axis=1)[:, :k]
        density_scores = np.mean(nearest_distances, axis=1)
        
        # 选择密度最高的50%点作为人体核心
        threshold = np.percentile(density_scores, 50)
        core_mask = density_scores <= threshold
        core_indices = sample_indices[core_mask]
        core_vertices = vertices[core_indices]
        
        print(f"Identified {len(core_vertices)} core body vertices")
        
        # 对核心顶点重新测量
        measurements = self.fallback_measurement(core_vertices)
        
        # 使用比例方法而非绝对尺度
        return self._proportional_measurement(core_vertices, target_height_cm, measurements)

    def _proportional_measurement(self, vertices, target_height_cm, raw_measurements):
        """
        基于比例的测量方法，避免异常尺度因子
        """
        print("Using proportional measurement method...")
        
        # 获取原始测量的比例关系
        waist_raw = raw_measurements.get('waist', 0)
        chest_raw = raw_measurements.get('chest', 0) or raw_measurements.get('bust', 0)
        hip_raw = raw_measurements.get('hip', 0)
        
        if waist_raw == 0:
            print("Warning: No valid waist measurement found")
            return {
                'height': target_height_cm,
                'chest': 0.0,
                'waist': 0.0,
                'hip': 0.0,
                'bust': 0.0,
                'error': 'No valid measurements'
            }
        
        # 使用标准人体比例进行估算
        # 基于腰围作为参考点，使用典型人体比例
        if target_height_cm == 170:  # 标准身高参考
            estimated_waist = 70.0  # cm
        else:
            # 身高与腰围的经验关系：腰围约为身高的0.41倍
            estimated_waist = target_height_cm * 0.41
        
        # 计算腰围的校正比例
        if waist_raw > 0:
            waist_correction = estimated_waist / waist_raw
        else:
            waist_correction = 1.0
        
        # 应用比例校正
        corrected_chest = chest_raw * waist_correction if chest_raw > 0 else estimated_waist * 1.2
        corrected_waist = estimated_waist
        corrected_hip = hip_raw * waist_correction if hip_raw > 0 else estimated_waist * 1.3
        
        # 确保测量值合理（基本人体比例检查）
        corrected_chest = max(50, min(150, corrected_chest))  # 胸围50-150cm
        corrected_waist = max(50, min(120, corrected_waist))  # 腰围50-120cm  
        corrected_hip = max(60, min(160, corrected_hip))      # 臀围60-160cm
        
        measurements = {
            'height': target_height_cm,
            'chest': corrected_chest,
            'waist': corrected_waist,
            'hip': corrected_hip,
            'bust': corrected_chest,  # 别名
            'metadata': {
                'method': 'proportional_correction',
                'waist_correction_factor': waist_correction,
                'estimated_waist_cm': estimated_waist,
                'raw_waist': waist_raw,
                'target_height_cm': target_height_cm,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print(f"Proportional measurements - Chest: {corrected_chest:.1f}cm, Waist: {corrected_waist:.1f}cm, Hip: {corrected_hip:.1f}cm")
        
        return measurements

    def measure_with_clothing_removal(self, ply_path: str, target_height_cm: float = 170.0):
        """
        使用衣物移除的高精度测量
        
        Returns:
            Dict: 包含穿衣和脱衣测量对比的结果
        """
        try:
            from threestudio.utils.semantic_utils import SemanticAnalyzer, ClothingRemover
            
            # 1. 标准穿衣测量
            clothed_measurements = self.measure_from_ply(ply_path, target_height_cm, enable_semantic=False)
            
            # 2. 语义分析
            analyzer = SemanticAnalyzer()
            semantic_result = analyzer.analyze_point_cloud(ply_path)
            
            if 'error' in semantic_result:
                return {
                    'clothed_measurements': clothed_measurements,
                    'nude_measurements': clothed_measurements,  # 退化到相同值
                    'semantic_analysis': semantic_result,
                    'clothing_impact': {'chest': 0, 'waist': 0, 'hip': 0}
                }
            
            # 3. 衣物厚度估算和修正
            remover = ClothingRemover(analyzer)
            nude_measurements, corrections = remover.estimate_nude_measurements(
                clothed_measurements, 
                semantic_result.get('clothing_analysis', {})
            )
            
            # 4. 计算衣物影响
            clothing_impact = {
                'chest_diff': clothed_measurements['chest'] - nude_measurements['chest'],
                'waist_diff': clothed_measurements['waist'] - nude_measurements['waist'],
                'hip_diff': clothed_measurements['hip'] - nude_measurements['hip'],
                'chest_percent': ((clothed_measurements['chest'] - nude_measurements['chest']) / nude_measurements['chest'] * 100) if nude_measurements['chest'] > 0 else 0,
                'waist_percent': ((clothed_measurements['waist'] - nude_measurements['waist']) / nude_measurements['waist'] * 100) if nude_measurements['waist'] > 0 else 0,
                'hip_percent': ((clothed_measurements['hip'] - nude_measurements['hip']) / nude_measurements['hip'] * 100) if nude_measurements['hip'] > 0 else 0
            }
            
            return {
                'clothed_measurements': clothed_measurements,
                'nude_measurements': nude_measurements,
                'clothing_corrections': corrections,
                'clothing_impact': clothing_impact,
                'semantic_analysis': semantic_result,
                'metadata': {
                    'method': 'semantic_clothing_removal',
                    'target_height_cm': target_height_cm,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Error in clothing removal measurement: {e}")
            return {
                'error': str(e),
                'clothed_measurements': self.measure_from_ply(ply_path, target_height_cm, enable_semantic=False)
            }

    def save_measurements(self, measurements, filepath):
        """
        保存测量结果到JSON文件
        """
        with open(filepath, 'w') as f:
            json.dump(measurements, f, indent=2, ensure_ascii=False)
        print(f"Measurements saved to {filepath}")

    def load_measurements(self, filepath):
        """
        从JSON文件加载测量结果
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def compare_measurements(measurements1, measurements2):
        """
        比较两组测量结果
        """
        comparison = {}
        common_keys = set(measurements1.keys()) & set(measurements2.keys())
        
        for key in common_keys:
            if isinstance(measurements1[key], (int, float)) and isinstance(measurements2[key], (int, float)):
                diff = measurements2[key] - measurements1[key]
                diff_percent = (diff / measurements1[key]) * 100 if measurements1[key] != 0 else 0
                comparison[key] = {
                    'value1': measurements1[key],
                    'value2': measurements2[key],
                    'absolute_diff': diff,
                    'percent_diff': diff_percent
                }
        
        return comparison