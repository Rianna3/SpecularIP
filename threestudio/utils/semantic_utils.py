import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import trimesh
import json
import os
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import cv2

class SemanticLabelInitializer:
    """
    语义标签初始化工具
    """
    
    def __init__(self, semantic_classes=['body', 'clothing', 'hair', 'accessories']):
        self.semantic_classes = semantic_classes
        self.n_classes = len(semantic_classes)

    def init_from_smpl(self, vertices, smpl_model=None):
        """
        基于SMPL模型初始化语义标签
        """
        n_points = len(vertices)
        semantic_logits = torch.zeros((n_points, self.n_classes))
        
        if smpl_model is not None:
            # 基于SMPL顶点的body part分割（简化实现）
            semantic_logits = self.smpl_based_init(vertices, smpl_model)
        else:
            # 简单的几何启发式初始化
            semantic_logits = self.geometric_heuristic_init(vertices)
        
        return semantic_logits

    def smpl_based_init(self, vertices, smpl_model):
        """基于SMPL模型的初始化"""
        n_points = len(vertices)
        semantic_logits = torch.zeros((n_points, self.n_classes))
        
        # 简化的SMPL-based初始化，默认大部分为body
        semantic_logits[:, 0] = 2.0  # body
        
        return semantic_logits

    def geometric_heuristic_init(self, vertices):
        """
        基于几何特征的启发式初始化
        """
        n_points = len(vertices)
        semantic_logits = torch.zeros((n_points, self.n_classes))
        
        # 计算点云的边界框
        min_coords = torch.min(vertices, dim=0)[0]
        max_coords = torch.max(vertices, dim=0)[0]
        
        # 身高和宽度
        height = max_coords[1] - min_coords[1]
        width_x = max_coords[0] - min_coords[0]
        
        for i, vertex in enumerate(vertices):
            # 相对位置
            rel_y = (vertex[1] - min_coords[1]) / height
            rel_x = abs(vertex[0] - (min_coords[0] + max_coords[0]) / 2) / (width_x / 2)
            
            # 启发式规则
            if rel_y > 0.85:  # 头部区域
                if rel_x < 0.6:  # 靠近中心，可能是头部或头发
                    semantic_logits[i, 0] = 2.0  # body (head)
                    semantic_logits[i, 2] = 1.0  # hair (较低概率)
                else:  # 边缘，更可能是头发
                    semantic_logits[i, 2] = 2.5  # hair
            elif 0.3 < rel_y < 0.85:  # 躯干区域
                if rel_x < 0.8:  # 核心身体区域
                    semantic_logits[i, 0] = 3.0  # body
                else:  # 边缘，可能是衣物
                    semantic_logits[i, 1] = 1.5  # clothing
                    semantic_logits[i, 0] = 1.0  # body
            else:  # 下半身
                semantic_logits[i, 0] = 2.5  # body
                semantic_logits[i, 1] = 0.5  # clothing (较低概率)
        
        return semantic_logits

    def refine_with_clustering(self, vertices, semantic_logits, n_iterations=3):
        """
        使用聚类方法精炼语义标签
        """
        from sklearn.cluster import KMeans
        
        # 结合空间位置和当前语义概率进行聚类
        features = torch.cat([
            vertices,
            F.softmax(semantic_logits, dim=-1)
        ], dim=-1)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_classes * 2, random_state=42)
        cluster_labels = kmeans.fit_predict(features.numpy())
        
        # 根据聚类结果调整语义标签
        refined_logits = semantic_logits.clone()
        
        for cluster_id in range(self.n_classes * 2):
            cluster_mask = cluster_labels == cluster_id
            if cluster_mask.sum() > 0:
                # 计算该聚类的主要语义类别
                cluster_semantics = semantic_logits[cluster_mask]
                dominant_class = torch.argmax(torch.mean(cluster_semantics, dim=0))
                
                # 增强该类别的置信度
                refined_logits[cluster_mask, dominant_class] += 0.5
        
        return refined_logits

class SemanticConsistencyLoss:
    """
    语义一致性损失函数
    """
    
    def __init__(self, lambda_spatial=1.0, lambda_temporal=0.5):
        self.lambda_spatial = lambda_spatial
        self.lambda_temporal = lambda_temporal

    def spatial_consistency_loss(self, vertices, semantic_logits, k=5):
        """
        空间一致性损失：相邻点应有相似的语义标签
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 找到每个点的k个最近邻
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(vertices.detach().cpu().numpy())
        distances, indices = nbrs.kneighbors(vertices.detach().cpu().numpy())
        
        # 排除自身（第一个邻居）
        neighbor_indices = indices[:, 1:]
        
        loss = 0.0
        for i in range(len(vertices)):
            # 当前点的语义概率
            current_probs = F.softmax(semantic_logits[i], dim=-1)
            
            # 邻居点的语义概率
            neighbor_probs = F.softmax(semantic_logits[neighbor_indices[i]], dim=-1)
            neighbor_mean_probs = torch.mean(neighbor_probs, dim=0)
            
            # KL散度损失
            kl_loss = F.kl_div(
                torch.log(current_probs + 1e-8),
                neighbor_mean_probs,
                reduction='sum'
            )
            loss += kl_loss
        
        return loss / len(vertices) * self.lambda_spatial

    def temporal_consistency_loss(self, semantic_logits_t1, semantic_logits_t2):
        """
        时间一致性损失：相邻时间步的语义标签应保持一致
        """
        if semantic_logits_t1 is None or semantic_logits_t2 is None:
            return torch.tensor(0.0, device=semantic_logits_t2.device)
        
        # 确保尺寸匹配（可能由于densification导致点数变化）
        min_points = min(len(semantic_logits_t1), len(semantic_logits_t2))
        
        probs_t1 = F.softmax(semantic_logits_t1[:min_points], dim=-1)
        probs_t2 = F.softmax(semantic_logits_t2[:min_points], dim=-1)
        
        # L2损失
        temporal_loss = F.mse_loss(probs_t1, probs_t2)
        
        return temporal_loss * self.lambda_temporal

class SemanticGuidedDensification:
    """
    语义指导的高斯点密化
    """
    
    def __init__(self, densify_clothing_more=True):
        self.densify_clothing_more = densify_clothing_more

    def semantic_aware_densification_mask(self, gaussian_model, gradients, grad_threshold):
        """
        基于语义信息调整密化阈值
        """
        semantic_probs = gaussian_model.get_semantic_labels
        
        # 基础的梯度阈值筛选
        base_mask = torch.norm(gradients, dim=-1) >= grad_threshold
        
        if self.densify_clothing_more:
            # 对衣物区域使用更低的阈值，促进更精细的几何
            clothing_mask = semantic_probs[:, 1] > 0.3  # clothing probability > 0.3
            clothing_grad_mask = torch.norm(gradients, dim=-1) >= (grad_threshold * 0.7)
            
            # 合并mask
            final_mask = base_mask | (clothing_mask & clothing_grad_mask)
        else:
            final_mask = base_mask
        
        return final_mask

    def semantic_split_strategy(self, gaussian_model, selected_mask):
        """
        基于语义的点分裂策略
        """
        semantic_probs = gaussian_model.get_semantic_labels[selected_mask]
        
        # 对不同语义类别使用不同的分裂参数
        n_splits = torch.ones(len(semantic_probs), dtype=torch.int)
        
        # 衣物区域分裂更多（更精细的褶皱）
        clothing_mask = semantic_probs[:, 1] > 0.5
        n_splits[clothing_mask] = 3
        
        # 身体区域标准分裂
        body_mask = semantic_probs[:, 0] > 0.5
        n_splits[body_mask] = 2
        
        return n_splits

class SemanticAnalyzer:
    """语义分析器 - 分析点云中的不同语义部分"""
    
    def __init__(self):
        self.semantic_labels = {
            0: "unknown",
            1: "body", 
            2: "clothing",
            3: "hair",
            4: "accessories"
        }
        
        # 预定义的人体关键点 (基于SMPL-X标准)
        self.body_landmarks = {
            'head': {'height_range': (0.85, 1.0), 'radius': 0.12},
            'neck': {'height_range': (0.82, 0.88), 'radius': 0.08},
            'chest': {'height_range': (0.65, 0.82), 'radius': 0.18},
            'waist': {'height_range': (0.55, 0.65), 'radius': 0.15},
            'hip': {'height_range': (0.45, 0.55), 'radius': 0.20},
            'thigh': {'height_range': (0.25, 0.45), 'radius': 0.12},
            'leg': {'height_range': (0.0, 0.25), 'radius': 0.08}
        }
        
        # 衣物厚度估计 (基于常见服装)
        self.clothing_thickness = {
            'blazer': 0.015,      # 1.5cm
            'shirt': 0.008,       # 0.8cm  
            'trousers': 0.012,    # 1.2cm
            'shoes': 0.025        # 2.5cm
        }
    
    def analyze_point_cloud(self, ply_path: str) -> Dict:
        """分析PLY点云文件，识别语义区域"""
        try:
            # 加载点云
            mesh = trimesh.load(ply_path)
            if hasattr(mesh, 'vertices'):
                points = np.array(mesh.vertices)
            else:
                # 如果是点云对象，直接使用
                points = np.array(mesh)
            
            print(f"Loaded point cloud with {len(points)} vertices")
            
            # 数据预处理
            cleaned_points = self._clean_point_cloud(points)
            
            # 几何分析
            geometry_analysis = self._analyze_geometry(cleaned_points)
            
            # 语义分割
            semantic_labels = self._semantic_segmentation(cleaned_points, geometry_analysis)
            
            # 衣物检测
            clothing_analysis = self._analyze_clothing(cleaned_points, semantic_labels)
            
            # 身体结构分析
            body_analysis = self._analyze_body_structure(cleaned_points, semantic_labels)
            
            results = {
                'total_points': len(points),
                'cleaned_points': len(cleaned_points),
                'geometry_analysis': geometry_analysis,
                'semantic_distribution': self._get_semantic_distribution(semantic_labels),
                'clothing_analysis': clothing_analysis,
                'body_analysis': body_analysis,
                'confidence_score': self._calculate_confidence(semantic_labels, cleaned_points)
            }
            
            return results
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return {'error': str(e)}
    
    def _clean_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """清理点云数据"""
        # 移除离群点
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        threshold = np.percentile(distances, 95)
        
        cleaned = points[distances <= threshold]
        
        # 移除地面点 (假设Y轴为高度)
        if len(cleaned) > 0:
            min_y = np.min(cleaned[:, 1])
            ground_threshold = min_y + 0.05  # 5cm above ground
            cleaned = cleaned[cleaned[:, 1] > ground_threshold]
        
        return cleaned
    
    def _analyze_geometry(self, points: np.ndarray) -> Dict:
        """分析点云几何特征"""
        if len(points) == 0:
            return {}
            
        # 基本统计
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = np.mean(points, axis=0)
        
        # 计算主要轴向
        height = max_coords[1] - min_coords[1]
        width = max_coords[0] - min_coords[0]
        depth = max_coords[2] - min_coords[2]
        
        # 密度分析
        volume = width * height * depth
        density = len(points) / volume if volume > 0 else 0
        
        return {
            'bounds': {
                'min': min_coords.tolist(),
                'max': max_coords.tolist(),
                'center': center.tolist()
            },
            'dimensions': {
                'height': float(height),
                'width': float(width),
                'depth': float(depth)
            },
            'density': float(density),
            'aspect_ratios': {
                'height_width': float(height/width) if width > 0 else 0,
                'height_depth': float(height/depth) if depth > 0 else 0
            }
        }
    
    def _semantic_segmentation(self, points: np.ndarray, geometry: Dict) -> np.ndarray:
        """基于几何特征的语义分割"""
        if len(points) == 0:
            return np.array([])
        
        labels = np.zeros(len(points), dtype=int)
        
        if 'bounds' not in geometry:
            return labels
        
        min_y = geometry['bounds']['min'][1]
        max_y = geometry['bounds']['max'][1]
        height_range = max_y - min_y
        
        if height_range <= 0:
            return labels
        
        # 基于高度的初步分割
        for i, point in enumerate(points):
            height_ratio = (point[1] - min_y) / height_range
            
            # 按高度分配基础标签
            if height_ratio > 0.85:  # 头部区域
                labels[i] = 3  # hair (可能包含头发)
            elif height_ratio > 0.6:  # 上半身
                labels[i] = 2  # clothing (上装区域)
            elif height_ratio > 0.45:  # 腰臀部
                labels[i] = 1  # body (相对贴身区域)
            else:  # 下半身
                labels[i] = 2  # clothing (下装区域)
        
        # 基于密度和距离的细化
        labels = self._refine_segmentation(points, labels)
        
        return labels
    
    def _refine_segmentation(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """细化语义分割结果"""
        # 使用DBSCAN进行聚类优化
        try:
            clustering = DBSCAN(eps=0.05, min_samples=10).fit(points)
            clusters = clustering.labels_
            
            # 为每个簇分配最常见的语义标签
            refined_labels = labels.copy()
            for cluster_id in np.unique(clusters):
                if cluster_id == -1:  # 噪声点
                    continue
                    
                cluster_mask = clusters == cluster_id
                cluster_labels = labels[cluster_mask]
                
                if len(cluster_labels) > 0:
                    # 使用众数作为簇标签
                    most_common = np.bincount(cluster_labels).argmax()
                    refined_labels[cluster_mask] = most_common
            
            return refined_labels
            
        except Exception as e:
            print(f"Error in segmentation refinement: {e}")
            return labels
    
    def _analyze_clothing(self, points: np.ndarray, labels: np.ndarray) -> Dict:
        """分析衣物特征"""
        clothing_points = points[labels == 2]  # clothing label
        
        if len(clothing_points) == 0:
            return {'detected_clothing': [], 'clothing_volume': 0}
        
        analysis = {
            'clothing_points_count': len(clothing_points),
            'clothing_percentage': len(clothing_points) / len(points) * 100,
            'detected_clothing': [],
            'clothing_volume': 0
        }
        
        # 检测具体衣物类型 (基于位置和形状)
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        height_range = max_y - min_y
        
        # 检测上装 (blazer + shirt)
        upper_clothing = clothing_points[
            clothing_points[:, 1] > min_y + height_range * 0.5
        ]
        if len(upper_clothing) > 0:
            analysis['detected_clothing'].append({
                'type': 'upper_clothing',
                'items': ['blazer', 'shirt'],
                'point_count': len(upper_clothing),
                'estimated_thickness': self.clothing_thickness['blazer'] + self.clothing_thickness['shirt']
            })
        
        # 检测下装 (trousers)
        lower_clothing = clothing_points[
            clothing_points[:, 1] <= min_y + height_range * 0.5
        ]
        if len(lower_clothing) > 0:
            analysis['detected_clothing'].append({
                'type': 'lower_clothing', 
                'items': ['trousers'],
                'point_count': len(lower_clothing),
                'estimated_thickness': self.clothing_thickness['trousers']
            })
        
        # 估算衣物体积
        if len(clothing_points) > 0:
            try:
                hull = ConvexHull(clothing_points)
                analysis['clothing_volume'] = hull.volume
            except:
                analysis['clothing_volume'] = 0
        
        return analysis
    
    def _analyze_body_structure(self, points: np.ndarray, labels: np.ndarray) -> Dict:
        """分析身体结构"""
        body_points = points[labels == 1]  # body label
        
        if len(body_points) == 0:
            return {'body_landmarks': {}, 'body_proportions': {}}
        
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        height_range = max_y - min_y
        
        landmarks = {}
        
        # 识别身体关键部位
        for landmark_name, landmark_info in self.body_landmarks.items():
            height_min = min_y + height_range * landmark_info['height_range'][0]
            height_max = min_y + height_range * landmark_info['height_range'][1]
            
            landmark_points = body_points[
                (body_points[:, 1] >= height_min) & 
                (body_points[:, 1] <= height_max)
            ]
            
            if len(landmark_points) > 0:
                landmarks[landmark_name] = {
                    'point_count': len(landmark_points),
                    'center': np.mean(landmark_points, axis=0).tolist(),
                    'height_range': [float(height_min), float(height_max)]
                }
        
        # 计算身体比例
        proportions = {}
        if 'chest' in landmarks and 'waist' in landmarks and 'hip' in landmarks:
            chest_width = self._estimate_width(body_points, landmarks['chest']['center'][1])
            waist_width = self._estimate_width(body_points, landmarks['waist']['center'][1])
            hip_width = self._estimate_width(body_points, landmarks['hip']['center'][1])
            
            proportions = {
                'chest_waist_ratio': float(chest_width / waist_width) if waist_width > 0 else 0,
                'hip_waist_ratio': float(hip_width / waist_width) if waist_width > 0 else 0,
                'chest_hip_ratio': float(chest_width / hip_width) if hip_width > 0 else 0
            }
        
        return {
            'body_landmarks': landmarks,
            'body_proportions': proportions,
            'body_points_count': len(body_points),
            'body_percentage': len(body_points) / len(points) * 100
        }
    
    def _estimate_width(self, points: np.ndarray, height: float, tolerance: float = 0.02) -> float:
        """估算指定高度处的身体宽度"""
        level_points = points[
            (points[:, 1] >= height - tolerance) & 
            (points[:, 1] <= height + tolerance)
        ]
        
        if len(level_points) == 0:
            return 0.0
        
        # 计算X轴方向的宽度
        x_range = np.max(level_points[:, 0]) - np.min(level_points[:, 0])
        z_range = np.max(level_points[:, 2]) - np.min(level_points[:, 2])
        
        # 返回较大的宽度值
        return max(x_range, z_range)
    
    def _get_semantic_distribution(self, labels: np.ndarray) -> Dict:
        """获取语义标签分布"""
        if len(labels) == 0:
            return {}
        
        distribution = {}
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            label_name = self.semantic_labels.get(label, f"unknown_{label}")
            distribution[label_name] = {
                'count': int(count),
                'percentage': float(count / len(labels) * 100)
            }
        
        return distribution
    
    def _calculate_confidence(self, labels: np.ndarray, points: np.ndarray) -> float:
        """计算分割置信度"""
        if len(labels) == 0:
            return 0.0
        
        # 基于标签分布的置信度
        unique_labels = np.unique(labels)
        label_confidence = len(unique_labels) / len(self.semantic_labels)
        
        # 基于空间连续性的置信度
        spatial_confidence = self._calculate_spatial_coherence(points, labels)
        
        # 综合置信度
        overall_confidence = (label_confidence + spatial_confidence) / 2
        
        return min(1.0, max(0.0, overall_confidence))
    
    def _calculate_spatial_coherence(self, points: np.ndarray, labels: np.ndarray) -> float:
        """计算空间连续性"""
        if len(points) < 2:
            return 0.0
        
        try:
            # 计算相邻点的标签一致性
            from sklearn.neighbors import NearestNeighbors
            
            nbrs = NearestNeighbors(n_neighbors=min(6, len(points))).fit(points)
            distances, indices = nbrs.kneighbors(points)
            
            consistent_neighbors = 0
            total_neighbors = 0
            
            for i, neighbors in enumerate(indices):
                for neighbor_idx in neighbors[1:]:  # 跳过自己
                    if labels[i] == labels[neighbor_idx]:
                        consistent_neighbors += 1
                    total_neighbors += 1
            
            coherence = consistent_neighbors / total_neighbors if total_neighbors > 0 else 0
            return coherence
            
        except Exception:
            return 0.5  # 默认中等置信度

class ClothingRemover:
    """衣物移除器 - 基于语义分析结果移除衣物"""
    
    def __init__(self, semantic_analyzer: SemanticAnalyzer):
        self.analyzer = semantic_analyzer
    
    def remove_clothing_from_points(self, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """从点云中移除衣物点"""
        # 保留身体和头发点，移除衣物和配饰
        body_mask = (labels == 1) | (labels == 3)  # body or hair
        return points[body_mask]
    
    def estimate_nude_measurements(self, clothed_measurements: Dict, clothing_analysis: Dict) -> Tuple[Dict, Dict]:
        """基于衣物分析估算裸体测量值"""
        nude_measurements = clothed_measurements.copy()
        
        # 衣物厚度修正
        corrections = {
            'chest': 0.0,
            'waist': 0.0, 
            'hip': 0.0
        }
        
        # 根据检测到的衣物类型应用修正
        for clothing_item in clothing_analysis.get('detected_clothing', []):
            if clothing_item['type'] == 'upper_clothing':
                # 上装影响胸围和腰围
                thickness = clothing_item['estimated_thickness'] * 100  # 转换为cm
                corrections['chest'] -= thickness * 2  # 周长 = 直径 * π，近似为 * 2
                corrections['waist'] -= thickness * 1.5
                
            elif clothing_item['type'] == 'lower_clothing':
                # 下装主要影响臀围
                thickness = clothing_item['estimated_thickness'] * 100
                corrections['hip'] -= thickness * 2
                corrections['waist'] -= thickness * 0.5
        
        # 应用修正，确保结果合理
        for measurement in ['chest', 'waist', 'hip']:
            corrected_value = clothed_measurements[measurement] + corrections[measurement]
            
            # 设置合理的最小值
            min_values = {'chest': 70, 'waist': 55, 'hip': 80}
            nude_measurements[measurement] = max(min_values[measurement], corrected_value)
        
        return nude_measurements, corrections
