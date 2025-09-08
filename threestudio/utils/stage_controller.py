"""
Stage Controller for Dual-Stage SDS Loss
Manages the transition between coarse and refinement training phases.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from typing import Dict, Any, Optional
from collections import defaultdict
import threestudio


@dataclass
class StageConfig:
    """Configuration for training stages"""
    switch_iter: int = 2000
    lambda_detail_sds: float = 1.5
    lr_refine_ratio: float = 0.1
    detail_prompt: str = "high-quality detailed portrait, 8k resolution"
    detail_negative_prompt: str = "blurry, low quality, pixelated"
    
    # Region-specific configurations
    region_prompts: Optional[Dict[str, str]] = None
    crop_regions: Optional[Dict[str, List[float]]] = None
    enabled_regions: Optional[List[str]] = None


class DualStageController:
    """
    Controls the transition between coarse and refinement stages in SDS training.
    
    Features:
    - Automatic stage switching at specified iteration
    - Learning rate adjustment for refinement phase
    - Region-specific detail enhancement
    - Progressive loss weight scheduling
    """
    
    def __init__(self, config: StageConfig):
        self.config = config
        self.current_stage = "coarse"
        self.stage_switched = False
        self.refinement_start_iter: Optional[int] = None
        
        # Initialize region prompts if provided
        if config.region_prompts is None:
            self.config.region_prompts = {
                "face": "detailed facial features, skin texture, 4k portrait",
                "clothing": "detailed fabric texture, clothing material, high resolution",
                "hands": "detailed hand anatomy, skin texture, realistic fingers"
            }
        
        if config.crop_regions is None:
            self.config.crop_regions = {
                "face_bbox": [0.3, 0.1, 0.7, 0.6],
                "torso_bbox": [0.2, 0.3, 0.8, 0.9]
            }
            
        if config.enabled_regions is None:
            self.config.enabled_regions = ["face", "torso"]
    
    def should_switch_stage(self, current_iter: int) -> bool:
        """Check if we should switch to refinement stage"""
        return current_iter >= self.config.switch_iter and not self.stage_switched
    
    def switch_to_refinement(self, current_iter: int, optimizer) -> None:
        """Switch to refinement stage with learning rate adjustment"""
        if self.stage_switched:
            return
            
        self.current_stage = "refinement"
        self.stage_switched = True
        self.refinement_start_iter = current_iter
        
        # Adjust learning rate
        if hasattr(optimizer, 'param_groups'):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.config.lr_refine_ratio
        elif isinstance(optimizer, list) and len(optimizer) > 0:
            for param_group in optimizer[0].param_groups:
                param_group['lr'] *= self.config.lr_refine_ratio
                
        print(f"🔄 Switched to refinement stage at iteration {current_iter}")
        print(f"📉 Learning rate reduced by factor {self.config.lr_refine_ratio}")
    
    def get_loss_weights(self, current_iter: int) -> Dict[str, float]:
        """Get loss weights for current training stage"""
        if self.current_stage == "coarse":
            return {
                "lambda_sds": 1.0,
                "lambda_detail_sds": 0.0
            }
        else:
            # Progressive weight increase for detail loss
            if self.refinement_start_iter is not None:
                progress = min(1.0, (current_iter - self.refinement_start_iter) / 500)
                detail_weight = self.config.lambda_detail_sds * progress
            else:
                detail_weight = 0.0
            
            return {
                "lambda_sds": 0.8,  # Reduce standard SDS weight
                "lambda_detail_sds": detail_weight
            }
    
    def crop_image_regions(self, image: torch.Tensor, region_name: str) -> torch.Tensor:
        """
        Crop image to specific region for detail enhancement
        
        Args:
            image: Input image tensor [B, H, W, C]
            region_name: Name of region to crop ("face", "torso", etc.)
            
        Returns:
            Cropped image tensor
        """
        if (self.config.enabled_regions is None or 
            region_name not in self.config.enabled_regions):
            return image
            
        bbox_key = f"{region_name}_bbox"
        if (self.config.crop_regions is None or 
            bbox_key not in self.config.crop_regions):
            return image
            
        bbox = self.config.crop_regions[bbox_key]
        B, H, W, C = image.shape
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] * W)
        y1 = int(bbox[1] * H)
        x2 = int(bbox[2] * W)
        y2 = int(bbox[3] * H)
        
        # Crop and resize back to original size for consistency
        cropped = image[:, y1:y2, x1:x2, :]
        resized = F.interpolate(
            cropped.permute(0, 3, 1, 2), 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        return resized
    
    def get_detail_prompt(self, region_name: Optional[str] = None) -> str:
        """Get appropriate prompt for detail enhancement"""
        if (region_name and 
            self.config.region_prompts and 
            region_name in self.config.region_prompts):
            return self.config.region_prompts[region_name]
        return self.config.detail_prompt
    
    def compute_detail_sds_loss(self, 
                               rendered_image: torch.Tensor,
                               guidance_model,
                               prompt_utils,
                               current_iter: int,
                               **kwargs) -> torch.Tensor:
        """
        Compute detail-oriented SDS loss with region-specific enhancement
        
        Args:
            rendered_image: Rendered image from 3DGS [B, H, W, C]
            guidance_model: SDS guidance model
            prompt_utils: Prompt utilities with detail prompts
            current_iter: Current training iteration
            
        Returns:
            Detail SDS loss tensor
        """
        if self.current_stage == "coarse":
            return torch.tensor(0.0, device=rendered_image.device)
        
        if self.config.enabled_regions is None:
            return torch.tensor(0.0, device=rendered_image.device)
        
        total_detail_loss = torch.tensor(0.0, device=rendered_image.device)
        num_regions = len(self.config.enabled_regions)
        
        for region_name in self.config.enabled_regions:
            # Crop to region of interest
            cropped_image = self.crop_image_regions(rendered_image, region_name)
            
            # Get region-specific prompt
            detail_prompt = self.get_detail_prompt(region_name)
            
            # Create detail prompt utils
            detail_prompt_utils = prompt_utils.get_prompt_utils({
                "prompt": detail_prompt,
                "negative_prompt": self.config.detail_negative_prompt
            })
            
            # Compute SDS loss for this region
            region_loss = guidance_model(
                step=current_iter,
                rgb=cropped_image,
                prompt_utils=detail_prompt_utils,
                **kwargs
            )
            
            if isinstance(region_loss, dict) and "loss_sds" in region_loss:
                total_detail_loss = total_detail_loss + region_loss["loss_sds"]
            elif isinstance(region_loss, torch.Tensor):
                total_detail_loss = total_detail_loss + region_loss
        
        return total_detail_loss / num_regions if num_regions > 0 else total_detail_loss
    
    def get_stage_info(self) -> Dict[str, Union[str, bool, int, None]]:
        """Get current stage information for logging"""
        return {
            "current_stage": self.current_stage,
            "stage_switched": self.stage_switched,
            "refinement_start_iter": self.refinement_start_iter,
            "switch_iter": self.config.switch_iter
        } 


class StageController:
    """
    管理双阶段SDS训练的平滑切换控制器
    
    功能:
    1. 平滑权重混合 (防止loss跳变)
    2. 学习率降温
    3. 优化器动量重置
    4. 几何密度冻结控制
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = cfg.get("enabled", False)
        self.switch_iter = cfg.get("switch_iter", 1800)
        self.blend_iters = cfg.get("blend_iters", 300)
        self.lambda_detail_max = cfg.get("lambda_detail_max", 1.0)
        self.lr_refine_ratio = cfg.get("lr_refine_ratio", 0.2)
        self.freeze_densify = cfg.get("freeze_densify", True)
        self.max_points_stage2 = cfg.get("max_points_stage2", 400000)
        
        # 状态追踪
        self.optimizer_reset = False
        self.stage = "standard"  # "standard" -> "blending" -> "detail"
        
        threestudio.info(f"StageController initialized: enabled={self.enabled}, switch_iter={self.switch_iter}")
    
    def get_current_stage(self, step: int) -> str:
        """获取当前训练阶段"""
        if not self.enabled:
            return "standard"
        
        if step < self.switch_iter:
            return "standard"
        elif step < self.switch_iter + self.blend_iters:
            return "blending"
        else:
            return "detail"
    
    def get_loss_weights(self, step: int) -> Dict[str, float]:
        """计算当前步骤的损失权重"""
        if not self.enabled:
            return {"lambda_sds": 1.0, "lambda_detail_sds": 0.0}
        
        stage = self.get_current_stage(step)
        
        if stage == "standard":
            return {"lambda_sds": 1.0, "lambda_detail_sds": 0.0}
        elif stage == "blending":
            # 线性混合: alpha从0到1
            alpha = (step - self.switch_iter) / self.blend_iters
            alpha = max(0.0, min(1.0, alpha))  # clip to [0,1]
            return {
                "lambda_sds": 1.0 - 0.5 * alpha,  # 从1.0平滑降到0.5
                "lambda_detail_sds": self.lambda_detail_max * alpha  # 从0平滑升到max
            }
        else:  # detail stage
            return {"lambda_sds": 0.5, "lambda_detail_sds": self.lambda_detail_max}
    
    def should_reset_optimizer(self, step: int) -> bool:
        """判断是否需要重置优化器状态"""
        if not self.enabled:
            return False
        
        # 只在切换开始时重置一次
        if step == self.switch_iter and not self.optimizer_reset:
            self.optimizer_reset = True
            return True
        return False
    
    def reset_optimizer_state(self, optimizer: torch.optim.Optimizer, base_lr: float):
        """重置优化器状态并调整学习率"""
        threestudio.info("Resetting optimizer state for stage transition")
        
        # 清空动量缓存
        optimizer.state = defaultdict(dict)
        
        # 调整学习率
        new_lr = base_lr * self.lr_refine_ratio
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        threestudio.info(f"Learning rate adjusted: {base_lr} -> {new_lr}")
    
    def should_allow_densify(self, step: int, current_points: int) -> bool:
        """判断是否允许densification"""
        if not self.enabled:
            return True
        
        stage = self.get_current_stage(step)
        
        # 第二阶段冻结densify或限制点数
        if stage in ["blending", "detail"]:
            if self.freeze_densify:
                return False
            else:
                return current_points < self.max_points_stage2
        
        return True
    
    def get_stage_info(self, step: int) -> Dict[str, Any]:
        """获取当前阶段的完整信息 (用于logging)"""
        if not self.enabled:
            return {"stage": "standard", "enabled": False}
        
        stage = self.get_current_stage(step)
        weights = self.get_loss_weights(step)
        
        return {
            "stage": stage,
            "enabled": True,
            "step": step,
            "switch_iter": self.switch_iter,
            "blend_progress": max(0.0, min(1.0, (step - self.switch_iter) / self.blend_iters)) if stage != "standard" else 0.0,
            "weights": weights,
            "allow_densify": self.should_allow_densify(step, 0)  # points数在外部传入
        } 