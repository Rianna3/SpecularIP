import torch
import pickle
import numpy as np
import os
from typing import Dict, Any, Optional
import copy


class SMPLXParameterUpdater:
    """用于在GaussianIP训练过程中更新SMPLX参数并保存为完整pkl模型的工具类"""
    
    def __init__(self, original_smplx_model, save_dir: str = "./updated_smplx_models"):
        self.original_model = original_smplx_model
        self.save_dir = save_dir
        self.device = next(original_smplx_model.parameters()).device
        
        os.makedirs(save_dir, exist_ok=True)
        self.original_params = self._extract_all_parameters()
        self.trainable_params = {}
        
    def _extract_all_parameters(self) -> Dict[str, torch.Tensor]:
        """提取SMPLX模型的所有参数"""
        params = {}
        
        # 基础网格参数
        params['v_template'] = self.original_model.v_template.detach().clone()
        params['faces'] = torch.tensor(self.original_model.faces, dtype=torch.long)
        params['lbs_weights'] = self.original_model.lbs_weights.detach().clone()
        
        # 形状和姿态相关参数
        if hasattr(self.original_model, 'shapedirs'):
            params['shapedirs'] = self.original_model.shapedirs.detach().clone()
        if hasattr(self.original_model, 'posedirs'):
            params['posedirs'] = self.original_model.posedirs.detach().clone()
        if hasattr(self.original_model, 'J_regressor'):
            params['J_regressor'] = self.original_model.J_regressor.detach().clone()
            
        return params
    
    def register_trainable_parameter(self, param_name: str, param_tensor: torch.Tensor):
        """注册可训练的参数"""
        self.trainable_params[param_name] = param_tensor
        print(f"Registered trainable parameter: {param_name}, shape: {param_tensor.shape}")
    
    def update_from_trainable_params(self):
        """从注册的可训练参数更新模型参数"""
        for param_name, param_tensor in self.trainable_params.items():
            if param_name in self.original_params:
                self.original_params[param_name] = param_tensor.detach().clone()
    
    def save_updated_model_as_pkl(self, filename: str, gender: str = 'neutral') -> str:
        """将更新后的参数保存为官方SMPLX格式的pkl文件"""
        self.update_from_trainable_params()
        
        save_data = {}
        save_data['v_template'] = self.original_params['v_template'].cpu().numpy()
        save_data['f'] = self.original_params['faces'].cpu().numpy()
        save_data['weights'] = self.original_params['lbs_weights'].cpu().numpy()
        
        if 'shapedirs' in self.original_params:
            save_data['shapedirs'] = self.original_params['shapedirs'].cpu().numpy()
        if 'posedirs' in self.original_params:
            save_data['posedirs'] = self.original_params['posedirs'].cpu().numpy()
        if 'J_regressor' in self.original_params:
            save_data['J_regressor'] = self.original_params['J_regressor'].cpu().numpy()
        
        save_data['gender'] = gender
        save_data['model_type'] = 'smplx'
        
        save_path = os.path.join(self.save_dir, f"{filename}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
            
        print(f"Updated SMPLX model saved to: {save_path}")
        return save_path


def integrate_smplx_updater_to_poser(poser_instance, save_dir: str = "./updated_smplx_models"):
    """将SMPLX参数更新器集成到poser实例中"""
    if not hasattr(poser_instance, 'smplx_model'):
        raise ValueError("Poser instance must have a loaded smplx_model")
        
    updater = SMPLXParameterUpdater(poser_instance.smplx_model, save_dir)
    poser_instance.smplx_updater = updater
    
    def save_updated_smplx(filename: str, gender: str = 'neutral'):
        return poser_instance.smplx_updater.save_updated_model_as_pkl(filename, gender)
    
    poser_instance.save_updated_smplx = save_updated_smplx
    
    print(f"SMPLX updater integrated to poser instance. Save directory: {save_dir}")
    return updater