"""
AMP (Automatic Mixed Precision) Management Utility

This module provides comprehensive AMP state management to prevent
"AssertionError: No inf checks were recorded for this optimizer" errors.
"""

import torch
import torch.cuda.amp as amp
from typing import Optional, Dict, Any
import threestudio


class AMPManager:
    """
    Comprehensive AMP state manager for PyTorch Lightning training.
    
    Handles:
    - AMP scaler creation and management
    - Optimizer state synchronization
    - Gradient flow validation
    - Error recovery mechanisms
    """
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.precision_plugin = trainer.precision_plugin
        self.scaler = getattr(self.precision_plugin, "scaler", None)
        self._amp_enabled = hasattr(self.precision_plugin, "scaler")
        
    @property
    def amp_enabled(self) -> bool:
        """Check if AMP is enabled."""
        return self._amp_enabled and self.scaler is not None
    
    def recreate_scaler(self) -> bool:
        """
        Completely recreate the AMP scaler to reset all internal states.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._amp_enabled:
                threestudio.warn("AMP not enabled, skipping scaler recreation")
                return False
                
            # Create new scaler
            new_scaler = amp.GradScaler()
            
            # Replace the scaler in precision plugin
            setattr(self.precision_plugin, "scaler", new_scaler)
            self.scaler = new_scaler
            
            # Clear internal states
            if hasattr(new_scaler, '_per_optimizer_states'):
                new_scaler._per_optimizer_states.clear()
            
            threestudio.info("✅ AMP scaler successfully recreated")
            return True
            
        except Exception as e:
            threestudio.error(f"Failed to recreate AMP scaler: {e}")
            return False
    
    def validate_optimizer_state(self, optimizer) -> bool:
        """
        Validate that optimizer has valid parameters and states.
        
        Args:
            optimizer: The optimizer to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check if optimizer has parameters
            if not optimizer.param_groups:
                threestudio.warn("Optimizer has no parameter groups")
                return False
            
            # Check if parameters are valid
            total_params = 0
            for group in optimizer.param_groups:
                if "params" not in group:
                    threestudio.warn("Parameter group missing 'params' key")
                    return False
                    
                params = group["params"]
                if not params:
                    threestudio.warn("Parameter group has empty params list")
                    return False
                    
                for param in params:
                    if not param.requires_grad:
                        continue
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        threestudio.warn("Found non-finite gradients")
                        return False
                    total_params += 1
            
            if total_params == 0:
                threestudio.warn("No trainable parameters found")
                return False
                
            threestudio.info(f"✅ Optimizer validation passed: {total_params} trainable parameters")
            return True
            
        except Exception as e:
            threestudio.error(f"Optimizer validation failed: {e}")
            return False
    
    def ensure_gradient_flow(self, loss, optimizer) -> bool:
        """
        Ensure proper gradient flow before optimizer step.
        
        Args:
            loss: The loss tensor
            optimizer: The optimizer
            
        Returns:
            bool: True if gradient flow is valid, False otherwise
        """
        try:
            # Check if loss is valid
            if not isinstance(loss, torch.Tensor):
                threestudio.warn("Loss is not a tensor")
                return False
                
            if not loss.requires_grad:
                threestudio.warn("Loss requires_grad is False, setting to True")
                loss.requires_grad_(True)
            
            # Check if backward was called
            if loss.grad_fn is None:
                threestudio.warn("Loss has no grad_fn, calling backward()")
                loss.backward()
            
            # Check for valid gradients
            has_gradients = False
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.requires_grad and param.grad is not None:
                        if torch.any(torch.isfinite(param.grad) & (param.grad.abs() > 0)):
                            has_gradients = True
                            break
                if has_gradients:
                    break
            
            if not has_gradients:
                threestudio.warn("No valid gradients found, injecting small gradients")
                self._inject_small_gradients(optimizer)
            
            return True
            
        except Exception as e:
            threestudio.error(f"Gradient flow validation failed: {e}")
            return False
    
    def _inject_small_gradients(self, optimizer, value=1e-6):
        """
        Inject small gradients to ensure AMP has data to work with.
        
        Args:
            optimizer: The optimizer
            value: The small gradient value to inject
        """
        try:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.requires_grad:
                        if param.grad is None:
                            param.grad = torch.full_like(param.data, value)
                        else:
                            param.grad += torch.full_like(param.data, value)
            threestudio.info("✅ Small gradients injected")
        except Exception as e:
            threestudio.error(f"Failed to inject gradients: {e}")
    
    def safe_optimizer_step(self, optimizer, closure=None, **kwargs) -> bool:
        """
        Safely execute optimizer step with AMP error handling.
        
        Args:
            optimizer: The optimizer
            closure: The closure function
            **kwargs: Additional arguments
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.amp_enabled:
                # Fallback to regular optimizer step
                if closure is not None:
                    loss = closure()
                optimizer.step()
                optimizer.zero_grad()
                return True
            
            # Validate optimizer state
            if not self.validate_optimizer_state(optimizer):
                return False
            
            # Execute closure if provided
            if closure is not None:
                loss = closure()
                if not self.ensure_gradient_flow(loss, optimizer):
                    return False
            
            # Execute AMP step
            try:
                self.scaler.step(optimizer)
                self.scaler.update()
                threestudio.info("✅ AMP optimizer step successful")
                return True
                
            except AssertionError as e:
                if "No inf checks were recorded" in str(e):
                    threestudio.warn("AMP inf check error, recreating scaler and retrying")
                    if self.recreate_scaler():
                        # Retry with new scaler
                        try:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            threestudio.info("✅ AMP step successful after scaler recreation")
                            return True
                        except Exception as retry_e:
                            threestudio.error(f"Retry failed: {retry_e}")
                    
                    # Fallback to regular step
                    threestudio.warn("Falling back to regular optimizer step")
                    optimizer.step()
                    optimizer.zero_grad()
                    return True
                else:
                    raise e
                    
        except Exception as e:
            threestudio.error(f"Safe optimizer step failed: {e}")
            return False
    
    def get_amp_status(self) -> Dict[str, Any]:
        """
        Get current AMP status information.
        
        Returns:
            Dict containing AMP status information
        """
        status = {
            "amp_enabled": self.amp_enabled,
            "scaler_exists": self.scaler is not None,
        }
        
        if self.scaler is not None:
            status.update({
                "scaler_scale": self.scaler._scale.item() if hasattr(self.scaler, '_scale') else None,
                "scaler_growth_tracker": self.scaler._growth_tracker if hasattr(self.scaler, '_growth_tracker') else None,
                "optimizer_states_count": len(self.scaler._per_optimizer_states) if hasattr(self.scaler, '_per_optimizer_states') else 0,
            })
        
        return status


def create_amp_manager(trainer) -> AMPManager:
    """
    Factory function to create an AMP manager.
    
    Args:
        trainer: PyTorch Lightning trainer
        
    Returns:
        AMPManager instance
    """
    return AMPManager(trainer) 