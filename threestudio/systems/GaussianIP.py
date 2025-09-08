from dataclasses import dataclass

import os
import torch
import torch.nn.functional as F
torch.set_float32_matmul_precision("medium")

import numpy as np
import math
import random
from argparse import ArgumentParser
import yaml
import json
import time
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

import lpips
import matplotlib.pyplot as plt


import threestudio
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
)
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy
from threestudio.utils.poser import Skeleton
from threestudio.utils.typing import *

from gaussiansplatting.gaussian_renderer import render, render_with_smaller_scale
from gaussiansplatting.arguments import PipelineParams, OptimizationParams
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.cameras import Camera
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
from threestudio.models.specular_model import SpecularNetwork



@threestudio.register("gaussianip-system")
class GaussianIP(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # basic settings
        log_path: str = "GaussianIP"
        cur_time: str = ""
        config_path: str = "/home/yunying/GaussianIP/configs/exp.yaml"
        stage: str = "stage1"
        apose: bool = True
        bg_white: bool = False
        radius: float = 4
        ipa_ori: bool = True
        use_pose_controlnet: bool = False
        smplx_path: str = "/home/yunying/GaussianIP/smplx/model"
        pts_num: int = 100000
        sh_degree: int = 0
        height: int = 512
        width: int = 512
        ori_height: int = 1024
        ori_width: int = 1024
        head_offset: float = 0.65

        # specular (Spec-Gaussian / ASG)
        enable_specular: bool = False
        asg_dim: int = 32
        spec_hidden_dim: int = 64
        asg_num_theta: int = 1
        asg_num_phi: int = 4
        specular_weight: float = 0.01

        # 3dgs optimization related
        disable_hand_densification: bool = False
        hand_radius: float = 0.05
        
        # densify & prune settings
        densify_prune_start_step: int = 300
        densify_prune_end_step: int = 2100
        densify_prune_interval: int = 300
        densify_prune_min_opacity: float = 0.15
        densify_prune_screen_size_threshold: int = 20
        densify_prune_world_size_threshold: float = 0.008
        densify_prune_screen_size_threshold_fix_step: int = 1500
        half_scheduler_max_step: int = 1500
        max_grad: float = 0.0002
        gender: str = 'neutral'
        
        # prune_only settings
        prune_only_start_step: int = 1700
        prune_only_end_step: int = 1900
        prune_only_interval: int = 300
        prune_opacity_threshold: float = 0.05
        prune_screen_size_threshold: int = 20
        prune_world_size_threshold: float = 0.008

        # refine related
        refine_start_step: int = 2400
        refine_n_views: int = 64
        refine_train_bs: int = 16
        refine_elevation: float = 17.
        refine_fovy_deg: float = 70.
        refine_camera_distance: float = 1.5
        refine_patch_size: int = 200
        refine_num_bboxes: int = 3
        lambda_l1: float = 1.0
        lambda_lpips: float = 0.5

    cfg: Config

    def configure(self) -> None:
        self.log_path = self.cfg.log_path
        self.cur_time = self.cfg.cur_time
        self.config_path = self.cfg.config_path
        self.stage = self.cfg.stage
        self.radius = self.cfg.radius
        # pass asg_dim into GaussianModel
        self.gaussian = GaussianModel(sh_degree = self.cfg.sh_degree, asg_dim=self.cfg.asg_dim)
        self.background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if self.cfg.bg_white else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)
        self.ipa_ori = self.cfg.ipa_ori
        self.use_pose_controlnet = self.cfg.use_pose_controlnet
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.head_offset = self.cfg.head_offset

        # specular network
        self.enable_specular = self.cfg.enable_specular and (self.cfg.asg_dim > 0)
        if self.enable_specular:
            self.specular_mlp = SpecularNetwork(
                asg_dim=self.cfg.asg_dim,
                num_theta=self.cfg.asg_num_theta,
                num_phi=self.cfg.asg_num_phi,
                hidden_dim=self.cfg.spec_hidden_dim,
            ).to('cuda')
        else:
            self.specular_mlp = None

        self.refine_start_step = self.cfg.refine_start_step
        self.refine_n_views = self.cfg.refine_n_views
        self.refine_train_bs = self.cfg.refine_train_bs
        self.refine_elevation = self.cfg.refine_elevation
        self.refine_fovy_deg = self.cfg.refine_fovy_deg
        self.refine_camera_distance = self.cfg.refine_camera_distance
        self.refine_patch_size = self.cfg.refine_patch_size
        self.refine_num_bboxes = self.cfg.refine_num_bboxes
        self.refine_batch = self.create_refine_batch()
        self.l1_loss_fn = F.l1_loss
        self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.refine_loss = {'training_step': [], 'l1_loss': [], 'lpips_loss': []}
        self.refine_logger = []

        if self.ipa_ori:
            self.skel = Skeleton(style="openpose", apose=self.cfg.apose)
            self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
            self.skel.scale(-10)
        else:
            self.skel = Skeleton(apose=self.cfg.apose)
            self.skel.load_smplx(self.cfg.smplx_path, gender=self.cfg.gender)
            self.skel.scale(-10)
        
        self.cameras_extent = 4.0

        # timing stats
        self.time_stats = {
            'total_start': None,
            'stage1_start': None,
            'stage1_time': 0.0,
            'refine_time': 0.0,
            'stage3_start': None,
            'stage3_time': 0.0,
        }

        # # specular metrics buffers
        # self.spec_metrics = {
        #     "step": [],
        #     "specular_weight": [],
        #     "highlight_coverage": [],
        #     "saturation_rate": [],
        #     "highlight_grad_strength": [],
        #     "multi_view_consistency": [],
        #     # "lpips": [],
        #     # "ssim": [],
        # }




    def pcd(self):
        points = self.skel.sample_smplx_points(N=self.cfg.pts_num)
        colors = np.ones_like(points) * 0.5
        pcd = BasicPointCloud(points, colors, None)
        return pcd
    
    # helpers
    def _to_tensor_if_needed(self, value: Any, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value, device=device)

    def _zero_loss_like_params(self) -> torch.Tensor:
        # Create a zero-valued loss that is connected to model params to keep trainer happy
        try:
            for group in self.gaussian.optimizer.param_groups:
                if len(group['params']) > 0:
                    return group['params'][0].sum() * 0.0
        except Exception:
            pass
        return torch.zeros([], device=self.device, requires_grad=True) * 0.0

    def _format_seconds(self, seconds: float) -> str:
        seconds = float(seconds)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:05.2f}"
        else:
            return f"{m:02d}:{s:05.2f}"


    def forward(self, batch: Dict[str, Any], renderbackground=None, phase='train') -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
            
        images = []
        depths = []
        pose_images = []
        all_vis_all = []
        self.viewspace_point_list = []

        for id in range(batch['c2w'].shape[0]):
            viewpoint_cam  = Camera(c2w = batch['c2w'][id], FoVy = batch['fovy'][id], height = batch['height'], width = batch['width'])

            # compute per-point specular color if enabled
            spec_color = None
            if self.enable_specular and self.gaussian.get_asg_features.numel() > 0:
                # viewdir from point to camera center
                dir_pp = (self.gaussian.get_xyz - viewpoint_cam.camera_center.repeat(self.gaussian.get_xyz.shape[0], 1))
                dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
                normal = self.gaussian.get_normal_axis(dir_pp_normalized)
                asg_feat = self.gaussian.get_asg_features
                # ensure specular mlp is on same device as features to avoid cross-device errors
                if (self.specular_mlp is not None):
                    try:
                        current_dev = next(self.specular_mlp.parameters()).device
                        target_dev = asg_feat.device
                        if current_dev != target_dev:
                            self.specular_mlp = self.specular_mlp.to(target_dev)
                    except StopIteration:
                        pass
                spec_color = self.cfg.specular_weight * self.specular_mlp(asg_feat, dir_pp_normalized, normal)  # (N,3) in [0,1]

            if phase == 'val' or phase == 'test':
                render_pkg = render_with_smaller_scale(viewpoint_cam, self.gaussian, self.pipe, renderbackground, specular_color=spec_color)
            else:
                render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground, specular_color=spec_color)

            image, viewspace_point_tensor, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)
                
            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

            if self.ipa_ori:
                enable_occlusion = True
                head_zoom = (batch['center'][id] == self.head_offset) & (batch['azimuth'][id] > 0)
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy()
                azimuth = batch['azimuth'][id]
                
                if phase == 'train':
                    self.height = self.cfg.height
                    self.width = self.cfg.width
                else:
                    self.height = self.cfg.ori_height
                    self.width = self.cfg.ori_width

                if self.skel.style == 'humansd':
                    pose_image, _ = self.skel.humansd_draw(mvp, self.height, self.width, enable_occlusion)
                else:
                    pose_image, all_vis, _ = self.skel.openpose_draw(mvp, self.height, self.width, azimuth, head_zoom, enable_occlusion)

                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to('cuda')
                all_vis_all.append(all_vis)
                pose_images.append(pose_image)

            else:
                enable_occlusion = True
                mvp = batch['mvp_mtx'][id].detach().cpu().numpy() # [4, 4]
                pose_image, _ = self.skel.draw(mvp, self.height, self.width, enable_occlusion = True)
                # kiui.vis.plot_image(pose_image)
                pose_image = torch.from_numpy(pose_image).to('cuda')
                pose_images.append(pose_image)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        pose_images = torch.stack(pose_images, 0)
        all_vis_all = torch.tensor(all_vis_all, device='cuda')

        self.visibility_filter = self.radii > 0.0

        # pass
        if self.cfg.disable_hand_densification:
            points = self.gaussian.get_xyz # [N, 3]
            hand_centers = torch.from_numpy(self.skel.hand_centers).to(points.dtype).to('cuda') # [2, 3]
            distance = torch.norm(points[:, None, :] - hand_centers[None, :, :], dim=-1) # [N, 2]
            hand_mask = distance.min(dim=-1).values < self.cfg.hand_radius # [N]
            self.visibility_filter = self.visibility_filter & (~hand_mask)

        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg['pose'] = pose_images
        render_pkg['all_vis_all'] = all_vis_all
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg["scale"] = self.gaussian.get_scaling

        return {
            **render_pkg,
        }

    def _compute_spec_metrics(self, images: torch.Tensor) -> Tuple[float, float, float, float]:
        """
        images: (B,H,W,3) in [0,1]
        returns: coverage, saturation, grad_strength, multi_view_consistency
        """
        if images.ndim != 4:
            return 0.0, 0.0, 0.0, float('nan')
        B = images.shape[0]
        y = 0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]  # (B,H,W)
        thr_hi = 0.95
        mask = (y > thr_hi).float()
        coverage = mask.mean().item()
        saturation = (images > 0.98).float().mean().item()
        # gradient magnitude on luminance
        gx = torch.abs(y[..., 1:] - y[..., :-1])
        gy = torch.abs(y[:, 1:, :] - y[:, :-1, :])
        # align masks
        mx = mask[..., :-1]
        my = mask[:, :-1, :]
        grad_mag = torch.cat([gx, gy], dim=1) if gx.shape[1:] == gy.shape[1:] else gx
        mm = (mx if gx.shape == mx.shape else mask)
        grad_strength = (grad_mag * mm).mean().item() if mm.numel() > 0 else 0.0
        # multi-view consistency: coefficient of variation of highlight energy
        with torch.no_grad():
            if B > 1:
                highlight_energy = (y * mask).view(B, -1).mean(dim=1)
                m = highlight_energy.mean()
                s = highlight_energy.std(unbiased=False)
                mvc = (1.0 / (1.0 + (s / (m + 1e-8)))) .item()
            else:
                mvc = float('nan')
        return coverage, saturation, grad_strength, mvc


    def create_refine_batch(self):
        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg = torch.linspace(-180., 180.0, self.refine_n_views + 1)[: self.refine_n_views]
        azimuth = azimuth_deg * math.pi / 180

        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        elevation_deg = torch.full_like(azimuth_deg, self.refine_elevation)
        elevation = elevation_deg * math.pi / 180

        camera_distances: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.refine_camera_distance)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),  # x
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),  # y
                camera_distances * torch.sin(elevation),                       # z
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)

        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(1, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(elevation_deg, self.refine_fovy_deg)
        fovy = fovy_deg * math.pi / 180

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)

        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
        c2w: Float[Tensor, "B 4 4"] = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
        c2w[:, 3, 3] = 1.0

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, self.width / self.height, 0.1, 1000.0)  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "mvp_mtx": mvp_mtx,
            "center": center[:,2],
            "c2w": c2w,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "height": self.cfg.ori_height,
            "width": self.cfg.ori_width,
            "fovy":fovy
        }


    def render_refine_rgb(self, phase='init', renderbackground=None, with_grad: bool = False):
        if renderbackground is None:
            renderbackground = self.background_tensor

        images = []
        depths = []
        pose_images = []
        self.viewspace_point_list = []

        assert phase in ['init', 'random', 'debug']

        if phase == 'init':
            id_list = [i for i in range(self.refine_n_views)]
        elif phase == 'random':
            # cap batch size by available GPU memory
            bs = self.refine_train_bs
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                # heuristic: ~2.5GB per view at 1024; scale down for safety
                max_views = max(1, min(bs, int((free_bytes / (2.5 * 1024**3)))))
                bs = max(1, min(bs, max_views))
            except Exception:
                pass
            id_list = random.sample(range(self.refine_n_views), bs)
        else:
            id_list = [0, 8, 16, 24, 31]

        refine_height = self.refine_batch['height']
        refine_width  = self.refine_batch['width']

        # refine 阶段不需要 densify 统计
        self.viewspace_point_list = []
        self.refine_radii = None

        CHUNK = 4  # 显存紧就用 4；更宽裕可调到 8
        # 训练阶段需要梯度，不要切 eval()
        if not with_grad and getattr(self, "specular_mlp", None) is not None:
            self.specular_mlp.eval()

        # 使用 autocast 节省显存；训练阶段保留梯度
        autocast_ctx = torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16)
        ctx_mgr = autocast_ctx if with_grad else torch.inference_mode()

        images_buf, depths_buf, poses_buf = [], [], []
        with ctx_mgr:
            for s in range(0, len(id_list), CHUNK):
                sub = id_list[s:s+CHUNK]
                for idx, id in enumerate(sub):
                    cam = Camera(
                        c2w=self.refine_batch['c2w'][id],
                        FoVy=self.refine_batch['fovy'][id],
                        height=refine_height,
                        width=refine_width
                    )
                    # 用缩放版渲染，显著省显存
                    render_pkg = render_with_smaller_scale(cam, self.gaussian, self.pipe, renderbackground)

                    image = render_pkg["render"].permute(1, 2, 0)
                    depth = render_pkg["depth_3dgs"].permute(1, 2, 0)
                    radii = render_pkg["radii"]

                    self.refine_radii = radii if self.refine_radii is None else torch.max(radii, self.refine_radii)

                    # 生成姿态图（GPU 上算）
                    enable_occlusion = True
                    head_zoom = (self.refine_batch['center'][id] == self.head_offset) & (self.refine_batch['azimuth'][id] > 0)
                    mvp = self.refine_batch['mvp_mtx'][id].detach().cpu().numpy()
                    azimuth = self.refine_batch['azimuth'][id]
                    if self.skel.style == 'humansd':
                        pose_image, _ = self.skel.humansd_draw(mvp, refine_height, refine_width, enable_occlusion)
                    else:
                        pose_image, _, _ = self.skel.openpose_draw(mvp, refine_height, refine_width, azimuth, head_zoom, enable_occlusion)
                    pose = torch.from_numpy(pose_image).to('cuda')

                    if with_grad:
                        images_buf.append(image)
                        depths_buf.append(depth)
                        poses_buf.append(pose)
                    else:
                        images_buf.append(image.detach().float().cpu())
                        depths_buf.append(depth.detach().float().cpu())
                        poses_buf.append(pose.detach().float().cpu())

                    del image, depth, pose, render_pkg
                torch.cuda.empty_cache()

        if with_grad:
            images = torch.stack(images_buf, 0)
            depths = torch.stack(depths_buf, 0)
            pose_images = torch.stack(poses_buf, 0)
            self.refine_visibility_filter = (self.refine_radii > 0.0) if self.refine_radii is not None else None
        else:
            images = torch.stack(images_buf, 0)
            depths = torch.stack(depths_buf, 0)
            pose_images = torch.stack(poses_buf, 0)
            self.refine_visibility_filter = (self.refine_radii > 0.0) if self.refine_radii is not None else None

        render_pkg = {
            "comp_rgb": images,
            "depth": depths,
            "pose": pose_images,
        }
        return {**render_pkg}, id_list

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # start timers
        now = time.perf_counter()
        self.time_stats['total_start'] = now
        if self.stage == "stage1":
            self.time_stats['stage1_start'] = now
        else:
            self.time_stats['stage3_start'] = now
        # stage 1: AHDS training
        if self.stage == "stage1":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            self.guidance.prepare_for_sds(self.prompt_processor.prompt, self.prompt_processor.negative_prompt, self.prompt_processor.null_prompt)
        # stage 3: 3d reconstruction
        else:
            self.refined_rgbs_small = torch.load(os.path.join(self.log_path, self.cur_time, 'after_refine.pth'))['refined_rgbs_small'].to(self.device)


    def training_step(self, batch, batch_idx):
        # stage 1: SDS
        if self.true_global_step < self.refine_start_step-1 and self.stage == "stage1":
            self.gaussian.update_learning_rate(self.true_global_step)

            # render under autocast to save memory, keep grads
            # with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            out = self(batch, phase='train')

            prompt_utils = self.prompt_processor()
            images = out["comp_rgb"]
            control_images = out['pose']
            all_vis_all = out['all_vis_all']

            guidance_out = self.guidance(self.true_global_step, images, control_images, prompt_utils, self.use_pose_controlnet, all_vis_all, **batch)

            # init loss as tensor on device
            device = images.device
            loss = torch.zeros([], device=device)

            # loss_sds (ensure tensor)
            loss_sds = guidance_out.get('loss_sds', None)
            if loss_sds is None:
                raise RuntimeError("guidance_out must contain 'loss_sds'")
            if not torch.is_tensor(loss_sds):
                loss_sds = torch.as_tensor(loss_sds, device=device)
            loss = loss + loss_sds * self.C(self.cfg.loss['lambda_sds'])
            
            # loss_sparsity
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss = loss + loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)
            
            # loss_opaque
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss = loss + loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))

            return {"loss": loss}

        # hit refine step: export data, keep trainer happy with zero-loss
        elif self.true_global_step == self.refine_start_step-1 and self.stage == "stage1":
            t0 = time.perf_counter()
            gs_out, _ = self.render_refine_rgb(phase='init', renderbackground=None, with_grad=False)
            images = gs_out["comp_rgb"].detach()      # [refine_n_views, H, W, 3]
            control_images = gs_out['pose'].detach()  # [refine_n_views, H, W, 3]
            # save image data before refine
            images = images.to('cpu')
            control_images = control_images.to('cpu')
            torch.save({'images': images, 'control_images': control_images}, os.path.join(self.log_path, self.cur_time, 'before_refine.pth'))

            # self.refined_rgbs, self.view_idx_all = self.guidance.refine_rgb(images, control_images, self.prompt_processor.prompt)  # [refine_n_views, H, W, 3]
            
            self.view_idx_all = [24, 8, 16, 0, 20, 28, 4, 12, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            # save images before refine
            for i, _ in enumerate(self.view_idx_all):
                cur_raw_rgb = images[i]
                cur_control_image = control_images[i]
                # cur_refined_rgb = self.refined_rgbs[i]
                self.save_image(f"raw_rgb_{i}.png", cur_raw_rgb)
                self.save_image(f"control_image_{i}.png", cur_control_image)
                # self.save_image(f"refined_rgb_{view_idx}.png", cur_refined_rgb)

            self.time_stats['refine_time'] += (time.perf_counter() - t0)
            torch.cuda.empty_cache()
            return {"loss": self._zero_loss_like_params()}
        
        # stage 3: supervised reconstruction
        elif self.stage == "stage3":
            self.gaussian.update_learning_rate(self.true_global_step + self.refine_start_step)
            gs_out, id_list = self.render_refine_rgb(phase='random', renderbackground=None, with_grad=True)
            rgb_render = gs_out["comp_rgb"].permute(0, 3, 1, 2)[:, :, 60:890, 220:800]
            rgb_render_small = F.interpolate(rgb_render, scale_factor=0.5, mode="bilinear", align_corners=False)
            rgb_gt_small = self.refined_rgbs_small[id_list]

            # init loss
            loss = torch.zeros([], device=rgb_render_small.device)
            # l1 and lpips loss, use crop and downsample to save memory
            l1_loss = self.l1_loss_fn(rgb_render_small, rgb_gt_small)
            # compute LPIPS on further downsampled images to save VRAM
            lpips_size = (256, 256)
            pred_lpips = F.interpolate(rgb_render_small, size=lpips_size, mode="bilinear", align_corners=False)
            gt_lpips = F.interpolate(rgb_gt_small, size=lpips_size, mode="bilinear", align_corners=False)
            # ensure LPIPS module is on the same device as inputs
            if next(self.lpips_loss_fn.parameters(), torch.tensor([])).device != pred_lpips.device:
                self.lpips_loss_fn = self.lpips_loss_fn.to(pred_lpips.device)
            lpips_loss = self.lpips_loss_fn(pred_lpips, gt_lpips, normalize=True).mean()
            loss = loss + self.cfg.lambda_l1 * l1_loss + self.cfg.lambda_lpips * lpips_loss

            # record loss
            self.refine_loss['training_step'].append(self.true_global_step)
            self.refine_loss['l1_loss'].append(l1_loss.item())
            self.refine_loss['lpips_loss'].append(lpips_loss.item())

            # return {"loss": loss}

        # # after refine step in stage1 but trainer still running: return zero loss to avoid crashes
        # elif self.stage == "stage1" and self.true_global_step > self.refine_start_step:
        #     return {"loss": self._zero_loss_like_params()}

        # default safe return
        return {"loss": self._zero_loss_like_params()}


    def on_before_optimizer_step(self, optimizer):
        if self.true_global_step % 100 == 0:
            threestudio.info('Gaussian points num: {}'.format(self.gaussian.get_features.shape[0]))
        # skip densify/prune if gradients are not available yet
        if not getattr(self, 'viewspace_point_list', None) or len(self.viewspace_point_list) == 0:
            return
        if self.true_global_step < self.refine_start_step and self.stage == "stage1":
            with torch.no_grad():
                if self.true_global_step < self.cfg.densify_prune_end_step:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                    # densify_and_prune
                    self.min_opacity = self.cfg.densify_prune_min_opacity if self.true_global_step > 1900 else 0.05
                    if self.true_global_step > self.cfg.densify_prune_start_step and self.true_global_step % self.cfg.densify_prune_interval == 0:
                        densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
                        self.gaussian.densify_and_prune(self.cfg.max_grad, self.min_opacity, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

                # "prune-only" phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
                if self.true_global_step > self.cfg.prune_only_start_step and self.true_global_step < self.cfg.prune_only_end_step:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                    if self.true_global_step % self.cfg.prune_only_interval == 0:
                        self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)
        
        if self.stage == "stage3":
            with torch.no_grad():
                if self.true_global_step + self.refine_start_step < 10000:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    if self.true_global_step == 0:
                        # When stage 3 starts, the loaded gaussians don't have max_radii2D
                        self.gaussian.max_radii2D = self.refine_radii  # self.get_xyz.shape[0]
                        self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
                        self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
                    else:
                        self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
                        self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
                    # densify_and_prune
                    if self.true_global_step + self.refine_start_step == 2500:
                        densify_prune_screen_size_threshold = self.cfg.densify_prune_screen_size_threshold if self.true_global_step > self.cfg.densify_prune_screen_size_threshold_fix_step else None
                        self.gaussian.densify_and_prune(self.cfg.max_grad, 0.05, self.cameras_extent, densify_prune_screen_size_threshold, self.cfg.densify_prune_world_size_threshold) 

                # "prune-only" phase according to Gaussian size, rather than the stochastic gradient to eliminate floating artifacts.
                if self.true_global_step + self.refine_start_step > 2500 and self.true_global_step + self.refine_start_step < 3000:
                    viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                    for idx in range(len(self.viewspace_point_list)):
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                    # Keep track of max radii in image-space for pruning
                    self.gaussian.max_radii2D[self.refine_visibility_filter] = torch.max(self.gaussian.max_radii2D[self.refine_visibility_filter], self.refine_radii[self.refine_visibility_filter])
                    self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.refine_visibility_filter)
                    if self.true_global_step + self.refine_start_step % self.cfg.prune_only_interval == 0:
                        self.gaussian.prune_only(min_opacity=self.cfg.prune_opacity_threshold, max_world_size=self.cfg.prune_world_size_threshold)


    def validation_step(self, batch, batch_idx):
        out = self(batch, phase='val')
        img0 = out["comp_rgb"][0].detach().cpu()

        if self.stage == "stage1":
            self.save_image(f"it{self.true_global_step}-{batch['index'][0]}_rgb.png", img0)
        else:
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-{batch['index'][0]}_rgb.png", img0)


    def on_validation_epoch_end(self):
        pass


    # test the gaussians
    def test_step(self, batch, batch_idx):
        if self.stage == "stage1":
            pass
        else:
            pass
            # Always test on black background for consistency
            bg_color = [1, 1, 1] if self.cfg.bg_white else [0, 0, 0]
            background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            out = self(batch, renderbackground=background_tensor, phase='test')
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/rgb/{batch['index'][0]}.png", out["comp_rgb"][0])
            self.save_image(f"it{self.true_global_step + self.refine_start_step}-test/pose/{batch['index'][0]}.png", out["pose"][0])        
        

    # save something
    def on_test_epoch_end(self):
        if self.stage == "stage1":
            self.gaussian.save_ply(os.path.join(self.log_path, self.cur_time, f"it{self.true_global_step}.ply"))
        else:
            self.save_img_sequence(
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                f"it{self.true_global_step + self.refine_start_step}-test/rgb",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step + self.refine_start_step,
            )
            save_path = self.get_save_path(f"it{self.true_global_step + self.refine_start_step}-test/last.ply")
            self.gaussian.save_ply(save_path)

            # change the max_steps in config.yaml from total_step to refine_start_step
            config_file_path = self.config_path

            # read config.yaml
            with open(config_file_path, 'r') as file:
                config = yaml.safe_load(file)

            # change args
            config['system']['stage'] = 'stage1'
            config['trainer']['max_steps'] = self.refine_start_step

            # write it back to config.yaml
            with open(config_file_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)

            print(f"Updated max_steps to {self.refine_start_step} in {self.config_path}.")
            

    def on_fit_end(self) -> None:
        # finalize timings and print summary
        try:
            now = time.perf_counter()
            total = (now - self.time_stats['total_start']) if self.time_stats['total_start'] is not None else None
            if self.stage == "stage1" and self.time_stats['stage1_start'] is not None:
                self.time_stats['stage1_time'] = now - self.time_stats['stage1_start']
            if self.stage == "stage3" and self.time_stats['stage3_start'] is not None:
                self.time_stats['stage3_time'] = now - self.time_stats['stage3_start']

            msg_parts = ["==== Time Summary ===="]
            if self.time_stats['stage1_time']:
                msg_parts.append(f"Stage1 time: {self._format_seconds(self.time_stats['stage1_time'])} ({self.time_stats['stage1_time']:.2f}s)")
            if self.time_stats['refine_time']:
                msg_parts.append(f"Refine time: {self._format_seconds(self.time_stats['refine_time'])} ({self.time_stats['refine_time']:.2f}s)")
            if self.time_stats['stage3_time']:
                msg_parts.append(f"Stage3 time: {self._format_seconds(self.time_stats['stage3_time'])} ({self.time_stats['stage3_time']:.2f}s)")
            if total is not None:
                msg_parts.append(f"Total time:  {self._format_seconds(total)} ({total:.2f}s)")
            print("\n".join(msg_parts))
        except Exception:
            pass

    def configure_optimizers(self):
        if self.stage == "stage1":
            opt = OptimizationParams(self.parser)
            point_cloud = self.pcd()
            self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)
            self.gaussian.training_setup(opt)
            # combine specular params into the same optimizer
            if self.enable_specular and (self.specular_mlp is not None):
                # build a new optimizer with existing groups plus specular
                groups = []
                for g in self.gaussian.optimizer.param_groups:
                    groups.append({'params': g['params'], 'lr': g['lr'], 'name': g.get('name', 'gs')})
                groups.append({'params': self.specular_mlp.parameters(), 'lr': opt.feature_lr, 'name': 'specular_mlp'})
                self.gaussian.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
            ret = {"optimizer": self.gaussian.optimizer}
        else:
            # load 3dgs from stage 1
            opt = OptimizationParams(self.parser)
            self.gaussian.load_ply(os.path.join(self.log_path, self.cur_time, f"it{self.refine_start_step}.ply"))
            self.gaussian.training_setup(opt)
            if self.enable_specular and (self.specular_mlp is not None):
                groups = []
                for g in self.gaussian.optimizer.param_groups:
                    groups.append({'params': g['params'], 'lr': g['lr'], 'name': g.get('name', 'gs')})
                groups.append({'params': self.specular_mlp.parameters(), 'lr': opt.feature_lr, 'name': 'specular_mlp'})
                self.gaussian.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
            ret = {"optimizer": self.gaussian.optimizer}
        return ret

    # def on_fit_end(self) -> None:
    #     # Write spec metrics CSV
    #     try:
    #         import csv
    #         save_dir = os.path.join(self.log_path, self.cur_time)
    #         os.makedirs(save_dir, exist_ok=True)
    #         csv_path = os.path.join(save_dir, "spec_metrics.csv")
    #         keys = ["step","specular_weight","highlight_coverage","saturation_rate","highlight_grad_strength","multi_view_consistency","lpips","ssim"]
    #         with open(csv_path, "w", newline="") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(keys)
    #             n = len(self.spec_metrics["step"]) if "step" in self.spec_metrics else 0
    #             for i in range(n):
    #                 row = [self.spec_metrics[k][i] for k in keys]
    #                 writer.writerow(row)
    #     except Exception:
    #         pass

    #     # After all training finished (when stage3 run completes), evaluate final metrics
    #     if self.stage == "stage3":
    #         try:
    #             self.eval()
    #             torch.cuda.empty_cache()
    #             # render all refine views with specular path
    #             refine_height = self.refine_batch['height']
    #             refine_width = self.refine_batch['width']
    #             renders = []
    #             with torch.no_grad():
    #                 for vid in range(self.refine_n_views):
    #                     cam = Camera(c2w=self.refine_batch['c2w'][vid], FoVy=self.refine_batch['fovy'][vid], height=refine_height, width=refine_width)
    #                     # compute specular color like in forward
    #                     dir_pp = (self.gaussian.get_xyz - cam.camera_center.repeat(self.gaussian.get_xyz.shape[0], 1))
    #                     dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-8)
    #                     normal = self.gaussian.get_normal_axis(dir_pp_normalized)
    #                     asg_feat = self.gaussian.get_asg_features
    #                     spec_color = None
    #                     if self.enable_specular and asg_feat.numel() > 0:
    #                         spec_color = self.cfg.specular_weight * self.specular_mlp(asg_feat, dir_pp_normalized, normal)
    #                     rpkg = render(cam, self.gaussian, self.pipe, self.background_tensor, specular_color=spec_color)
    #                     img = rpkg["render"].permute(1, 2, 0)  # HWC
    #                     renders.append(img)
    #             renders = torch.stack(renders, dim=0)  # (V,H,W,3)
    #             # match training crop/downsample
    #             rgb_render = renders.permute(0, 3, 1, 2)[:, :, 60:890, 220:800]
    #             rgb_render_small = F.interpolate(rgb_render, scale_factor=0.5, mode="bilinear", align_corners=False)
    #             rgb_render_small = rgb_render_small.clamp(0, 1)
    #             # ground-truth refined images
    #             gts = self.refined_rgbs_small  # (V,3,h,w), values in [0,1]
    #             # compute LPIPS (mean)
    #             # lpips_val = self.lpips_loss_fn(rgb_render_small, gts, normalize=True).mean().item()
    #             # compute SSIM/PSNR (mean over all views)
    #             ssim_vals = []
    #             psnr_vals = []
    #             for i in range(min(gts.shape[0], rgb_render_small.shape[0])):
    #                 pred = rgb_render_small[i].permute(1, 2, 0).detach().cpu().numpy()
    #                 gt = gts[i].permute(1, 2, 0).detach().cpu().numpy()
    #                 # skimage expects float in [0,1]
    #                 ssim_i = sk_ssim(gt, pred, channel_axis=2, data_range=1.0)
    #                 psnr_i = sk_psnr(gt, pred, data_range=1.0)
    #                 ssim_vals.append(float(ssim_i))
    #                 psnr_vals.append(float(psnr_i))
    #             results = {
    #                 "psnr": float(np.mean(psnr_vals)) if len(psnr_vals) > 0 else None,
    #                 "ssim": float(np.mean(ssim_vals)) if len(ssim_vals) > 0 else None,
    #                 # "lpips_vgg": float(lpips_val),
    #                 "num_views": int(renders.shape[0]),
    #                 "crop": [60, 890, 220, 800],
    #             }
    #             save_dir = os.path.join(self.log_path, self.cur_time)
    #             os.makedirs(save_dir, exist_ok=True)
    #             with open(os.path.join(save_dir, "results.json"), "w") as jf:
    #                 json.dump(results, jf, indent=2)
    #         except Exception as e:
    #             try:
    #                 # best-effort log
    #                 with open(os.path.join(self.log_path, self.cur_time, "results_error.txt"), "w") as ef:
    #                     ef.write(str(e))
    #             except Exception:
    #                 pass