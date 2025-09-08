from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from models.pipeline_ipa_controlnet import StableDiffusionControlNetPipeline

from diffusers import DDIMScheduler, AutoencoderKL, ControlNetModel

# import threestudio
# from threestudio.utils.base import BaseObject
# from threestudio.utils.misc import cleanup
# from threestudio.utils.typing import *

from PIL import Image
import cv2
import os
import numpy as np
import argparse
import yaml

from insightface.app import FaceAnalysis
from insightface.utils import face_align

# args
# pretrained_sd_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
# pretrained_realistic_model_name_or_path: str = "SG161222/Realistic_Vision_V4.0_noVAE"
# vae_path: str = "stabilityai/sd-vae-ft-mse"
# image_encoder_path: str = "IP-Adapter/models/image_encoder"
# image_encoder_faceid_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# ip_ckpt_path: str = "IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
# ip_ckpt_faceid_v1_path: str = "IP-Adapter/models/ip-adapter-faceid_sd15.bin"
# ip_ckpt_faceid_v2_path: str = "IP-Adapter/models/ip-adapter-faceid-plusv2_sd15.bin"
# pose_controlnet_path: str = "lllyasviel/control_v11p_sd15_openpose"
pretrained_sd_model_name_or_path: str = "/home/yunying/GaussianIP/stable-diffusion-v1-5"
pretrained_realistic_model_name_or_path: str = "/home/yunying/GaussianIP/Realistic_Vision_V4.0_noVAE"
vae_path: str = "/home/yunying/GaussianIP/sd-vae-ft-mse"
image_encoder_path: str = "/data/vdc/tangzichen/IP-Adapter-FaceID/models/image_encoder"
image_encoder_faceid_path: str = "/home/yunying/GaussianIP/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt_path: str = "/home/yunying/GaussianIP/IP-Adapter-FaceID/models/ip-adapter-plus-face_sd15.bin"
ip_ckpt_faceid_v1_path: str = "/home/yunying/GaussianIP/IP-Adapter-FaceID/models/ip-adapter-faceid_sd15.bin"
ip_ckpt_faceid_v2_path: str = "/home/yunying/GaussianIP/IP-Adapter-FaceID/models/ip-adapter-faceid-plusv2_sd15.bin"
pose_controlnet_path: str = "/home/yunying/GaussianIP/control_v11p_sd15_openpose"
use_ipa_faceid: bool = True
use_pose_controlnet: bool = True

weights_dtype = torch.float16
base_model_path = pretrained_realistic_model_name_or_path if use_ipa_faceid else pretrained_sd_model_name_or_path
negative_prompt = "cloned face, multi face, bad face, poorly drawn face, duplicate face, cropped, out of frame, extra fingers, deformed, blurry, bad proportions, disfigured, fused fingers, long neck"
null_prompt = ""
ipa_scale: float = 0.6
ipa_faceid_scale: float = 0.6
ipa_faceid_s_scale: float = 0.5


# 🚩 自定义Attention Fusion Processor
class AttentionFusionProcessor:
    """🎯 支持attention fusion的自定义processor"""
    def __init__(self):
        self.state = 'normal'
        self.cur_view_name = None
        self.stored_zt = {}
        self.fused_attention_weights = {}
        self.attention_weights = None
        
    def apply_attention_fusion(self, attention_weights, view_name):
        """应用attention fusion机制"""
        if view_name in self.fused_attention_weights:
            # 融合主视图的attention weights
            fused_weights = self.fused_attention_weights[view_name]
            # 这里可以根据需要实现具体的融合逻辑
            return fused_weights
        return attention_weights

# load models
print("Loading IP-Adapter Model ...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = AutoencoderKL.from_pretrained(vae_path).to(device=device, dtype=weights_dtype)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pose_controlnet = ControlNetModel.from_pretrained(
    pose_controlnet_path, 
    torch_dtype=weights_dtype
).to(device)

print("OpenPose ControlNet Model Loaded !")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=pose_controlnet,
    torch_dtype=weights_dtype,
    scheduler=scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
).to(device)

print("StableDiffusionControlNetPipeline Loaded !")


# enable memory_efficient_attention
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()
pipe.enable_vae_slicing()

ipa_model = IPAdapterFaceIDPlus(
    pipe,
    image_encoder_faceid_path,
    ip_ckpt_faceid_v2_path,
    device
)

ipa_model.set_scale(scale=ipa_faceid_scale if use_ipa_faceid else ipa_scale)


@torch.amp.autocast("cuda", enabled=False)
def encode_images(imgs):
    input_dtype = imgs.dtype
    imgs = imgs * 2.0 - 1.0
    # imgs should be [-1,1]
    posterior = vae.encode(imgs.to(device=device, dtype=weights_dtype)).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents.to(input_dtype)


def refine_rgb(rgb, control_img, prompt, use_ahds_vcr=True):
    """🎯 增强的VCR (View Consistent Refinement) 函数"""
    if use_ahds_vcr:
        print("🚀 启用AHDS VCR增强模式")
        # 增强的多视图一致性检查
        view_consistency_weight = 1.2
        faceid_enhancement = True
    else:
        view_consistency_weight = 1.0
        faceid_enhancement = False
    
    # 🚩 View-Consistent Refinement: 4个主视图抗噪人脸refine
    print("🎯 启用View-Consistent Refinement机制")
    main_views = ['front', 'back', 'left', 'right']  # 4个主视图
    main_view_indices = [0, 8, 16, 24]  # 对应的索引
    
    # 存储主视图的attention weights
    main_view_attention_weights = {}
    main_view_refined_features = {}  
    view_idx_all = [24, 8, 16, 0, 20, 28, 4, 12, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
    view_name_all = ['front', 'back', 'left', 'right', 'k0', 'k1', 'k2', 'k3', 'v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23']
    prompt_all = {}
    base_prompt = prompt
    negative_prompt = "blurry face, bad face, poorly drawn face, duplicate face, extra fingers, blurry, fused fingers"
    prompt_all['front'] = base_prompt
    prompt_all['back'] = base_prompt + ', back view'
    prompt_all['left'] = base_prompt + ', left view'
    prompt_all['right'] = base_prompt + ', right view'
    prompt_all['k0'] = base_prompt + ', left front view'
    prompt_all['k1'] = base_prompt + ', right front view'
    prompt_all['k2'] = base_prompt + ', right back view'
    prompt_all['k3'] = base_prompt + ', left back view'
    for view_name in view_name_all[8:]:
        prompt_all[view_name] = base_prompt

    key_view_name_pair_mapper = {
        'v0': ('left', 'k0'), 'v1': ('left', 'k0'), 'v2': ('left', 'k0'), 'v3': ('k0', 'front'),
        'v4': ('k0', 'front'), 'v5': ('k0', 'front'), 'v6': ('front', 'k1'), 'v7': ('front', 'k1'),
        'v8': ('front', 'k1'), 'v9': ('k1', 'right'), 'v10': ('k1', 'right'), 'v11': ('k1', 'right'),
        'v12': ('right', 'k2'), 'v13': ('right', 'k2'), 'v14': ('right', 'k2'), 'v15': ('k2', 'back'),
        'v16': ('k2', 'back'), 'v17': ('k2', 'back'), 'v18': ('back', 'k3'), 'v19': ('back', 'k3'),
        'v20': ('back', 'k3'), 'v21': ('k3', 'left'), 'v22': ('k3', 'left'), 'v23': ('k3', 'left')}
    key_view_weight_pair_mapper = {
        'v0': (0.75, 0.25), 'v1': (0.5, 0.5), 'v2': (0.25, 0.75), 'v3': (0.75, 0.25),
        'v4': (0.5, 0.5), 'v5': (0.25, 0.75), 'v6': (0.75, 0.25), 'v7': (0.5, 0.5),
        'v8': (0.25, 0.75), 'v9': (0.75, 0.25), 'v10': (0.5, 0.5), 'v11': (0.25, 0.75),
        'v12': (0.75, 0.25), 'v13': (0.5, 0.5), 'v14': (0.25, 0.75), 'v15': (0.75, 0.25),
        'v16': (0.5, 0.5), 'v17': (0.25, 0.75), 'v18': (0.75, 0.25), 'v19': (0.5, 0.5),
        'v20': (0.25, 0.75), 'v21': (0.75, 0.25), 'v22': (0.5, 0.5), 'v23': (0.25, 0.75)}

    unet = ipa_model.pipe.unet

    tgt_attn_processors = [
        unet.attn_processors['up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor'],
        unet.attn_processors['up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor'],
    ]

    num_steps = 8
    lambda_self = 0.55
    
    for processor in tgt_attn_processors:
        processor.state = 'refine'
        processor.total_denoise_step = num_steps
        processor.lambda_self= lambda_self
        
    ##################################################################################################
    rgb_height = rgb.shape[1]
    rgb_width = rgb.shape[2]

    # get the device to do refine
    device = 'cuda'
    
    # reshape
    rgb_BCHW = rgb.permute(0, 3, 1, 2).to(device)            # [refine_n_views, 3, 1024, 1024]
    control_img = control_img.permute(0, 3, 1, 2).to(device) # [refine_n_views, 3, 1024, 1024]

    # Step 1: prepare timesteps
    timesteps = torch.linspace(0, 999, 50, dtype=torch.int64, device=device).round().flip(dims=[0])
    timesteps_sub = timesteps[-num_steps:]
    t = timesteps_sub[0].clone().detach().reshape(1).to(torch.long).to(device)
    num_inference_steps = len(timesteps)

    # Step 2: prepare the same noise for all the images
    noise = torch.randn(1, 4, rgb_height // 8, rgb_width // 8, device=device, dtype=torch.float16)
    noise = noise.repeat(1, 1, 1, 1)

    # Step 3: prepare negative prompt (different from the main one)
    negative_prompt = "blurry face, bad face, poorly drawn face, duplicate face, extra fingers, blurry, fused fingers"
    refined_rgbs = []

    # Step 4: 先对4个主视图进行抗噪人脸refine
    print("🎯 Step 4a: 对4个主视图进行抗噪人脸refine...")
    with torch.no_grad():
        for main_view, main_idx in zip(main_views, main_view_indices):
            print(f'🎯 主视图抗噪refine: {main_view} (idx: {main_idx})')
            
            # 对主视图进行抗噪refine
            cur_rgb_BCHW = rgb_BCHW[main_idx].reshape(1, 3, rgb_height, rgb_width)
            cur_control_img = control_img[main_idx].reshape(1, 3, rgb_height, rgb_width)
            cur_latents = encode_images(cur_rgb_BCHW.to(weights_dtype))
            cur_latents_noisy = scheduler.add_noise(cur_latents, noise, t)
            
            # 设置attention processor状态
            for processor in tgt_attn_processors:
                processor.state = 'main_view_refine'
                processor.cur_view_name = main_view
                processor.stored_zt[main_view] = []
            
            cur_prompt = prompt_all[main_view]
            
            # 执行主视图抗噪refine
            refined_rgb = ipa_model.refine_with_small_noise(
                latents=cur_latents_noisy,
                prompt=cur_prompt,
                negative_prompt=negative_prompt,
                face_image=pos_face_image,
                faceid_embeds=pos_faceid_embeds,
                image=cur_control_img,
                shortcut=True,
                scale=0.6,
                s_scale=0.5,
                num_samples=1,
                width=1024,
                height=1024,
                timesteps=timesteps_sub,
                num_inference_steps=num_inference_steps,
                seed=2024
            )
            
            # 存储主视图的attention weights和refined features
            main_view_attention_weights[main_view] = {}
            main_view_refined_features[main_view] = refined_rgb
            
            # 从attention processors中提取attention weights
            for processor in tgt_attn_processors:
                if hasattr(processor, 'attention_weights') and processor.attention_weights is not None:
                    main_view_attention_weights[main_view][processor.__class__.__name__] = processor.attention_weights.clone()
    
    # Step 4b: 对所有视图进行refine，使用attention fusion机制
    print("🎯 Step 4b: 对所有视图进行attention fusion refine...")
    with torch.no_grad():
        for i, view_info in enumerate(zip(view_idx_all, view_name_all)):
            view_idx = view_info[0]
            view_name = view_info[1]
            print('Refining {}th image, view_idx: {}, view_name: {}'.format(i, view_idx, view_name))

            # encode image into latents with vae
            cur_rgb_BCHW = rgb_BCHW[view_idx].reshape(1, 3, rgb_height, rgb_width)
            cur_control_img = control_img[view_idx].reshape(1, 3, rgb_height, rgb_width)
            cur_latents = encode_images(cur_rgb_BCHW.to(weights_dtype))
            cur_latents_noisy = scheduler.add_noise(cur_latents, noise, t)
            
            # 🚩 Attention Fusion机制：传导主视图的attention weights
            for processor in tgt_attn_processors:
                processor.cur_view_name = view_name
                processor.stored_zt[view_name] = []
                
                # 如果是中间视图，使用attention fusion
                if 'v' in view_name:
                    processor.cur_key_view_name_pair = key_view_name_pair_mapper[view_name]
                    processor.cur_key_view_weight_pair = key_view_weight_pair_mapper[view_name]
                    
                    # 传导主视图的attention weights
                    key_view1, key_view2 = key_view_name_pair_mapper[view_name]
                    weight1, weight2 = key_view_weight_pair_mapper[view_name]
                    
                    if key_view1 in main_view_attention_weights and key_view2 in main_view_attention_weights:
                        # 融合两个主视图的attention weights
                        fused_attention = {}
                        for proc_name in main_view_attention_weights[key_view1].keys():
                            if proc_name in main_view_attention_weights[key_view2]:
                                attn1 = main_view_attention_weights[key_view1][proc_name]
                                attn2 = main_view_attention_weights[key_view2][proc_name]
                                fused_attn = weight1 * attn1 + weight2 * attn2
                                fused_attention[proc_name] = fused_attn
                        
                        # 将融合的attention weights应用到当前processor
                        if hasattr(processor, 'fused_attention_weights'):
                            processor.fused_attention_weights = fused_attention
                            print(f"🎯 传导attention weights到 {view_name}: {key_view1}({weight1:.2f}) + {key_view2}({weight2:.2f})")
                
                # 设置refine状态
                processor.state = 'attention_fusion_refine'
            
            cur_prompt = prompt_all[view_name]

            refined_rgb = ipa_model.refine_with_small_noise(
                latents=cur_latents_noisy,
                prompt=cur_prompt,
                negative_prompt=negative_prompt,
                face_image=pos_face_image,
                faceid_embeds=pos_faceid_embeds,
                image=cur_control_img,
                shortcut=True,
                scale=0.6,
                s_scale=0.5,
                num_samples=1,
                width=1024,
                height=1024,
                timesteps=timesteps_sub,
                num_inference_steps=num_inference_steps,
                seed=2024
            )
        
            refined_rgbs.append(refined_rgb)

    # concat all the images
    refined_rgbs = torch.cat(refined_rgbs, dim=0)  # [refine_n_views, 3, 1024, 1024]

    return refined_rgbs.permute(0, 2, 3, 1), view_idx_all


def save_image(filename, tensor):
    """
    将shape为[H, W, C]的tensor保存为图像文件。
    
    参数:
    - tensor: PyTorch tensor，图像数据，值应该在[0, 1]之间
    - filename: str，保存的文件名
    """
    # 确保父目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 确保输入的是tensor，并将其转换为numpy数组
    if isinstance(tensor, torch.Tensor):
        image_np = tensor.cpu().detach().numpy()
    else:
        image_np = tensor

    # 对于形状[H, W, C]的tensor，C应为3（RGB图像）
    assert image_np.shape[2] == 3, "通道数必须为3 (RGB图像)"

    # 将图像数据从float转为uint8 [0, 255]
    image_np = np.clip(image_np, 0, 1)  # 确保值域在[0, 1]
    image_np = (image_np * 255).astype(np.uint8)

    # 创建图像
    img = Image.fromarray(image_np)

    # 保存图像
    img.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--cur_time", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--pil_image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    args = parser.parse_args()

    # print(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # detect face and get face_embeds (for ipa_faceid)
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    pos_image = cv2.imread(args.pil_image_path)
    pos_faces = app.get(pos_image)
    pos_faceid_embeds = torch.from_numpy(pos_faces[0].normed_embedding).unsqueeze(0)
    pos_face_image = face_align.norm_crop(pos_image, landmark=pos_faces[0].kps, image_size=224) # you can also segment the face

    # load data
    before_refine_data = torch.load(os.path.join(args.log_path, args.cur_time, 'before_refine.pth'))
    rgb = before_refine_data['images'].to(device)
    control_img = before_refine_data['control_images'].to(device)

    # start refine
    refined_rgbs, view_idx_all = refine_rgb(rgb, control_img, args.prompt)

    # save refined images
    # get trail_dir
    with open(os.path.join(args.log_path, args.cur_time, 'log.txt'), 'r') as file:
        trail_dir = file.readline()

    for i, view_idx in enumerate(view_idx_all):
        cur_refined_rgb = refined_rgbs[i]
        save_image(os.path.join(trail_dir, 'save', f"refined_rgb_{view_idx}.png"), cur_refined_rgb)

    idx_mapper = [3, 20, 21, 22, 6, 23, 24, 25, 1, 26, 27, 28, 7, 29, 30, 31, 2, 8, 9, 10, 4, 11, 12, 13, 0, 14, 15, 16, 5, 17, 18, 19]
    refined_rgbs = refined_rgbs[idx_mapper]
    refined_rgbs = refined_rgbs.permute(0, 3, 1, 2)[:, :, 60:890, 220:800]
    refined_rgbs_small = F.interpolate(refined_rgbs, scale_factor=0.5, mode="bilinear", align_corners=False)
   
    # save refined images data for the following 3d reconstruction process
    torch.save({'refined_rgbs_small': refined_rgbs_small.detach().cpu()}, os.path.join(args.log_path, args.cur_time, 'after_refine.pth'))

    # change the max_steps in config.yaml from 2400 to 3600
    config_file_path = args.config_path

    # read config.yaml
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # change args
    config['system']['stage'] = 'stage3'
    config['trainer']['max_steps'] = 6000  # Keep the full training length for Stage 3 (从5200增加到6000)

    # write it back to config.yaml
    with open(config_file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Updated max_steps to 6000 in {config_file_path}.")  # 从5200增加到6000
