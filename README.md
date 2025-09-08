#  SpecularIP: ASG-Driven Specular Modeling for Real-Time 3D Gaussian Avatars

[Yunying Wang](https://github.com/Rianna3)<sup>1</sup>.  
<sup>1</sup>Durham University;


Text-/image-guided single-image 3D human generation has benefited from efficient 3D representations and 2D-to-3D lifting via Score Distillation Sampling (SDS). Yet standard 3D Gaussian Splatting (3DGS) models view dependence with low-order harmonics, which underrepresents sharp, anisotropic specularities—producing highlights that drift across views and degrade identity fidelity. Building on 3DGS’s real-time, visibility-aware rasterization, we target view-consistent specular appearance as a first-class objective. We introduce a two-stage framework that augments 3DGS with explicit specular modeling. In stage 1: Specular-Enhanced Identity Distillation (SEID), we initialize Gaussians on a canonical human prior and optimize them with SDS under cross-modal conditioning (text, identity, and pose), interleaving densify/prune to concentrate capacity. This stage rapidly establishes geometry and identity while bootstrapping view-dependent appearance. In Stage 2: Specular-Consistency Refinement (SCR), we attach an ASG-parameterized Specular Network and a lightweight mutual-attention fusion with the diffuse branch to enforce cross-view highlight coherence; refined multi-view RGB targets then supervise photometric and perceptual consistency, updating both the Gaussians and the specular head. Extensive experiments (portraits with satin, leather, and metallic accessories) show crisper and more stable highlights, improved facial fidelity, and preserved real-time rendering versus a 3DGS/SH baseline.

## Installation
```
# clone the github repo
git clone https://github.com/Rianna3/SpecularIP.git
cd SpecularIP

# install torch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install -r requirements.txt

# install a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
```

## Text Prompts Gallery
The text prompts that are used for qualitative/ablation visual results demonstration are included in `gallery_text_prompts.txt`.

## Image Prompts Gallery
The image prompts that are used for qualitative/ablation visual results demonstration are included in `gallery_image_prompts.txt`.

## Pretrained Models
Please prepare below pre-trained models before the training process:

* **SMPL-X**: Download SMPL-X model from https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip, unzip and place smpl under `/path/to/smplx`.

* **stabilityai/sd-vae-ft-mse**: Download VAE model from https://huggingface.co/stabilityai/sd-vae-ft-mse.

* **SG161222/Realistic_Vision_V4.0_noVAE**: Download diffusion base model from https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE.

* **lllyasviel/control_v11p_sd15_openpose**: Download ControlNet model from https://huggingface.co/lllyasviel/control_v11p_sd15_openpose.

* **laion/CLIP-ViT-H-14-laion2B-s32B-b79K**: Download image encoder model from https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K.

* **ip-adapter-faceid-plusv2_sd15.bin**: Download ip-adapter model from https://huggingface.co/h94/IP-Adapter-FaceID.

After the above models are downloaded, please specify their paths in `configs/exp.yaml` and `threestudio/models/guidance/refine.py` properly.

## Train
First, specify `TEXT_PROMPT` and `IMAGE_PROMPT` in run.sh.
Then run:
```bash
bash run.sh
```


## Acknowledgement
This work is inspired by and builds upon numerous outstanding research efforts and open-source projects, including [Specular-Gaussians](https://github.com/ingra14m/Specular-Gaussians.git), [GaussianIP](https://github.com/silence-tang/GaussianIP.git), [Threestudio](https://github.com/threestudio-project/threestudio), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization), [HumanGaussian](https://github.com/alvinliu0/HumanGaussian/), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/). We are deeply grateful to all the authors for their contributions and for sharing their work openly!

## Notes
We train on the resolution of 1024x1024 with a batch size of 4. The whole optimization process takes around 1h 20 minutes on a single a single Quadro RTX 5000(16GB)GPU.
