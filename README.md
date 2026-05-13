# DRCAM

This repository provides a PyTorch/MONAI implementation of **DRCAM: Dose Reduction via Contrast Amplification Modeling for brain contrast-enhanced magnetic resonance images**.

DRCAM synthesizes full-dose contrast-enhanced T1-weighted MRI (FD-CEMRI) from a non-contrast T1-weighted MRI (NCMRI) and a low-dose contrast-enhanced T1-weighted MRI (LD-CEMRI).

## Method

DRCAM does not directly translate the input images into FD-CEMRI. Instead, it models the contrast amplification relationship among NCMRI, LD-CEMRI, and FD-CEMRI.

Let `x0` be the harmonized NCMRI, `x1` be the LD-CEMRI, and `x2` be the FD-CEMRI. The dual-stream contrast amplification modeling (CAM) module is:

```text
lambda0 = (x2 - x0) / (x1 - x0)
lambda1 = (x2 - x1) / (x1 - x0)

cam0 = lambda0 * (x1 - x0) + x0
cam1 = lambda1 * (x1 - x0) + x1
initial_fd = (cam0 + cam1) / 2
```

During training, `G_L` predicts the two amplification ratio maps `lambda0` and `lambda1`. The two CAM outputs are averaged and then passed to `G_D`, a denoising SwinUNETR, to produce the final FD-CEMRI. A 3D PatchGAN discriminator is used for adversarial learning.

The implementation follows the paper settings:

- Patch size: `96 x 96 x 96`
- Batch size: `4`
- Training epochs: `200`
- Optimizer: Adam
- Initial learning rate: `2e-4`
- Learning rate schedule: fixed for the first 100 epochs, then linearly decayed to 0
- Loss weights: `100 * image MSE + 100 * lambda MSE + 1 * adversarial MSE`
- Inference: 3D sliding-window inference with `0.5` overlap and gaussian blending

## Requirements

Create a Python environment and install the required packages:

```bash
pip install torch monai torchio SimpleITK tqdm
```

Install the PyTorch build that matches your CUDA version. Training DRCAM with 3D SwinUNETR requires a GPU with sufficient memory.

## Data Preparation

This repository assumes that preprocessing has already been completed. Each subject directory should contain:

```text
case_dir/
  T1_0_dose_Harm_norm.nii.gz      # Huber-regression harmonized NCMRI
  T1_low_dose_Harm_norm.nii.gz    # normalized LD-CEMRI
  T1_100_dose_norm.nii.gz         # normalized FD-CEMRI target
  contrast_mask.nii.gz            # potential enhanced-region mask
  lambda.nii.gz                   # ground-truth lambda0 map
  lambda_low.nii.gz               # ground-truth lambda1 map
```

Create split files such as `data/train.txt`, `data/val.txt`, and `data/test.txt`. Each line should be a subject directory:

```text
D:/dataset/DRCAM/case_001
D:/dataset/DRCAM/case_002
```

The preprocessing described in the paper includes:

- Rigidly registering NCMRI, LD-CEMRI, and FD-CEMRI to the FD-CEMRI.
- Resampling all images to isotropic 1 mm voxels.
- Removing the background with a brain mask before Huber regression.
- Applying Huber regression with threshold `epsilon = 2.5` to harmonize NCMRI to LD-CEMRI and identify potential enhanced regions.
- Normalizing with the 99.5th percentile of the LD-CEMRI maximum intensity.
- Clipping harmonized NCMRI and LD-CEMRI to 1 after normalization. FD-CEMRI is not clipped.

## Training

By default, training reads `./data/train.txt` and `./data/val.txt`:

```bash
python train.py
```

Example with explicit arguments:

```bash
python train.py \
  --data-dir ./data \
  --epochs 200 \
  --batch-size 4 \
  --lr 2e-4 \
  --experiment-root experiments
```

Checkpoints and logs are saved under `experiments/<timestamp>/`:

```text
G_L_<epoch>.pth
G_D_<epoch>.pth
D_<epoch>.pth
train.log
```

Resume training from checkpoints:

```bash
python train.py \
  --resume-lambda experiments/2026-05-13-20-00-00/G_L_100.pth \
  --resume-denoise experiments/2026-05-13-20-00-00/G_D_100.pth \
  --resume-discriminator experiments/2026-05-13-20-00-00/D_100.pth
```

## Inference

Run inference with the trained `G_L` and `G_D` checkpoints:

```bash
python test.py \
  --lambda-checkpoint experiments/2026-05-13-20-00-00/G_L_200.pth \
  --denoise-checkpoint experiments/2026-05-13-20-00-00/G_D_200.pth \
  --test-list ./data/test.txt \
  --output-dir experiments/2026-05-13-20-00-00/predictions
```

To also save intermediate CAM outputs:

```bash
python test.py \
  --lambda-checkpoint experiments/2026-05-13-20-00-00/G_L_200.pth \
  --denoise-checkpoint experiments/2026-05-13-20-00-00/G_D_200.pth \
  --test-list ./data/test.txt \
  --output-dir experiments/2026-05-13-20-00-00/predictions \
  --save-intermediate
```

Intermediate outputs include:

```text
<case>_lambda_0.nii.gz
<case>_lambda_1.nii.gz
<case>_cam_initial.nii.gz
```

## Citation

If you use this code or find this repository helpful, please cite:

```bibtex
@article{pang2026drcam,
  title={DRCAM: Dose Reduction via Contrast Amplification Modeling for brain contrast-enhanced magnetic resonance images},
  author={Pang, Haowen and Xu, Siyao and Zhang, Xinru and An, Fengping and Zhang, Xiaofeng and Wang, Ying and Liu, Fujun and Yan, Tianyi and Marais, Patrick and Qian, Yinfeng and others},
  journal={Biomedical Signal Processing and Control},
  volume={113},
  pages={109074},
  year={2026},
  publisher={Elsevier}
}
```
