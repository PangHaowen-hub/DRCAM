# DRCAM

This repository provides a PyTorch implementation of **DRCAM: Dose Reduction via Contrast Amplification Modeling for brain contrast-enhanced magnetic resonance images**.

DRCAM synthesizes full-dose contrast-enhanced T1-weighted MRI (FD-CEMRI) from a non-contrast T1-weighted MRI (NCMRI) and a low-dose contrast-enhanced T1-weighted MRI (LD-CEMRI).

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
