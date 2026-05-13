import argparse
from pathlib import Path

import monai
import SimpleITK as sitk
import torch
import torchio as tio
from monai.inferers import SlidingWindowInferer
from torch.utils.data import DataLoader
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DRCAM inference on a test set.")
    parser.add_argument("--lambda-checkpoint", required=True, help="Path to G_L checkpoint.")
    parser.add_argument("--denoise-checkpoint", required=True, help="Path to G_D checkpoint.")
    parser.add_argument("--test-list", default="./data/test.txt", help="Text file with one subject directory per line.")
    parser.add_argument("--data-root", default=None, help="Optional root prepended to relative subject paths.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated NIfTI outputs.")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--save-intermediate", action="store_true", help="Save lambda maps and the initial CAM image.")
    return parser.parse_args()


def make_swin_unetr(in_channels: int, out_channels: int, patch_size: int):
    return monai.networks.nets.SwinUNETR(
        img_size=(patch_size, patch_size, patch_size),
        in_channels=in_channels,
        out_channels=out_channels,
        depths=(2, 4, 2, 2),
    )


def cam_fusion(lambda_pred, image_0, image_low):
    lambda_0 = lambda_pred[:, 0:1]
    lambda_1 = lambda_pred[:, 1:2]
    difference = image_low - image_0
    cam_from_nc = lambda_0 * difference + image_0
    cam_from_ld = lambda_1 * difference + image_low
    return 0.5 * (cam_from_nc + cam_from_ld)


def read_subject_list(path: Path, data_root):
    with path.open("r", encoding="utf-8") as f:
        subjects = [line.strip() for line in f if line.strip()]
    if data_root is None:
        return [Path(subject) for subject in subjects]
    root = Path(data_root)
    return [Path(subject) if Path(subject).is_absolute() else root / subject for subject in subjects]


def build_subjects(subject_dirs):
    subjects = []
    for subject_dir in subject_dirs:
        subject_path = Path(subject_dir)
        subjects.append(
            tio.Subject(
                mask=tio.ScalarImage(subject_path / "contrast_mask.nii.gz"),
                T1_0=tio.ScalarImage(subject_path / "T1_0_dose_Harm_norm.nii.gz"),
                T1_25=tio.ScalarImage(subject_path / "T1_low_dose_Harm_norm.nii.gz"),
            )
        )
    return subjects


def to_sitk_image(tensor, reference_path):
    array = tensor.detach().cpu().squeeze().permute(2, 1, 0).numpy()
    reference = sitk.ReadImage(str(reference_path))
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    return image


def save_volume(tensor, reference_path, output_path):
    image = to_sitk_image(tensor, reference_path)
    sitk.WriteImage(image, str(output_path))


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.denoise_checkpoint).with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_lambda = make_swin_unetr(in_channels=3, out_channels=2, patch_size=args.patch_size).to(DEVICE)
    model_denoise = make_swin_unetr(in_channels=1, out_channels=1, patch_size=args.patch_size).to(DEVICE)
    model_lambda.load_state_dict(torch.load(args.lambda_checkpoint, map_location=DEVICE))
    model_denoise.load_state_dict(torch.load(args.denoise_checkpoint, map_location=DEVICE))
    model_lambda.eval()
    model_denoise.eval()

    subject_dirs = read_subject_list(Path(args.test_list), args.data_root)
    test_subjects = tio.SubjectsDataset(build_subjects(subject_dirs))
    print("Test set:", len(test_subjects), "subjects")
    test_dataloader = DataLoader(test_subjects, batch_size=1, pin_memory=True)

    roi_size = (args.patch_size, args.patch_size, args.patch_size)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.5, mode="gaussian")

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            mask = batch["mask"]["data"].to(DEVICE)
            image_0 = batch["T1_0"]["data"].to(DEVICE)
            image_low = batch["T1_25"]["data"].to(DEVICE)
            image_0_low = torch.cat([image_0, image_low, mask], dim=1)

            pred_lambda = inferer(inputs=image_0_low, network=model_lambda)
            pred_initial_fd = cam_fusion(pred_lambda, image_0, image_low)
            pred_fd = inferer(inputs=pred_initial_fd, network=model_denoise)

            reference_path = Path(batch["T1_0"]["path"][0])
            subject_name = reference_path.parent.name
            save_volume(pred_fd, reference_path, output_dir / f"{subject_name}.nii.gz")

            if args.save_intermediate:
                save_volume(pred_lambda[:, 0:1], reference_path, output_dir / f"{subject_name}_lambda_0.nii.gz")
                save_volume(pred_lambda[:, 1:2], reference_path, output_dir / f"{subject_name}_lambda_1.nii.gz")
                save_volume(pred_initial_fd, reference_path, output_dir / f"{subject_name}_cam_initial.nii.gz")
