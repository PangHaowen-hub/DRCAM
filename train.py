import argparse
import logging
import os
import time
from pathlib import Path

import monai
import torch
import torch.nn as nn
import torchio as tio
from monai.inferers import SlidingWindowInferer
from torch.utils.data import DataLoader
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NLayerDiscriminator(nn.Module):
    """3D PatchGAN discriminator used for adversarial learning."""

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, input_tensor):
        return self.model(input_tensor)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DRCAM for full-dose CEMRI synthesis.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing train.txt and val.txt.")
    parser.add_argument("--train-list", default="train.txt", help="Training subject list under --data-dir.")
    parser.add_argument("--val-list", default="val.txt", help="Validation subject list under --data-dir.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--samples-per-volume", type=int, default=16)
    parser.add_argument("--queue-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--decay-start-epoch", type=int, default=100)
    parser.add_argument("--image-loss-weight", type=float, default=100.0)
    parser.add_argument("--lambda-loss-weight", type=float, default=100.0)
    parser.add_argument("--gan-loss-weight", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument("--resume-lambda", default=None)
    parser.add_argument("--resume-denoise", default=None)
    parser.add_argument("--resume-discriminator", default=None)
    return parser.parse_args()


def read_subject_list(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_subjects(subject_dirs, require_lambda: bool = True):
    subjects = []
    for subject_dir in subject_dirs:
        subject_path = Path(subject_dir)
        images = {
            "mask": tio.ScalarImage(subject_path / "contrast_mask.nii.gz"),
            "T1_0": tio.ScalarImage(subject_path / "T1_0_dose_Harm_norm.nii.gz"),
            "T1_25": tio.ScalarImage(subject_path / "T1_low_dose_Harm_norm.nii.gz"),
            "T1_100": tio.ScalarImage(subject_path / "T1_100_dose_norm.nii.gz"),
        }
        if require_lambda:
            images["lambda_"] = tio.ScalarImage(subject_path / "lambda.nii.gz")
            images["lambda_low"] = tio.ScalarImage(subject_path / "lambda_low.nii.gz")
        subjects.append(tio.Subject(**images))
    return subjects


def load_data(args, logger):
    data_dir = Path(args.data_dir)
    train_dirs = read_subject_list(data_dir / args.train_list)
    val_dirs = read_subject_list(data_dir / args.val_list)

    train_subjects = tio.SubjectsDataset(build_subjects(train_dirs, require_lambda=True))
    val_subjects = tio.SubjectsDataset(build_subjects(val_dirs, require_lambda=False))
    logger.info("Training set: %d subjects, Validation set: %d subjects", len(train_subjects), len(val_subjects))

    sampler = tio.data.UniformSampler(args.patch_size)
    patches_training_set = tio.Queue(
        subjects_dataset=train_subjects,
        max_length=args.queue_length,
        samples_per_volume=args.samples_per_volume,
        sampler=sampler,
        num_workers=args.num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    trainloader = DataLoader(patches_training_set, batch_size=args.batch_size)
    valloader = DataLoader(val_subjects, batch_size=1, pin_memory=True)
    return trainloader, valloader


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


def lr_lambda(epoch: int, epochs: int, decay_start_epoch: int):
    if epoch < decay_start_epoch:
        return 1.0
    decay_length = max(1, epochs - decay_start_epoch)
    return max(0.0, (epochs - epoch) / decay_length)


def train_and_val(G_L, G_D, D, trainloader, valloader, args, output_dir: Path, logger):
    criterion_gan = nn.MSELoss()
    criterion_image = nn.MSELoss()
    criterion_lambda = nn.MSELoss()

    optimizer_G_L = torch.optim.Adam(G_L.parameters(), lr=args.lr)
    optimizer_G_D = torch.optim.Adam(G_D.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, args.decay_start_epoch),
        )
        for optimizer in (optimizer_G_L, optimizer_G_D, optimizer_D)
    ]

    roi_size = (args.patch_size, args.patch_size, args.patch_size)
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=1, overlap=0.5, mode="gaussian")

    for epoch in range(args.epochs):
        G_L.train()
        G_D.train()
        D.train()
        running = {"G": 0.0, "image": 0.0, "lambda": 0.0, "gan": 0.0, "D": 0.0}

        for images in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            mask = images["mask"]["data"].to(DEVICE)
            lambda_0 = images["lambda_"]["data"].to(DEVICE)
            lambda_1 = images["lambda_low"]["data"].to(DEVICE)
            lambda_target = torch.cat([lambda_0, lambda_1], dim=1)
            image_0 = images["T1_0"]["data"].to(DEVICE)
            image_low = images["T1_25"]["data"].to(DEVICE)
            real_A = torch.cat([image_0, image_low, mask], dim=1)
            real_B = images["T1_100"]["data"].to(DEVICE)

            lambda_pred = G_L(real_A)
            initial_fd = cam_fusion(lambda_pred, image_0, image_low)
            fake_B = G_D(initial_fd)

            optimizer_G_L.zero_grad()
            optimizer_G_D.zero_grad()
            pred_fake = D(torch.cat([fake_B, real_A], dim=1))
            loss_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_image = criterion_image(fake_B, real_B)
            loss_lambda = criterion_lambda(lambda_pred, lambda_target)
            loss_G = (
                args.gan_loss_weight * loss_gan
                + args.image_loss_weight * loss_image
                + args.lambda_loss_weight * loss_lambda
            )
            loss_G.backward()
            optimizer_G_L.step()
            optimizer_G_D.step()

            optimizer_D.zero_grad()
            pred_real = D(torch.cat([real_B, real_A], dim=1))
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            pred_fake = D(torch.cat([fake_B.detach(), real_A], dim=1))
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            running["G"] += loss_G.item()
            running["image"] += loss_image.item()
            running["lambda"] += loss_lambda.item()
            running["gan"] += loss_gan.item()
            running["D"] += loss_D.item()

        for scheduler in schedulers:
            scheduler.step()

        num_batches = len(trainloader)
        logger.info(
            "Epoch %03d/%03d lr=%.6f G=%.6f image_mse=%.6f lambda_mse=%.6f gan=%.6f D=%.6f",
            epoch + 1,
            args.epochs,
            optimizer_G_L.param_groups[0]["lr"],
            running["G"] / num_batches,
            running["image"] / num_batches,
            running["lambda"] / num_batches,
            running["gan"] / num_batches,
            running["D"] / num_batches,
        )

        if (epoch + 1) % args.save_interval == 0:
            torch.save(G_L.state_dict(), output_dir / f"G_L_{epoch + 1}.pth")
            torch.save(G_D.state_dict(), output_dir / f"G_D_{epoch + 1}.pth")
            torch.save(D.state_dict(), output_dir / f"D_{epoch + 1}.pth")

        if (epoch + 1) % args.val_interval == 0:
            loss_val = 0.0
            G_L.eval()
            G_D.eval()
            with torch.no_grad():
                for images in tqdm(valloader, desc="Validation"):
                    mask = images["mask"]["data"].to(DEVICE)
                    image_0 = images["T1_0"]["data"].to(DEVICE)
                    image_low = images["T1_25"]["data"].to(DEVICE)
                    labels = images["T1_100"]["data"].to(DEVICE)
                    image_0_low = torch.cat([image_0, image_low, mask], dim=1)

                    pred_lambda = inferer(inputs=image_0_low, network=G_L)
                    pred_initial_fd = cam_fusion(pred_lambda, image_0, image_low)
                    pred_fd = inferer(inputs=pred_initial_fd, network=G_D)
                    loss_val += criterion_image(pred_fd, labels).item()
            logger.info("Epoch %03d/%03d val_image_mse=%.6f", epoch + 1, args.epochs, loss_val / len(valloader))


def configure_logger(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DRCAM")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(levelname)s %(filename)s(%(lineno)d): %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def maybe_load(model, checkpoint):
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))


if __name__ == "__main__":
    args = parse_args()
    now_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(args.experiment_root) / now_time
    logger = configure_logger(output_dir)
    logger.info("Using device: %s", DEVICE)
    logger.info("Arguments: %s", vars(args))

    net_lambda = make_swin_unetr(in_channels=3, out_channels=2, patch_size=args.patch_size).to(DEVICE)
    net_denoise = make_swin_unetr(in_channels=1, out_channels=1, patch_size=args.patch_size).to(DEVICE)
    discriminator = NLayerDiscriminator(input_nc=4, n_layers=3).to(DEVICE)

    maybe_load(net_lambda, args.resume_lambda)
    maybe_load(net_denoise, args.resume_denoise)
    maybe_load(discriminator, args.resume_discriminator)

    trainloader, valloader = load_data(args, logger)
    train_and_val(net_lambda, net_denoise, discriminator, trainloader, valloader, args, output_dir, logger)
