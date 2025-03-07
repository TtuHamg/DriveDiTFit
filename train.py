# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

from pytorch_fid import fid_score


import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

os.environ["NCCL_SOCKET_IFNAME"] = "enp0s31f6"
os.environ["NCCL_IB_DISABLE"] = "1"

from drivefit_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dataset import CustomImageFolder

import wandb

os.environ["WANDB_MODE"] = "offline"


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def extract_task_specific_parameters(model):
    task_specific_parameters = {}
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            task_specific_parameters[name] = param
    return task_specific_parameters


def requires_grad(model, flag=True, depth=28):
    if model.modulation:
        for name, param in model.named_parameters():
            if flag:
                if (
                    name.find("weight_modulation") >= 0
                    or name.find("scenario_embedding_table") >= 0
                ):
                    if name.find("blocks") >= 0:
                        param.requires_grad = int(name.split(".")[1]) < depth
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = flag


def calculate_params_num(model, rank):
    """
    Calculate the number of parameters in a model.
    """
    tune_param_num = 0
    origin_param_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            tune_param_num += torch.numel(param)
        else:
            origin_param_num += torch.numel(param)
    return tune_param_num, origin_param_num


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("nccl")

    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, device={device}."
    )

    if rank == 0:
        log = wandb.init(project="DriveDiTFit", resume=False, config=args)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        results_dir = f"{args.results_dir}/{args.dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        experiment_index = len(glob(f"{results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    logger.info(args)
    logger.info("visible device: " + os.environ["CUDA_VISIBLE_DEVICES"])

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        modulation=args.modulation,
        patch_modulation=args.patch_modulation,
        block_mlp_modulation=args.block_mlp_modulation,
        cond_mlp_modulation=args.cond_mlp_modulation,
        rank=args.rank,
        scenario_num=args.scenario_num if args.dataset_name is not None else 0,
        rope=args.rope,
        finetune_depth=args.finetune_depth,
    )

    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    if args.modulation_checkpoint is not None:
        checkpoint = torch.load(args.modulation_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    if args.embed_checkpoint is not None:
        checkpoint = torch.load(args.embed_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

    requires_grad(model, True, args.finetune_depth)

    model = DDP(model.to(device), device_ids=[device])#,find_unused_parameters=True)

    vae = AutoencoderKL.from_pretrained(args.vae_checkpoint).to(device)
    modulation_params_num, origin_params_num = calculate_params_num(model, rank)
    if rank == 0:
        log.config.update(
            {
                "modulation_params_num": modulation_params_num,
                "origin_params_num": origin_params_num,
            }
        )
    logger.info(
        f"DriveDiTFit Trainable Parameters: {modulation_params_num}\n DiT Frozen Parameters: {origin_params_num}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    if args.boxes_path is None:
        dataset = ImageFolder(args.data_path, transform=transform)
    else:
        dataset = CustomImageFolder(
            args.data_path, args.boxes_path, trans_flip=True, transform=transform
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    model.train()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    interval = args.epochs // 6

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=args.noise_schedule,
            flag=epoch // interval,
        )

        for data in loader:
            if args.boxes_path is None:
                x, y = data
                boxes_mask = None
            else:
                x, y, boxes_mask, _ = data
                boxes_mask = boxes_mask.unsqueeze(1).to(device)
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(
                model, x, t, boxes_mask, args.mask_rl, model_kwargs
            )

            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank == 0:
                    log.log(
                        {
                            "step": train_steps,
                            "loss": avg_loss,
                        },
                    )
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": extract_task_specific_parameters(model),
                        "args": args,
                        "certain_betas": diffusion.base_diffusion.betas
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % args.training_sample_steps == 0 and train_steps > 0:
                if rank == 0:
                    model.eval()
                    with torch.no_grad():
                        class_labels = [
                            i for i in range(args.scenario_num) for _ in range(4)
                        ]

                        n = len(class_labels)
                        z = torch.randn(n, 4, 32, 32, device=device)
                        y = torch.tensor(class_labels, device=device)

                        z = torch.cat([z, z], 0)
                        y_null = torch.tensor([args.scenario_num] * n, device=device)
                        y = torch.cat([y, y_null], 0)
                        model_kwargs = dict(y=y, cfg_scale=4)
                        diffusion_sample = create_diffusion(
                            str(250),
                            noise_schedule=args.noise_schedule,
                            certain_betas=diffusion.base_diffusion.betas,
                        )

                        samples = diffusion_sample.p_sample_loop(
                            model.module.forward_with_cfg,
                            z.shape,
                            z,
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            progress=True,
                            device=device,
                        )
                        samples, _ = samples.chunk(2, dim=0)
                        samples = vae.decode(samples / 0.18215).sample
                        log.log(
                            {
                                "step": train_steps,
                                "img": wandb.Image(
                                    samples.float(), caption=f"step-{train_steps:06d}"
                                ),
                                "loss": loss.item(),
                            },
                        )
                    model.train()
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--boxes-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL/2",
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=100)

    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parser.add_argument("--modulation-checkpoint", type=str, default=None)
    parser.add_argument("--embed-checkpoint", type=str, default=None)

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--training_sample_steps", type=int, default=None)
    parser.add_argument("--scenario_num", type=int, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--modulation", action="store_true")
    parser.add_argument("--patch_modulation", action="store_true")
    parser.add_argument("--block_mlp_modulation", action="store_true")
    parser.add_argument("--cond_mlp_modulation", action="store_true")

    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--finetune_depth", type=int, default=28)
    parser.add_argument("--mask_rl", type=float, default=1)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    args = parser.parse_args()
    run()
    print(args)
    main(args)
