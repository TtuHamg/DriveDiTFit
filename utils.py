import torch
from torchvision import transforms
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import torch.nn as nn
from diffusion import create_diffusion
from tqdm import tqdm
from PIL import Image
import os
from random import sample

from pytorch_fid import fid_score

import pandas as pd


def vae_compress(img_path: str, model_path: str):
    """
    Returns:
        numpy: size: (batch, h, w, c)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(model_path).to(device)
    img = Image.open(img_path)

    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    img_trans = trans(img).unsqueeze(0).to(device)
    img_latent = vae.encode(img_trans).latent_dist.sample().mul_(0.18215)
    img_recover = vae.decode(img_latent / 0.18215).sample

    img = (
        torch.clamp(127.5 * img_recover.detach().cpu() + 128.0, 0, 255)
        .permute(0, 2, 3, 1)
        .numpy()
        .astype(np.uint8)
    )

    return img


def generate_and_fid(
    model: nn.Module,
    model_path: list[str],
    real_dataset_path: str,
    save_dir: str,
    per_proc_sample_num: int = 1000,
    per_proc_batch_size: int = 32,
    num_domain_class=5,
    dm_config: dict = {
        "latent_size": 32,
        "num_sampling_steps": 250,
        "noise_schedule": "linear",
        "flag": 0,
    },
    device: str = "cuda",
    use_ddp: bool = False,
    load_ckpt: bool = False,
    cfg_scale: float = 4.0,
    seed: int = 0,
):
    """use model sample image and calculate fid. this function support ddp or one gpu.
    Note: Please don't wrap the model with DDP, distributed is sufficient. If the model is wrapped by DDP,
    please not pass model_path, which means it happened on training process.

    Args:
        model (nn.Module): generation model
        model_path (list[str]): vae pth, model pth | (model pth fine tune pth)
        real_dataset_path (str): the real distribution of dataset
        save_dir (str): the save location of sampled image
        per_proc_sample_num (int, optional): _description_. Defaults to 1000.
        per_proc_batch_size (int, optional): _description_. Defaults to 32.
        num_domain_class (int, optional): _description_. Defaults to 5.
        dm_config (_type_, optional): _description_. Defaults to {"latent_size": 32, "num_sampling_steps": 250}.
        device (str, optional): _description_. Defaults to "cuda".
        use_ddp (bool, optional): _description_. Defaults to False.
        load_ckpt (bool, optional): _description_. Defaults to False.
        cfg_scale (float, optional): _description_. Defaults to 4.0.
    """
    use_cfg = cfg_scale > 1.0

    with torch.no_grad():
        vae = AutoencoderKL.from_pretrained(model_path[0]).to(device)

        if load_ckpt:
            for iter_path in model_path[1:]:
                state = torch.load(iter_path, map_location="cpu")
                model.load_state_dict(
                    state["model"] if "model" in state else state, strict=False
                )
                model = model.to(device)

        model.eval()

        rank = torch.distributed.get_rank() if use_ddp else 0
        torch.manual_seed(seed)

        if not os.path.exists(save_dir):
            if rank == 0:
                os.makedirs(save_dir)

        diffusion = create_diffusion(
            timestep_respacing=str(dm_config["num_sampling_steps"]),
            noise_schedule=dm_config["noise_schedule"],
            certain_betas=dm_config["certain_betas"],
        )

        epoch = per_proc_sample_num // per_proc_batch_size
        epoch = epoch + 1 if per_proc_sample_num % per_proc_batch_size else epoch
        pbar = tqdm(range(epoch)) if rank == 0 else range(epoch)
        total = 0

        for iter in pbar:
            cl_domain = torch.randint(
                0,
                num_domain_class,
                (
                    (
                        per_proc_batch_size
                        if iter != (epoch - 1)
                        or per_proc_sample_num % per_proc_batch_size == 0
                        else per_proc_sample_num % per_proc_batch_size
                    ),
                ),
                device=device,
            )
            cl_domain_num = len(cl_domain)
            cl_latent = torch.randn(
                cl_domain_num,
                4,
                dm_config["latent_size"],
                dm_config["latent_size"],
                device=device,
            )

            if use_cfg:
                cl_latent = torch.cat([cl_latent, cl_latent], dim=0)
                cl_domain_null = torch.tensor(
                    [num_domain_class] * cl_domain_num, device=device
                )
                cl_domain = torch.cat([cl_domain, cl_domain_null], dim=0)
                model_kwargs = dict(y=cl_domain, cfg_scale=cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=cl_domain)
                sample_fn = (
                    model.module.forward if hasattr(model, "module") else model.forward
                )

            samples = diffusion.p_sample_loop(
                sample_fn,
                cl_latent.shape,
                cl_latent,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True if rank == 0 else False,
                device=device,
            )
            if use_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = vae.decode(samples / 0.18215).sample
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0.0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            for i, sample in enumerate(samples):
                if use_ddp:
                    index = i * torch.distributed.get_world_size() + rank + total
                else:
                    index = i + total

                Image.fromarray(sample).save(
                    f"{save_dir}/{index:06d}_{cl_domain[i]}.png"
                )
            total += (
                cl_domain_num * torch.distributed.get_world_size()
                if use_ddp
                else per_proc_batch_size
            )

        if not load_ckpt:
            model.train()
        if rank == 0:
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dataset_path, save_dir],
                batch_size=32,
                device="cuda",
                dims=2048,
            )
            print(f"fid: {fid_value}")

            with open(f"{save_dir}/fid_score_{fid_value}.txt", "w") as f:
                f.close()

        if use_ddp:
            torch.distributed.barrier()


def extract_ImageNet_class(sample_path: str, labelset_path: str):
    """read class infomation according classes index.
    Note: Using this API requires that the image be named 000001_XX(class index).

    Args:
        sample_path (str): the sampled image dir.
        labelset_path (str): Imagenet devkit's data path, usually is csv.

    Returns:
        dict: contain class_token, class_name and class description.
    """
    df = pd.read_csv(labelset_path)
    df.set_index("ILSVRC2012_ID", inplace=True)
    df = df[0:1000]

    df.set_index("WNID", inplace=True)
    df = df.sort_index()

    result = {}
    for img_name in sorted(os.listdir(sample_path)):
        class_id = img_name.split("_")[-1]
        class_id = int(class_id.split(".")[0])
        class_name = df.iloc[class_id]["words"]
        class_desc = df.iloc[class_id]["gloss"]
        result[class_id] = {
            "class_token": df.index[class_id],
            "class_name": class_name,
            "class_desc": class_desc,
        }
    return result


def validate_param_correct(finetune_model_path: str, origin_model_path: str):
    """check whether frozen param is consistent.

    Args:
        finetune_model_path (str): contain all param, not only fine tune part.
        origin_model_path (str): origin model path.
    """
    origin_state = torch.load(origin_model_path, map_location="cpu")
    origin_state = origin_state["model"] if "model" in origin_state else origin_state
    finetune_state = torch.load(finetune_model_path, map_location="cpu")
    finetune_model_state = (
        finetune_state["model"] if "model" in finetune_state else finetune_state
    )

    results = {}
    for name, param in origin_state.items():
        diff = (param - finetune_model_state[name]).sum()
        results[name] = diff
        # print(f"{name} difference: {diff}")

    return results


def validate_grad_correct(
    finetune_model_path_former: str, finetune_model_path_later: str
):
    """check whether fine tune param has change."""
    finetune_state_former = torch.load(finetune_model_path_former, map_location="cpu")
    finetune_model_state_former = (
        finetune_state_former["model"]
        if "model" in finetune_state_former
        else finetune_state_former
    )
    finetune_state_later = torch.load(finetune_model_path_later, map_location="cpu")
    finetune_model_state_later = (
        finetune_state_later["model"]
        if "model" in finetune_state_later
        else finetune_state_later
    )
    for name, param in finetune_model_state_later.items():
        diff = (param - finetune_model_state_former[name]).sum()
        print(f"{name} difference: {diff}")


def calculate_similarity_score(input1: torch.tensor, input2: torch.tensor):
    assert input1.shape[1] == input2.shape[1]
    assert len(input1.shape) == 2 and len(input2.shape) == 2

    result = []
    for index in range(input1.shape[0]):
        cos_sim = torch.cosine_similarity(input1[index], input2)
        result.append(cos_sim)
    return torch.stack(result, dim=0)


def select_img_by_token(token: str, dataset_path):
    for img_name in os.listdir(os.path.join(dataset_path, token)):
        print(img_name)
        img = Image.open(os.path.join(dataset_path, token, img_name))
        return img
