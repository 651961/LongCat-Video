import argparse
import datetime
import os

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.distributed as dist
from diffusers.utils import load_image
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from torchvision.io import write_video
from transformers import AutoTokenizer, UMT5EncoderModel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_stage1(args):
    df = pd.read_csv(args.csv_path)

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # Prepare distributed environment
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    # Initialize context parallel
    init_context_parallel(
        context_parallel_size=args.context_parallel_size, global_rank=global_rank, world_size=num_processes
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # Load models
    checkpoint_dir = args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
    )

    # Load LoRA for distill
    cfg_step_lora_path = os.path.join(checkpoint_dir, "lora/cfg_step_lora.safetensors")
    dit.load_lora(cfg_step_lora_path, "cfg_step_lora")
    dit.enable_loras(["cfg_step_lora"])

    if args.enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    os.makedirs(args.stage1_output_dir, exist_ok=True)

    # Set seed
    seed = args.seed + global_rank
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    for idx, row in df.iterrows():
        image_path = row["file_name"]
        prompt = row["text"]
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load image
        image = load_image(image_path)

        if local_rank == 0:
            print(f"\n{'=' * 60}")
            print(f"[{idx + 1}/{len(df)}]: {os.path.basename(row['file_name'])}")
            print(f"{'=' * 60}")

        # Generate 480p distill
        output_distill = pipe.generate_i2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution="480p",
            num_frames=93,
            num_inference_steps=16,
            use_distill=True,
            guidance_scale=1.0,
            generator=generator,
        )[0]

        # Save intermediate results (numpy array)
        if local_rank == 0:
            save_path = os.path.join(args.stage1_output_dir, f"{image_name}.mp4")
            output_frames = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
            output_frames = [PIL.Image.fromarray(img) for img in output_frames]
            output_tensor = torch.from_numpy(np.array(output_frames))
            write_video(save_path, output_tensor, fps=args.fps, video_codec="libx264", options={"crf": str(args.crf)})

        del output_distill
        torch_gc()

    if local_rank == 0:
        print(f"Saved npy results to {args.stage1_output_dir}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--stage1_output_dir", type=str, required=True, help="Directory to save stage1 results")
    parser.add_argument("--context_parallel_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--crf", type=int, default=10)
    parser.add_argument("--enable_compile", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    generate_stage1(args)
