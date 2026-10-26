"""Checkpoint conversion only; this script never runs model inference.
Requires PyTorch and safetensors. Only trusted .pth archives should be supplied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

FORMAT = "lino-rs-v1"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open
    digest = hashlib.sha256()
    with args.input.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    sha = digest.hexdigest()
    if args.output.exists():
        with safe_open(str(args.output), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        if metadata.get("source_sha256") == sha and metadata.get("format") == FORMAT:
            print(f"Using converted checkpoint: {args.output}")
            return
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a state_dict or a checkpoint containing state_dict")
    weights = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        if key in weights:
            raise ValueError(f"Duplicate normalized weight name: {key}")
        weights[key] = value.detach().cpu().float().contiguous().clone()
    pos_key = "image_encoder.backbone.aggregator.patch_embed.pos_embed"
    if pos_key in weights and weights[pos_key].shape != (1, 1025, 384):
        pos = weights[pos_key]
        side = int((pos.shape[1] - 1) ** 0.5)
        if pos.shape[0] != 1 or pos.shape[2] != 384 or side * side != pos.shape[1] - 1:
            raise ValueError("Unsupported positional embedding shape")
        grid = pos[:, 1:].reshape(1, side, side, 384).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(grid, size=(32, 32), mode="bicubic", align_corners=False, antialias=True)
        weights[pos_key] = torch.cat((pos[:, :1], grid.permute(0, 2, 3, 1).reshape(1, 1024, 384)), dim=1).contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    try:
        save_file(weights, str(temporary), metadata={"format": FORMAT, "source_sha256": sha, "source": args.input.name})
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Converted {len(weights)} tensors to {args.output}")

if __name__ == "__main__":
    main()
