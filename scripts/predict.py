"""Run inference on a single frame-folder video clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from lsvit_action.config import ExperimentConfig
from lsvit_action.constants import IMAGE_EXTENSIONS
from lsvit_action.engine.checkpoint import load_checkpoint
from lsvit_action.models import LSViTForAction
from lsvit_action.utils.visualization import denormalize, plot_clip_grid
from lsvit_action.data.transforms import VideoTransform


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Predict one frame-folder video clip.")
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Path to a directory containing extracted RGB frames.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional dataset root used to infer class names from folder names.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to print.",
    )
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Visualize the sampled frames after preprocessing.",
    )
    return parser.parse_args()


def load_clip_from_folder(
    video_dir: Path,
    num_frames: int,
    frame_stride: int,
    image_size: int,
) -> torch.Tensor:
    frame_paths = sorted(
        [path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in: {video_dir}")

    if len(frame_paths) == 1:
        indices = torch.zeros(num_frames, dtype=torch.long)
    else:
        grid = torch.linspace(
            0,
            len(frame_paths) - 1,
            steps=max(num_frames * frame_stride, num_frames),
        )
        indices = grid[::frame_stride].long()
        if indices.numel() < num_frames:
            padding = indices.new_full((num_frames - indices.numel(),), indices[-1].item())
            indices = torch.cat([indices, padding], dim=0)
        indices = indices[:num_frames]

    to_tensor = transforms.ToTensor()
    frames = []
    for index in indices:
        frame_path = frame_paths[int(index.item())]
        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            frames.append(to_tensor(image))

    video = torch.stack(frames, dim=0)
    transform = VideoTransform(image_size=image_size, is_train=False)
    return transform(video)


def infer_class_names(data_root: Path | None) -> list[str] | None:
    if data_root is None or not data_root.is_dir():
        return None

    classes = sorted([path.name for path in data_root.iterdir() if path.is_dir()])
    return classes or None


def main() -> None:
    """Main inference entry point."""
    args = parse_args()

    config = ExperimentConfig()
    device = config.train.resolve_device()

    model = LSViTForAction(config.model).to(device)
    _ = load_checkpoint(args.checkpoint, model=model, map_location=device)

    model.eval()

    video_dir = Path(args.video_dir)
    clip = load_clip_from_folder(
        video_dir=video_dir,
        num_frames=config.data.num_frames,
        frame_stride=config.data.frame_stride,
        image_size=config.data.image_size,
    )

    input_tensor = clip.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    class_names = infer_class_names(Path(args.data_root)) if args.data_root else None
    top_k = min(args.top_k, probs.numel())
    values, indices = torch.topk(probs, k=top_k)

    print("\nTop predictions:")
    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        label = class_names[index] if class_names and index < len(class_names) else str(index)
        print(f"{rank}. {label}: {value:.4f}")

    if args.show_frames:
        vis_clip = denormalize(clip)
        plot_clip_grid(vis_clip, title=f"Sampled frames from: {video_dir.name}")


if __name__ == "__main__":
    main()