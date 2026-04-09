from __future__ import annotations

import argparse
from pathlib import Path

from lsvit_action.config import ExperimentConfig
from lsvit_action.data import build_dataloaders
from lsvit_action.engine import build_optimizer, fit, load_vit_checkpoint
from lsvit_action.models import LSViTForAction
from lsvit_action.utils.io import save_json
from lsvit_action.utils.logging_utils import setup_logger
from lsvit_action.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSViT on HMDB51 frame folders.")

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig()
    config.prepare()

    if args.data_root is not None:
        config.data.data_root = Path(args.data_root)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name

    logger = setup_logger(
        name="lsvit_train",
        log_file=config.paths.logs_dir / f"{config.experiment_name}.log",
    )

    set_seed(config.data.seed)

    logger.info("Preparing dataloaders...")
    train_loader, val_loader = build_dataloaders(config.data)
    logger.info(
        "Loaded datasets | train_clips=%d | val_clips=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    logger.info("Building model...")
    model = LSViTForAction(config.model)

    if not args.skip_pretrained:
        logger.info(
            "Loading pretrained backbone weights from timm model: %s",
            config.train.pretrained_name,
        )
        missing_keys, unexpected_keys = load_vit_checkpoint(
            backbone=model.backbone,
            pretrained_name=config.train.pretrained_name,
            weights_dir=config.paths.weights_dir,
        )
        logger.info("Backbone weights loaded.")
        if missing_keys:
            logger.info("Missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.info("Unexpected keys: %s", unexpected_keys)

    optimizer = build_optimizer(model, config.train)

    logger.info("Starting training...")
    trained_model, history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
    )

    _ = trained_model

    history_path = config.paths.outputs_dir / f"{config.experiment_name}_history.json"
    config_path = config.paths.outputs_dir / f"{config.experiment_name}_config.json"

    save_json(history.to_dict(), history_path)
    save_json(config.to_dict(), config_path)

    logger.info("Training completed.")
    logger.info("History saved to: %s", history_path)
    logger.info("Config saved to: %s", config_path)


if __name__ == "__main__":
    main()