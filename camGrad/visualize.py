from nanodet.model.arch import build_model
import argparse
import pytorch_lightning as pl
import torch
import warnings

from pytorch_lightning.callbacks import TQDMProgressBar

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.taskDA import TrainingTaskDA
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    load_model_weight,
    mkdir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args

def main(args):
    load_config(cfg, args.config)

    model = build_model(cfg.model)
    logger = NanoDetLightningLogger(cfg.save_dir)

    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
    if "pytorch-lightning_version" not in ckpt:
        warnings.warn(
            "Warning! Old .pth checkpoint is deprecated. "
            "Convert the checkpoint with tools/convert_old_checkpoint.py "
        )
        ckpt = convert_old_model(ckpt)
    load_model_weight(model, ckpt, logger)
    logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    print(model.backbone)


if __name__ == "__main__":
    args = parse_args()
    main(args)
