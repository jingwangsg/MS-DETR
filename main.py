import torch
import argparse
from torch.nn.parallel import DistributedDataParallel
from kn_util.basic import registry, get_logger
from kn_util.config import LazyConfig
from kn_util.tools import get_command
from kn_util.basic import yapf_pformat, global_set, commit
from kn_util.nn_utils.init import match_name_keywords
from engine import train_one_epoch, evaluate, overfit_one_epoch
from kn_util.config import instantiate, serializable
from kn_util.nn_utils.amp import NativeScalerWithGradNormCount
from data.build import build_dataloader, build_datapipe
from evaluater import TrainEvaluater, ValTestEvaluater, ScalarMeter
import torch
from kn_util.basic import get_logger
from kn_util.nn_utils import CheckPointer
from kn_util.basic import save_json, save_pickle
from kn_util.distributed import initialize_ddp_from_env, get_env
from misc import dict2str
from lightning_lite.utilities.seed import seed_everything
import time
from omegaconf import OmegaConf
import os.path as osp
from pprint import pformat
import subprocess
import wandb
import os


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("cfg", type=str)
    args.add_argument("--cfg-override", "-co", nargs="+", default=[])
    args.add_argument("--resume", action="store_true", default=False)
    args.add_argument("--eval", action="store_true", default=False)
    args.add_argument("--no-multiproc", action="store_true", default=False)
    args.add_argument("--commit", action="store_true", default=False)
    args.add_argument("--wandb", action="store_true", default=False)
    args.add_argument("--exp", required=True, type=str)
    args.add_argument("--overfit", default=None, type=int)
    args.add_argument("--ddp", action="store_true", default=False)
    args.add_argument("--amp", action="store_true", default=False)

    return args.parse_args()


def main(args):
    cfg = LazyConfig.load(args.cfg)
    LazyConfig.apply_overrides(cfg, args.cfg_override)
    cfg.flags.exp = args.exp
    cfg.flags.wandb = args.wandb
    cfg.flags.ddp = args.ddp
    cfg.flags.amp = args.amp
    global_set("cfg", cfg)
    if args.commit or args.wandb:
        commit(cfg.flags.exp)

    # use_amp = cfg.flags.amp
    seed_everything(cfg.flags.seed)
    use_ddp = cfg.flags.ddp
    use_amp = cfg.flags.amp

    # initialize ddp
    if use_ddp:
        initialize_ddp_from_env()

    # resume training if possible
    if args.resume:
        ckpt.load_checkpoint(model, optimizer, lr_scheduler, mode="best")
    else:
        subprocess.run(f"rm -rf {cfg.paths.work_dir}/*", shell=True)

    if args.eval:
        ckpt.load_checkpoint(model, optimizer)

    # debug flag
    if args.no_multiproc:
        cfg.train.prefetch_factor = 2
        cfg.train.num_workers = 0

    logger = get_logger(output_dir=cfg.paths.work_dir)
    logger.info(pformat(get_command()))
    logger.info(pformat([(k, v) for k, v in os.environ.items() if k.startswith("KN")]))
    logger.info(pformat(cfg))
    OmegaConf.save(cfg, osp.join(cfg.paths.work_dir, "config.yaml"), resolve=False)

    # wandb init
    if get_env("rank") == 0 and args.wandb:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True), project="query-moment", name=cfg.flags.exp)
        wandb.run.log_code(
            cfg.paths.root_dir,
            include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
            exclude_fn=lambda path: "logs/" in path,
        )

    # build dataloader
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")
    test_loader = build_dataloader(cfg, split="test")

    # instantiate model
    model = instantiate(cfg.model)
    model = model.cuda()
    # model = torch.compile(model)
    if use_ddp:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    # build evaluater
    train_evaluater = TrainEvaluater(cfg)
    val_evaluater = ValTestEvaluater(cfg)

    # build optimizer & scheduler
    if os.getenv("KN_GROUP_LR", False):
        param_dict = [
            dict(params=[p for n, p in model.named_parameters() if not match_name_keywords(n, ["model.head"])],
                 lr=cfg.train.optimizer.lr),
        ]
        lr_group = eval(os.getenv("KN_GROUP_LR", []))
        assert len(lr_group) == cfg.model_cfg.num_layers_dec
        for i in range(cfg.model_cfg.num_layers_dec):
            param_dict.append(
                dict(params=[p for n, p in model.named_parameters() if match_name_keywords(n, [f"model.head.{i}"])],
                     lr=cfg.train.optimizer.lr * lr_group[i]))
    else:
        param_dict = model.parameters()
    optimizer = instantiate(cfg.train.optimizer, params=param_dict, _convert_="partial")
    lr_scheduler = instantiate(cfg.train.lr_scheduler, optimizer=optimizer) if hasattr(cfg.train,
                                                                                       "lr_scheduler") else None
    # build amp loss scaler & ckpt
    loss_scaler = NativeScalerWithGradNormCount() if use_amp else None
    ckpt = CheckPointer(monitor=cfg.eval.best_monitor, work_dir=cfg.paths.work_dir, mode=cfg.eval.is_best)

    logger = get_logger(output_dir=cfg.paths.work_dir)
    global_set("logger", logger)
    num_epochs = cfg.train.num_epochs

    if args.overfit:
        logger.info("==============START OVERFITTING=============")
        train_loader = build_datapipe(cfg, "train").header(1)
        train_loader_for_val = build_datapipe(cfg, "train").header(1)

        for epoch in range(10000):
            overfit_one_epoch(model=model,
                              train_loader=train_loader,
                              train_evaluater=train_evaluater,
                              val_evaluater=val_evaluater,
                              val_loader=train_loader_for_val,
                              optimizer=optimizer,
                              lr_scheduler=lr_scheduler,
                              loss_scaler=loss_scaler,
                              cur_epoch=epoch,
                              logger=logger,
                              cfg=cfg,
                              overfit_sample=args.overfit)
        return

    logger.info("==============START TRAINING=============")
    for epoch in range(num_epochs):
        train_one_epoch(model=model,
                        train_loader=train_loader,
                        train_evaluater=train_evaluater,
                        val_loader=val_loader,
                        val_evaluater=val_evaluater,
                        ckpt=ckpt,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        loss_scaler=loss_scaler,
                        cur_epoch=epoch,
                        logger=logger,
                        cfg=cfg,
                        test_loader=test_loader)

    # evaluate on best validation
    st = time.time()
    ckpt.load_checkpoint(model, optimizer, lr_scheduler, mode="best")
    metric_vals = evaluate(model, val_loader, val_evaluater, "val", cfg)
    logger.info("=========BEST VALIDATION RESULT==========")
    logger.info(f'{dict2str(metric_vals)}\t', f'eta {time.time() - st:.4f}')


if __name__ == "__main__":
    args = parse_args()
    main(args)