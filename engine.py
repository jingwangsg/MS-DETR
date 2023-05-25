from tqdm import tqdm
from evaluater import TrainEvaluater, ValTestEvaluater, ScalarMeter, Evaluater
import torch
from kn_util.basic import add_prefix_dict, global_get, global_set, save_json
from kn_util.data import collection_to_device
from kn_util.distributed import initialize_ddp_from_env, is_ddp_initialized_and_available, get_env
from misc import dict2str
import time
import torch.distributed as dist
from tqdm import tqdm
from torch import nn
import contextlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import copy
import os
import numpy as np
import os.path as osp
from kn_util.basic import global_get as G
from kn_util.basic import global_set as GS
from kn_util.basic import registry
import wandb


@registry.register_function("post_process_msat")
def post_process_msat(out):
    ret = []
    topk = 5
    for idx in range(len(out["video_id"])):
        video_id = out["video_id"][idx]
        text = out["text"][idx]
        boxxes = out["boxxes"][idx].detach().numpy()
        scores = out["scores"][idx].sigmoid().detach().numpy()

        gt = out["gt"][idx].detach().numpy().tolist()
        topk_indices = np.argsort(scores)[-1:-(topk + 1):-1]
        topk_boxxes = boxxes[topk_indices].tolist()
        topk_scores = scores[topk_indices].tolist()

        cur_item = dict(video_id=video_id,
                        text=text,
                        topk_boxxes=topk_boxxes,
                        topk_scores=topk_scores,
                        topk_indices=topk_indices.tolist(),
                        gt=gt)

        ret.append(cur_item)
    return ret


@registry.register_function("post_process_mst_detr")
def post_process_mst_detr(out):
    ret = []
    topk = 5
    for idx in range(len(out["video_id"])):
        video_id = out["video_id"][idx]
        text = out["text"][idx]
        boxxes = out["boxxes"][idx].detach().numpy()
        scores = out["scores"][idx].sigmoid().detach().numpy()
        stage_logits = out["stage_logits"][idx].detach().numpy()
        reference_centers = out["reference_centers"][idx].detach().numpy()

        gt = out["gt"][idx].detach().numpy().tolist()
        topk_indices = np.argsort(scores)[-1:-(topk + 1):-1]
        topk_boxxes = boxxes[topk_indices].tolist()
        topk_scores = scores[topk_indices].tolist()

        reference_centers = reference_centers.tolist()[:topk]

        cur_item = dict(video_id=video_id,
                        text=text,
                        topk_boxxes=topk_boxxes,
                        topk_scores=topk_scores,
                        topk_indices=topk_indices.tolist(),
                        reference_centers=reference_centers,
                        gt=gt)

        ret.append(cur_item)
    return ret


# class Runner:
#     def __init__(self) -> None:
#         pass

#     def train_one_epoch(model, dataloader):
#         for batch in dataloader():


@torch.no_grad()
def evaluate(model, loader, evaluater, cfg):
    # evaluater = ValTestEvaluater(cfg, domain=domain)
    model.eval()

    processed_out = []

    for idx, batch in enumerate(tqdm(loader, desc="evaluating")):
        batch = collection_to_device(batch, "cuda")
        out = model(**batch, mode="test")
        out.update(batch)
        out = collection_to_device(out, "cpu")
        evaluater.update_all(out)
        processed_out.extend(registry.call_function(cfg.data.post_process, out=out))

    if is_ddp_initialized_and_available():
        torch.cuda.synchronize()

    metric_vals = evaluater.compute_all()

    return metric_vals, processed_out


def train_one_epoch(
    model,
    train_loader,
    train_evaluater: Evaluater,
    val_loader: Evaluater,
    val_evaluater,
    ckpt,
    optimizer,
    lr_scheduler,
    loss_scaler,
    cur_epoch,
    logger,
    cfg,
    test_loader=None,
):
    use_amp = cfg.flags.amp
    use_wandb = cfg.flags.wandb
    use_ddp = is_ddp_initialized_and_available()

    num_batches_train = train_loader.num_batches // get_env("world_size")
    val_interval = int(cfg.train.val_interval * num_batches_train)
    print_interval = int(cfg.train.print_interval * num_batches_train)
    validate_on_test = test_loader is not None

    global_step = global_get("global_step", 0)
    global_set("cur_epoch", cur_epoch)

    for idx, batch in enumerate(train_loader):
        st = time.time()
        model.train()
        # update_grad = ((idx + 1) % cfg.train.accum_grad_steps == 0)
        update_grad = True
        use_print = ((idx + 1) % print_interval == 0)
        use_validate = ((idx + 1) % val_interval == 0)

        # forward
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            batch = collection_to_device(batch, "cuda")
            losses = model(**batch, mode="train")
            loss = losses["loss"]

        # backward & optimize
        if use_amp:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=cfg.train.clip_grad, parameters=model.parameters())
            grad_norm = grad_norm.item()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_grad).item()
            optimizer.step()

        if update_grad:
            optimizer.zero_grad()

        # update training metrics
        train_evaluater.update_scalar("grad_norm", grad_norm)
        train_evaluater.update_scalar("batch_eta", time.time() - st)
        # losses = collection_to_device(losses, "cpu")
        # print(f"Rank#{get_env("rank")}", EC(batch, ))
        train_evaluater.update_all(losses)

        if use_validate:
            if False:
                st = time.time()
                metric_vals = evaluate(model, val_loader, val_evaluater, cfg)

                # only display all losses for training
                ks = list(metric_vals.keys())
                for nm in ks:
                    if nm.endswith("loss") and nm != "loss":
                        metric_vals.pop(nm)

                logger.info(f'Evaluated Val\t'
                            f'{dict2str(metric_vals)}\t'
                            f'eta {time.time() - st:.4f}s')
                if use_wandb:
                    wandb.log(add_prefix_dict(metric_vals, "val/"), step=global_step)

            if validate_on_test:
                st = time.time()
                metric_vals, prediction = evaluate(model, test_loader, val_evaluater, cfg)

                # only display all losses for training
                ks = list(metric_vals.keys())
                for nm in ks:
                    if nm.endswith("loss") and nm != "loss":
                        metric_vals.pop(nm)

                logger.info(f'Evaluated Test\t'
                            f'{dict2str(metric_vals)}\t'
                            f'eta {time.time() - st:.4f}s')
                if use_wandb:
                    wandb.log(add_prefix_dict(metric_vals, "test/"), step=global_step)

            # reduce lr on plateau
            # if isinstance(lr_scheduler, ReduceLROnPlateau):
            #     lr_scheduler.step(metric_vals["loss"])

            # save checkpoint
            save_model = model if not use_ddp else model.module
            better = ckpt.save_checkpoint(model=save_model,
                                          optimizer=optimizer,
                                          num_epochs=cur_epoch,
                                          metric_vals=metric_vals,
                                          lr_scheduler=lr_scheduler,
                                          loss_scaler=loss_scaler)
            if better:
                save_json(prediction, osp.join(cfg.paths.work_dir, "predictions-best.json"))
                if wandb.run:
                    for k, v in metric_vals.items():
                        wandb.run.summary[k] = v
            save_json(prediction, osp.join(cfg.paths.work_dir, "predictions-latest.json"))

        if use_print and get_env("rank") == 0:
            train_metrics = train_evaluater.compute_all()

            if os.getenv("KN_SCHEDULE_TRAIN", False) and isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(train_metrics["bd_2_gr0_loss"])

            lr = optimizer.param_groups[0]['lr']
            if use_wandb:
                wandb.log(add_prefix_dict(train_metrics, "train/"), step=global_step)
                wandb.log({"train/lr": lr})
            mem = torch.cuda.max_memory_allocated()
            # metric_str = dict2str(train_metrics, ordered_keys=["train/loss", "train/grad_norm", "train/batch_eta"])
            other_loss = {k: v for k, v in train_metrics.items() if k.endswith("_loss")}
            logger.info(f"Train Epoch{cur_epoch:4d} [{idx+1}/{num_batches_train}]\t"
                        f"loss {train_metrics['loss']:.6f}\t"
                        f"{dict2str(other_loss)}\t"
                        f"grad_norm {train_metrics['grad_norm']:.4f}\t"
                        f"lr {lr}\t"
                        f"batch_eta {train_metrics['batch_eta']:.4f}s\t"
                        f"mem {mem / (1024 ** 2):.0f}MB\t")

        global_step += 1

        global_set("global_step", global_step)


def overfit_one_epoch(model, train_loader, train_evaluater: Evaluater, val_loader, val_evaluater, optimizer,
                      lr_scheduler, loss_scaler, cur_epoch, logger, cfg, overfit_sample):
    use_amp = cfg.flags.amp
    use_wandb = cfg.flags.wandb

    global_step = global_get("global_step", 0)
    global_set("cur_epoch", cur_epoch)

    target_idx = [161]

    for idx, batch in enumerate(train_loader):
        st = time.time()
        model.train()
        # update_grad = ((idx + 1) % cfg.train.accum_grad_steps == 0)
        update_grad = True
        use_print = True
        use_validate = True

        # forward
        with torch.autocast("cuda", enabled=use_amp):
            batch = collection_to_device(batch, "cuda")
            losses = model(**batch, mode="train")
            loss = losses["loss"]

        # backward & optimize
        if use_amp:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=cfg.train.clip_grad, parameters=model.parameters())
            grad_norm = grad_norm.item()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_grad).item()

            for i in range(cfg.model_cfg.num_layers_dec):
                print(
                    f"decoder head grad #{i}: {nn.utils.clip_grad_norm_(model.head.layers[i].parameters(), max_norm=100)}"
                )
            optimizer.step()

        if update_grad:
            optimizer.zero_grad()

        # update training metrics
        train_evaluater.update_scalar("grad_norm", grad_norm)
        train_evaluater.update_scalar("batch_eta", time.time() - st)
        # losses = collection_to_device(losses, "cpu")
        # print(f"Rank#{get_env("rank")}", EC(batch, ))
        train_evaluater.update_all(losses)

        # if use_validate:
        #     st = time.time()
        #     metric_vals, prediction = evaluate(model, val_loader, val_evaluater, cfg)

        #     # only display all losses for training
        #     ks = list(metric_vals.keys())
        #     for nm in ks:
        #         if nm.endswith("loss") and nm != "loss":
        #             metric_vals.pop(nm)

        #     logger.info(f'Evaluated Test\t'
        #                 f'{dict2str(metric_vals)}\t'
        #                 f'eta {time.time() - st:.4f}s')

        #     # reduce lr on plateau
        #     if isinstance(lr_scheduler, ReduceLROnPlateau):
        #         lr_scheduler.step(metric_vals["loss"])

        if use_print and get_env("rank") == 0:
            train_metrics = train_evaluater.compute_all()
            mem = torch.cuda.max_memory_allocated()
            # metric_str = dict2str(train_metrics, ordered_keys=["train/loss", "train/grad_norm", "train/batch_eta"])
            other_loss = {k: v for k, v in train_metrics.items() if k.endswith("_loss")}
            logger.info(f"Train Epoch{cur_epoch:4d} \t"
                        f"loss {train_metrics['loss']:.6f}\t"
                        f"{dict2str(other_loss)}\t"
                        f"grad_norm {train_metrics['grad_norm']:.4f}\t"
                        f"batch_eta {train_metrics['batch_eta']:.4f}s\t"
                        f"mem {mem / (1024 ** 2):.0f}MB\t")

        global_step += 1

        global_set("global_step", global_step)
