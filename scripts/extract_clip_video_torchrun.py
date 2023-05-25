import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from kn_util.data import VisionCLIPWrapper
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import datetime
from torch.utils.data import DistributedSampler, DataLoader
import argparse
import os
import sys
import h5py
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from tqdm import tqdm
from kn_util.basic.file import load_json, load_csv, LargeHDF5Cache
import subprocess
import glob
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from kn_util.distributed import get_available_port
from kn_util.basic import get_logger
import numpy as np
from pytorch_lightning.loggers import CSVLogger

data_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw"


def load_data(image_root):
    video_paths = glob.glob(osp.join(image_root, "*"))
    ret_dataset = []
    for video_path in video_paths:
        video_id = osp.basename(video_path)
        frame_paths = glob.glob(osp.join(video_path, "*.png"))
        frame_paths = sorted(frame_paths)
        cur_elem = dict(video_id=video_id, frame_paths=frame_paths)
        ret_dataset += [cur_elem]
    return ret_dataset


def distributed_inference(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"[{local_rank}] enter inference")

    dataset_dir = osp.join(data_dir, args.dataset)
    pretrained = osp.basename(args.pretrained)
    hdf5_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")

    pretrained = osp.basename(args.pretrained)
    image_root = osp.join(data_dir, args.dataset, "images")
    dataset_dir = osp.join(data_dir, args.dataset)
    hdf5_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")
    rank = local_rank
    hdf5_cache = LargeHDF5Cache(hdf5_file, compression="gzip", compression_opts=9)
    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)

    log = get_logger(output_dir="./log/" + args.dataset)
    model = CLIPVisionModel.from_pretrained(args.pretrained)
    model = model.cuda()
    extractor = CLIPFeatureExtractor.from_pretrained(args.pretrained)
    model_wrapper = VisionCLIPWrapper(model=model, extractor=extractor, batch_size=64, use_cuda=True)

    log.info(f"[{local_rank}] loading data...")
    dataset = load_data(image_root)

    dataloader = DataLoader(dataset,
                            sampler=DistributedSampler(dataset, num_replicas=world_size, rank=rank),
                            batch_size=1,
                            collate_fn=lambda x: x)

    if rank == 0:
        dataloader = tqdm(dataloader)

    with torch.no_grad():
        for e in dataloader:
            e = e[0]
            video_id = e["video_id"]
            if hdf5_cache.key_exists(video_id):
                log.info(f"{video_id} exists, skip")
            else:
                frame_paths = e["frame_paths"]
                log.info(f"{video_id} inference")
                outputs = model_wrapper(frame_paths)
                outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
                save_dict = {video_id: outputs}
                hdf5_cache.cache_save(save_dict)
                log.info(f"{video_id} saved")

            dist.barrier()
    print("===============inference done===============")

    if rank == 0:
        hdf5_cache.final_save()


def inference(args):
    pretrained = osp.basename(args.pretrained)
    image_root = osp.join(data_dir, args.dataset, "images")
    dataset_dir = osp.join(data_dir, args.dataset)
    hdf5_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")
    hdf5_cache = LargeHDF5Cache(hdf5_file, compression="gzip", compression_opts=9)

    torch.manual_seed(0)

    model = CLIPVisionModel.from_pretrained(args.pretrained)
    model = model.cuda()
    extractor = CLIPFeatureExtractor.from_pretrained(args.pretrained)
    model_wrapper = VisionCLIPWrapper(model=model, extractor=extractor, batch_size=64)

    dataset = load_data(image_root)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    dataloader = tqdm(dataloader)

    with torch.no_grad():
        for e in dataloader:
            e = e[0]
            video_id = e["video_id"]
            if hdf5_cache.key_exists(video_id):
                continue
            frame_paths = e["frame_paths"]
            outputs = model_wrapper(frame_paths)
            outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            save_dict = {video_id: outputs}
            hdf5_cache.cache_save(save_dict)

    hdf5_cache.final_save()


def main():

    args = argparse.ArgumentParser()
    args.add_argument("dataset", choices=["tacos", "charades", "activitynet"])
    # args.add_argument("--gpu", default="0", type=str)
    args.add_argument("--ddp", action="store_true", default=False)
    args.add_argument("--pretrained", default="openai/clip-vit-large-patch14-336", type=str)
    args = args.parse_args()

    if args.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # port = int(os.environ["MASTER_PORT"])
        # print(port)
        rank = local_rank
        # dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                world_size=world_size,
                                rank=rank,
                                timeout=datetime.timedelta(seconds=5400))
        print(f"local_rank {local_rank} | world_size {world_size}")

    dataset_dir = osp.join(data_dir, args.dataset)
    pretrained = osp.basename(args.pretrained)
    hdf5_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")

    if dist.get_rank() == 0:
        subprocess.run(f"rm -rf {hdf5_file}", shell=True)
        subprocess.run(f"rm -rf ./log/{args.dataset}", shell=True)
        tmp_dir = osp.join(dataset_dir, hdf5_file + ".tmp")
        os.makedirs(osp.dirname(hdf5_file), exist_ok=True)
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()  # make sure folders get created

    distributed_inference(args)


if __name__ == "__main__":
    main()