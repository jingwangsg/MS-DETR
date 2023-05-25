import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from kn_util.data import VisionCLIPWrapper
from transformers import CLIPFeatureExtractor, CLIPVisionModel

import sys
import os.path as osp
from torch.utils.data import DistributedSampler, DataLoader

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
import os.path as osp
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


def distributed_inference(local_rank, args):
    
    pretrained = osp.basename(args.pretrained)
    image_root = osp.join(data_dir, args.dataset, "images")
    dataset_dir = osp.join(data_dir, args.dataset)
    hdf5_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")
    rank = args.nr * args.num_gpu + local_rank
    hdf5_cache = LargeHDF5Cache(hdf5_file, compression="gzip", compression_opts=9)

    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    torch.cuda.set_device(local_rank)

    log = get_logger(output_dir="./log/" + args.dataset)
    model = CLIPVisionModel.from_pretrained(args.pretrained)
    model = model.cuda()
    log.info(f"local_rank {local_rank} launched")
    extractor = CLIPFeatureExtractor.from_pretrained(args.pretrained)
    model_wrapper = VisionCLIPWrapper(model=model, extractor=extractor, batch_size=64)

    dataset = load_data(image_root)

    dataloader = DataLoader(dataset,
                            sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=rank),
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
                continue
            frame_paths = e["frame_paths"]
            log.info(f"{video_id} inference")
            outputs = model_wrapper(frame_paths)
            outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            save_dict = {video_id: outputs}
            hdf5_cache.cache_save(save_dict)
            log.info(f"{video_id} saved")

    if dist.is_initialized():
        dist.barrier()

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
    args.add_argument("--gpu", default="0", type=str)
    args.add_argument("--pretrained", default="openai/clip-vit-large-patch14-336", type=str)
    args.add_argument("-n",
                      "--nodes",
                      default=1,
                      type=int,
                      metavar="N",
                      help="number of data loading workers (default: 4)")
    args.add_argument("-g", "--num_gpu", default=1, type=int, help="number of gpus per node")
    args.add_argument("-nr", "--nr", default=0, type=int, help="ranking within the nodes")
    args = args.parse_args()
    args.num_gpu = len(args.gpu.split(","))
    args.world_size = args.num_gpu * args.nodes
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    avail_port = str(get_available_port())
    print(f"running on {avail_port}")
    os.environ["MASTER_PORT"] = avail_port

    dataset_dir = osp.join(data_dir, args.dataset)
    pretrained = osp.basename(args.pretrained)
    hdf5_file = osp.join(dataset_dir, pretrained + ".vid.hdf5")
    subprocess.run(f"rm -rf {hdf5_file}", shell=True)

    tmp_dir = osp.join(dataset_dir, hdf5_file + ".tmp")
    os.makedirs(osp.dirname(hdf5_file), exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    if args.num_gpu > 1:
        print("using distribtued")
        mp.spawn(distributed_inference, nprocs=args.num_gpu, args=(args,))
    else:
        print("using single gpu inference")
        inference(args)


if __name__ == "__main__":
    main()