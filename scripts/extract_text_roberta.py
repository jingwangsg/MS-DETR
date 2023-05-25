import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
import os.path as osp
import argparse
import os
import sys
import h5py
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from kn_util.basic.file import load_json, save_hdf5, load_csv
import subprocess

data_dir = "/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw"


def load_data_activitynet():
    annot_dir = osp.join(data_dir, "activitynet", "annot")
    ret_dataset = []
    for domain in ["train", "val", "test"]:
        json_file = osp.join(annot_dir, domain + ".json")
        json_dict = load_json(json_file)
        for video_id, annot in json_dict.items():
            for idx, sentence in enumerate(annot["sentences"]):
                text_id = f"{video_id}_{idx}"
                cur_elem = dict(text_id=text_id, text=sentence)
                ret_dataset += [cur_elem]

    return ret_dataset


def load_data_tacos():
    annot_dir = osp.join(data_dir, "tacos", "annot")
    ret_dataset = []
    for domain in ["train", "val", "test"]:
        json_file = osp.join(annot_dir, domain + ".json")
        json_dict = load_json(json_file)
        for video_id, annot in json_dict.items():
            video_id = video_id[:-4]
            for idx, sentence in enumerate(annot["sentences"]):
                text_id = f"{video_id}_{idx}"
                cur_elem = dict(text_id=text_id, text=sentence)
                ret_dataset += [cur_elem]

    return ret_dataset


def load_data_charades():
    annot_dir = osp.join(data_dir, "charades", "annot")
    ret_dataset = []
    for domain in ["train", "test"]:
        txt_file = osp.join(annot_dir, f"charades_sta_{domain}.txt")
        with open(txt_file, "r") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                annot, sentence = line.split("##")
                video_id, st, ed = annot.split()
                text_id = f"{video_id}_{idx}"
                cur_elem = dict(text=sentence, text_id=text_id)
                ret_dataset += [cur_elem]

    return ret_dataset


def load_data(dataset):
    if dataset == "activitynet":
        return load_data_activitynet()
    elif dataset == "tacos":
        return load_data_tacos()
    elif dataset == "charades":
        return load_data_charades()


args = argparse.ArgumentParser()
args.add_argument("dataset", choices=["tacos", "charades", "activitynet"])
args.add_argument("--gpu", default="0", type=str)
args.add_argument("--pretrained", default="roberta-large", type=str)
args = args.parse_args()

dataset_dir = osp.join(data_dir, args.dataset)
os.environ["CUDA_VISIBLE_DEIVCES"] = args.gpu
model = AutoModel.from_pretrained(args.pretrained)
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

dataset = load_data(args.dataset)

hdf5_file = osp.join(dataset_dir, args.pretrained + ".txt.hdf5")
subprocess.run(f"rm -rf {hdf5_file}", shell=True)
f = h5py.File(hdf5_file, "w")

with torch.no_grad():
    for e in tqdm(dataset):
        text = e["text"]
        text_id = e["text_id"]
        inputs = tokenizer(text, max_length=model.config.max_position_embeddings, return_tensors="pt", truncation=True)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden_state = outputs["last_hidden_state"].detach().cpu().numpy()
        input_ids = inputs["input_ids"].detach().cpu().numpy()

        save_dict = {text_id: dict(last_hidden_state=last_hidden_state, input_ids=input_ids)}
        save_hdf5(save_dict, f, compression="gzip", compression_opts=9)

f.close()