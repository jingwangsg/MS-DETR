import os.path as osp
from kn_util.data.datapipe import *
import numpy as np
from kn_util.data.datapipe import prepare_for_dataloader

# def get_word_mask_(result, mask_rate=0.15):
#     text_inds = result["text.inds"]

#     length_text = len(text_inds)
#     num_mask = int(np.ceil(length_text * 0.15))
#     word_mask = np.zeros_like(text_inds, dtype=bool)
#     if num_mask:
#         mask_inds = np.random.choice(np.arange(length_text), size=num_mask)
#         word_mask[mask_inds] = True

#     result["word_mask"] = word_mask

#     return result

# def get_word_mask(result, mask_rate=0.15):
#     text_inds = result["text.inds"]

#     length_text = len(text_inds)
#     word_mask = np.array([np.random.uniform() < mask_rate for _ in range(length_text)])

#     if np.sum(word_mask) == 0 or np.sum(word_mask) == length_text:
#         random_idx = np.random.choice(np.arange(length_text))
#         word_mask[random_idx] ^= 1

#     result["word_mask"] = word_mask

#     return result


def _squeeze0(result):
    x = result["text.hdf5"]
    result["text.hdf5"] = x.squeeze(0)
    return result


def build_datapipe_default(cfg, split):
    dataset_dir = cfg.data.dataset_dir
    max_len_video = cfg.data.max_len_video
    vid_hdf5 = osp.join(dataset_dir, cfg.data.vid_hdf5)
    vid_hdf5_key_template = cfg.data.vid_hdf5_key_template
    txt_hdf5 = osp.join(dataset_dir, cfg.data.txt_hdf5)
    txt_hdf5_key_template = cfg.data.txt_hdf5_key_template
    batch_size = cfg.train.batch_size
    # vocab_file = osp.join(dataset_dir, "annot", "vocab.txt")
    # cache_dir = cfg.paths.cache_dir
    is_train = (split == "train")

    # parse dataset
    dataset_dp = build_tsvg_parser(cfg, split=split)
    # filter nonexisted hdf5 key
    dataset_dp = dataset_dp.filter_by_hdf5_key(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template)
    dataset_dp = dataset_dp.in_memory_cache()

    # shuffle + sharding_filter
    dataset_dp = prepare_for_dataloader(dataset_dp)

    # load text feature
    dataset_dp = dataset_dp.load_hdf5(hdf5_file=txt_hdf5, key_template=txt_hdf5_key_template,
                                      output_key_prefix="text").map(_squeeze0)

    if is_train:
        dataset_dp = dataset_dp.shuffle()
    # load video feature
    dataset_dp = dataset_dp.load_hdf5(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template, output_key_prefix="video")
    # sample video feature
    dataset_dp = dataset_dp.sample_sequence(from_key="video.hdf5", axis=0, max_len=max_len_video, inplace=True)

    # ========== BATCH BELOW ==========
    # batchify
    dataset_dp = dataset_dp.batch(batch_size).rows2columnar()

    # pad
    dataset_dp = dataset_dp.pad_sequence(from_key="video.hdf5", axis=0, fill_value="last")
    dataset_dp = dataset_dp.pad_sequence(from_key="text.hdf5", axis=0, fill_value=0.0)

    # collect
    dataset_dp = dataset_dp.collect(["video.hdf5.pad", "video.hdf5.mask", "text.hdf5.pad", "text.hdf5.mask"],
                                    ["vid_feat", "vid_mask", "txt_feat", "txt_mask"])

    dataset_dp = dataset_dp.collate(default_collate_fn)
    return dataset_dp
