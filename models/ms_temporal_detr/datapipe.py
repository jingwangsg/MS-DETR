import os.path as osp
import numpy as np
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from torchdata.datapipes.iter import IterDataPipe
from kn_util.data.datapipe import *
from kn_util.basic import registry
import torch.nn.functional as F
from ..msat.datapipe import ActivityNetAnnotationProcessor, TACoSAnnotationProcessor
from misc import WordLabelTranslater


def before_batch(result):
    x = result["text.hdf5"]
    result["text.hdf5"] = x.squeeze(0)
    return result


def after_load_vid(result):
    result["video.hdf5"] = F.normalize(torch.from_numpy(result["video.hdf5"]), dim=1).numpy()
    return result


def after_collate(result):
    result["txt_mask"] = result["txt_mask"].float()
    # result["visual_input"] = F.normalize(result["visual_input"], dim=1)
    return result


def get_word_mask(result, mask_rate=0.15):
    text_inds = result["text.inds"]

    length_text = len(text_inds)
    word_mask = np.array([np.random.uniform() < mask_rate for _ in range(length_text)])

    if np.sum(word_mask) == 0 or np.sum(word_mask) == length_text:
        random_idx = np.random.choice(np.arange(length_text))
        word_mask[random_idx] ^= 1

    result["word_mask"] = word_mask

    return result


@registry.register_datapipe("mst_detr")
def build_datapipe_mst_detr(cfg, split):
    dataset = cfg.data.dataset
    assert dataset != "charades"
    dataset_dir = cfg.data.dataset_dir
    max_len_video = cfg.data.max_len_video
    vid_hdf5 = osp.join(dataset_dir, cfg.data.vid_hdf5)
    vid_hdf5_key_template = cfg.data.vid_hdf5_key_template
    batch_size = cfg.train.batch_size
    use_word_mask = cfg.data.word_mask_rate > 0.0
    is_train = (split == "train")

    # parse dataset
    dataset_dp = build_tsvg_parser(cfg, split=split)
    # filter nonexisted hdf5 key
    dataset_dp = dataset_dp.filter_by_hdf5_key(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template)
    dataset_dp = dataset_dp

    # shuffle + sharding_filter
    dataset_dp = prepare_for_dataloader(dataset_dp, shuffle=is_train)
    num_samples = len(list(dataset_dp))

    # load text feature
    dataset_dp = dataset_dp.tokenize_glove(cache_dir=osp.join(cfg.paths.cache_dir, ".glove"),
                                           to_embeddings=True,
                                           to_indices=True,
                                           upload_vocab_key="glove_vocab",
                                           from_key="text")
    dataset_dp = WordLabelTranslater(dataset_dp, "text.inds")
    # load video feature
    dataset_dp = dataset_dp.load_hdf5(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template, output_key_prefix="video")
    dataset_dp = dataset_dp.map(after_load_vid)
    # sample video feature
    dataset_dp = dataset_dp.sample_sequence(from_key="video.hdf5",
                                            axis=0,
                                            stride="round",
                                            max_len=max_len_video,
                                            inplace=True,
                                            mode="avgpool")

    # add word mask
    if use_word_mask:
        dataset_dp = dataset_dp.map(get_word_mask)

    # ========== BATCH BELOW ==========
    num_batches = int(np.ceil(num_samples / batch_size))
    # batchify
    dataset_dp = dataset_dp.batch(batch_size).rows2columnar()

    # pad
    dataset_dp = dataset_dp.pad_sequence(from_key="video.hdf5",
                                         axis=0,
                                         fill_value="last",
                                         to_length=cfg.data.max_len_video)
    dataset_dp = dataset_dp.pad_sequence(from_key="text.embs", axis=0, fill_value=0.0)

    # collect
    from_keys = ["video.hdf5.pad", "text.embs.pad", "text.embs.mask", "gt"]
    to_keys = ["vid_feat", "txt_feat", "txt_mask", "gt"]

    if use_word_mask:
        dataset_dp = dataset_dp.pad_sequence(from_key="word_mask", axis=0, fill_value=False, return_mask=False)
        dataset_dp = dataset_dp.pad_sequence(from_key="text.inds", axis=0, fill_value=0.0, return_mask=False)
        from_keys += ["word_mask.pad", "text.inds.pad"]
        to_keys += ["word_mask", "word_label"]

    dataset_dp = dataset_dp.collect(from_keys, to_keys)

    # collate
    dataset_dp = dataset_dp.collate(default_collate_fn).map(after_collate)

    return dataset_dp.set_length(num_batches)


@registry.register_datapipe("mst_detr_v2")
def build_datapipe_mst_detr_v2(cfg, split):
    dataset = cfg.data.dataset
    dataset_dir = cfg.data.dataset_dir
    max_len_video = cfg.data.max_len_video
    vid_hdf5 = osp.join(dataset_dir, cfg.data.vid_hdf5)
    vid_hdf5_key_template = cfg.data.vid_hdf5_key_template
    batch_size = cfg.train.batch_size
    use_word_mask = cfg.data.word_mask_rate > 0.0
    is_train = (split == "train")

    # parse dataset
    dataset_dp = build_tsvg_parser(cfg, split=split)
    # filter nonexisted hdf5 key
    dataset_dp = dataset_dp.filter_by_hdf5_key(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template)

    # shuffle + sharding_filter
    if is_train and not cfg.data.get("no_shuffle", False):
        dataset_dp = dataset_dp.shuffle()
    num_samples = len(list(dataset_dp))

    # load text feature
    dataset_dp = dataset_dp.tokenize_glove(cache_dir=osp.join(cfg.paths.cache_dir, ".glove"),
                                           to_embeddings=True,
                                           to_indices=True,
                                           upload_vocab_key="glove_vocab",
                                           from_key="text")
    dataset_dp = WordLabelTranslater(dataset_dp, "text.inds")
    # load video feature
    dataset_dp = dataset_dp.load_hdf5(hdf5_file=vid_hdf5, key_template=vid_hdf5_key_template, output_key_prefix="video")
    dataset_dp = dataset_dp.map(after_load_vid)
    # sample video feature
    dataset_dp = dataset_dp.sample_sequence(from_key="video.hdf5",
                                            axis=0,
                                            stride="round",
                                            max_len=max_len_video,
                                            inplace=True,
                                            mode="avgpool")

    # process annotation
    # num_clips = cfg.data.max_len_video // cfg.data.target_stride
    # if dataset == "activitynet":
    #     dataset_dp = ActivityNetAnnotationProcessor(dataset_dp, num_clips)
    # if dataset == "tacos":
    #     dataset_dp = TACoSAnnotationProcessor(dataset_dp, num_clips)

    # add word mask
    if use_word_mask:
        dataset_dp = dataset_dp.map(get_word_mask)

    # ========== BATCH BELOW ==========
    num_batches = int(np.ceil(num_samples / batch_size))
    # batchify
    dataset_dp = dataset_dp.batch(batch_size).rows2columnar()

    # pad
    dataset_dp = dataset_dp.pad_sequence(from_key="video.hdf5",
                                         axis=0,
                                         fill_value="last",
                                         to_length=cfg.data.max_len_video)
    dataset_dp = dataset_dp.pad_sequence(from_key="text.embs", axis=0, fill_value=0.0)
    dataset_dp = dataset_dp.pad_sequence(from_key="text.inds", axis=0, fill_value=0.0, return_mask=False)

    # collect
    from_keys = ["video.hdf5.pad", "text.embs.pad", "text.embs.mask", "gt", "video_id", "text"]
    to_keys = ["vid_feat", "txt_feat", "txt_mask", "gt", "video_id", "text"]

    if use_word_mask:
        from_keys.extend(["word_mask.pad", "text.inds.topk_freq.pad"])
        to_keys.extend(["word_mask", "word_label"])
        dataset_dp = dataset_dp.pad_sequence(from_key="word_mask", axis=0, fill_value=False, return_mask=False)
        dataset_dp = dataset_dp.pad_sequence(from_key="text.inds.topk_freq", axis=0, fill_value=False, return_mask=False)

    dataset_dp = dataset_dp.collect(from_keys, to_keys)

    # collate
    dataset_dp = dataset_dp.collate(default_collate_fn).map(after_collate)

    dataset_dp = prepare_for_dataloader(dataset_dp, num_batches=num_batches)

    return dataset_dp