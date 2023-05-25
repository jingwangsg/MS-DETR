import os.path as osp
import numpy as np
import sys
import os.path as osp

sys.path.insert(0, osp.join(osp.dirname(__file__), "../.."))
from torchdata.datapipes.iter import IterDataPipe
from kn_util.data.datapipe import *
from kn_util.basic import registry
import torch.nn.functional as F
import os
from misc import WordLabelTranslater


class ActivityNetAnnotationProcessor(IterDataPipe):

    def __init__(self, src_pipeline, num_clips) -> None:
        self.src_pipeline = src_pipeline
        self.num_clips = num_clips

    def get_item(self, annotation):
        gt_s, gt_e = annotation["gt"][0], annotation["gt"][1]
        if np.allclose(gt_s, gt_e):
            gt_e = gt_s + 1e-5
        num_clips = self.num_clips
        if not os.getenv("KN_NUM_CLIP_PLUS_1", False):
            num_clips -= 1  #HACK
        gt_s *= num_clips
        gt_e *= num_clips

        map_gt = np.zeros((5, num_clips + 1), dtype=np.float32)

        gt_length = gt_e - gt_s
        gt_center = (gt_e + gt_s) / 2.
        map_gt[0, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_s) / (0.25 * gt_length)))
        map_gt[0, map_gt[0, :] >= 0.6] = 1.
        map_gt[0, map_gt[0, :] < 0.1353] = 0.
        map_gt[1, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_e) / (0.25 * gt_length)))
        map_gt[1, map_gt[1, :] >= 0.6] = 1.
        map_gt[1, map_gt[1, :] < 0.1353] = 0.
        # map_gt[2, gt_s_idx:gt_e_idx] = 1.
        map_gt[2, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_center) / (0.21233 * gt_length)))
        map_gt[2, map_gt[2, :] >= 0.78] = 1.
        map_gt[2, map_gt[2, :] < 0.0625] = 0.
        map_gt[3, :] = gt_s - np.arange(num_clips + 1)
        map_gt[4, :] = gt_e - np.arange(num_clips + 1)
        if (map_gt[0, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_s) / (0.25 * gt_length)))
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_e) / (0.25 * gt_length)))
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.
        if map_gt[2, :].sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_center) / (0.21233 * gt_length)))
            idx = np.argmax(p)
            map_gt[2, idx] = 1.

        item = dict(map_gt=map_gt, gt_times=np.array([gt_s, gt_e]))

        return item

    def __iter__(self):
        for x in self.src_pipeline:
            ret_dict = self.get_item(x)
            x.update(ret_dict)
            yield x


class TACoSAnnotationProcessor(IterDataPipe):

    def __init__(self, src_pipeline, num_clips) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.num_clips = num_clips

    def get_item(self, x):
        num_clips = self.num_clips

        map_gt = np.zeros((5, num_clips + 1), dtype=np.float32)
        gt_s, gt_e = x["gt"][0], x["gt"][1]
        gt_s *= num_clips
        gt_e *= num_clips

        if not os.getenv("KN_NUM_CLIP_PLUS_1", False):
            num_clips -= 1  #HACK

        gt_length = gt_e - gt_s
        gt_center = (gt_e + gt_s) / 2.
        map_gt[0, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_s) / (0.25 * gt_length)))
        map_gt[0, map_gt[0, :] >= 0.7] = 1.
        map_gt[0, map_gt[0, :] < 0.1353] = 0.
        map_gt[1, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_e) / (0.25 * gt_length)))
        map_gt[1, map_gt[1, :] >= 0.7] = 1.
        map_gt[1, map_gt[1, :] < 0.1353] = 0.
        # map_gt[2, gt_s_idx:gt_e_idx] = 1.
        map_gt[2, :] = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_center) / (0.21233 * gt_length)))
        map_gt[2, map_gt[2, :] >= 0.78] = 1.
        map_gt[2, map_gt[2, :] < 0.0625] = 0.
        map_gt[3, :] = gt_s - np.arange(num_clips + 1)
        map_gt[4, :] = gt_e - np.arange(num_clips + 1)
        if (map_gt[0, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_s) / (0.25 * gt_length)))
            idx = np.argsort(p)
            map_gt[0, idx[-1]] = 1.
        if (map_gt[1, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_e) / (0.25 * gt_length)))
            idx = np.argsort(p)
            map_gt[1, idx[-1]] = 1.
        if map_gt[2, :].sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(num_clips + 1) - gt_center) / (0.21233 * gt_length)))
            idx = np.argmax(p)
            map_gt[2, idx] = 1.

        return dict(map_gt=map_gt, gt_times=np.array([gt_s, gt_e]))

    def __iter__(self):
        for x in self.src_pipeline:
            item = self.get_item(x)
            x.update(item)
            yield x


def before_batch(result):
    x = result["text.hdf5"]
    result["text.hdf5"] = x.squeeze(0)
    return result


def after_load_vid(result):
    result["video.hdf5"] = F.normalize(torch.from_numpy(result["video.hdf5"]), dim=1).numpy()
    return result


def after_collate(result):
    result["textual_mask"] = result["textual_mask"].float()
    # result["visual_input"] = F.normalize(result["visual_input"], dim=1)
    return result


@registry.register_datapipe("msat_roberta")
def build_datapipe_msat_roberta(cfg, split):
    dataset = cfg.data.dataset
    assert dataset != "charades"
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

    dataset_dp = dataset_dp

    # shuffle + sharding_filter
    dataset_dp = prepare_for_dataloader(dataset_dp, shuffle=is_train)
    if is_train:
        dataset_dp = dataset_dp.shuffle()
    num_samples = len(list(dataset_dp))

    # load text feature
    dataset_dp = dataset_dp.load_hdf5(hdf5_file=txt_hdf5, key_template=txt_hdf5_key_template, output_key_prefix="text")

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
    num_clips = cfg.data.max_len_video // cfg.data.target_stride
    if dataset == "activitynet":
        dataset_dp = ActivityNetAnnotationProcessor(dataset_dp, num_clips)
    if dataset == "tacos":
        dataset_dp = TACoSAnnotationProcessor(dataset_dp, num_clips)

    dataset_dp = dataset_dp.map(before_batch)

    # ========== BATCH BELOW ==========
    num_batches = int(np.ceil(num_samples / batch_size))
    # batchify
    dataset_dp = dataset_dp.batch(batch_size).rows2columnar()

    # pad
    dataset_dp = dataset_dp.pad_sequence(from_key="video.hdf5",
                                         axis=0,
                                         fill_value="last",
                                         to_length=cfg.data.max_len_video)
    dataset_dp = dataset_dp.pad_sequence(from_key="text.hdf5", axis=0, fill_value=0.0)

    # collect
    dataset_dp = dataset_dp.collect(
        ["video.hdf5.pad", "text.hdf5.pad", "text.hdf5.mask", "gt_times", "map_gt", "gt", "text_id"],
        ["visual_input", "textual_input", "textual_mask", "gt_times", "gt_maps", "gt", "text_id"])
    dataset_dp = dataset_dp.collate(default_collate_fn).map(after_collate)

    return dataset_dp.set_length(num_batches)


def get_word_mask(result, mask_rate=0.15):
    text_inds = result["text.inds"]

    length_text = len(text_inds)
    word_mask = np.array([np.random.uniform() < mask_rate for _ in range(length_text)])

    if np.sum(word_mask) == 0 or np.sum(word_mask) == length_text:
        random_idx = np.random.choice(np.arange(length_text))
        word_mask[random_idx] ^= 1

    result["word_mask"] = word_mask

    return result


@registry.register_datapipe("msat")
def build_datapipe_msat(cfg, split):
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

    if is_train:
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
    num_clips = cfg.data.max_len_video // cfg.data.target_stride
    if dataset == "activitynet":
        dataset_dp = ActivityNetAnnotationProcessor(dataset_dp, num_clips)
    if dataset == "tacos":
        dataset_dp = TACoSAnnotationProcessor(dataset_dp, num_clips)

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
    dataset_dp = dataset_dp.pad_sequence(from_key="text.inds.topk_freq", axis=0, fill_value=0.0, return_mask=False)

    # collect
    from_keys = [
        "video.hdf5.pad", "text.embs.pad", "text.embs.mask", "text.inds.topk_freq.pad", "gt_times", "map_gt", "gt",
        "text", "video_id"
    ]
    to_keys = [
        "visual_input", "textual_input", "textual_mask", "word_label", "gt_times", "gt_maps", "gt", "text", "video_id"
    ]

    if use_word_mask:
        from_keys.append("word_mask.pad")
        to_keys.append("word_mask")
        dataset_dp = dataset_dp.pad_sequence(from_key="word_mask", axis=0, fill_value=False, return_mask=False)

    dataset_dp = dataset_dp.collect(from_keys, to_keys)

    # collate
    dataset_dp = dataset_dp.collate(default_collate_fn).map(after_collate)
    dataset_dp = prepare_for_dataloader(dataset_dp, num_batches)

    return dataset_dp