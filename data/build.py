from torchdata.dataloader2.reading_service import PrototypeMultiProcessingReadingService, MultiProcessingReadingService
from torchdata.dataloader2.dataloader2 import DataLoader2
# from torch.utils.data import DataLoader
# from .datapipe.parse import *
from torchdata.datapipes.iter import IterableWrapper
from models import *
from kn_util.basic import registry
from kn_util.distributed import is_ddp_initialized_and_available

def build_datapipe(cfg, split):
    return registry.build_datapipe(cfg.data.datapipe, cfg=cfg, split=split)
    # if cfg.data.datapipe == "default":
    #     return build_datapipe_default(cfg, split=split)
    # if cfg.data.datapipe == "msat":
    #     return build_datapipe_msat(cfg, split=split)


def build_dataloader(cfg, split="train"):
    assert split in ["train", "test", "val", "train_no_shuffle"]
    datapipe = build_datapipe(cfg, split=split)
    num_batches = len(datapipe)

    reading_service = PrototypeMultiProcessingReadingService(num_workers=cfg.train.num_workers) if cfg.train.num_workers else None
    dataloader = DataLoader2(datapipe, reading_service=reading_service)

    if is_ddp_initialized_and_available():
        # https://github.com/pytorch/data/issues/911
        dataloader = IterableWrapper(dataloader).fullsync()

    dataloader.num_batches = num_batches
    return dataloader
