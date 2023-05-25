from kn_util.data import FFMPEG
import argparse
args = argparse.ArgumentParser()
args.add_argument("dataset", choices=["tacos", "charades", "activitynet"])
args.add_argument("--overwrite", action="store_true", default=False)
args = args.parse_args()

dataset = args.dataset
video_dir = f"/export/home2/kningtg/DATASET/TSGV_Data/videos/{dataset}/"
image_root = f"/export/home2/kningtg/DATASET/TSGV_Data/images/{dataset}"

FFMPEG.multiple_video_to_image_async(
    video_dir=video_dir,
    image_root=image_root,
    frame_scale="224:224",
    max_frame=512,
    overwrite=args.overwrite)
