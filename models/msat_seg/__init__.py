from kn_util.basic import import_modules
import os.path as osp

cur_dir = osp.dirname(__file__)
import_modules(cur_dir, "models.msat")