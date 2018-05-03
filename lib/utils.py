import config as cfg
import os


def create_architecture():
    if not os.path.exists(cfg.model_dir):
        os.mkdir(cfg.model_dir)

    if not os.path.exists(cfg.loss_record_dir):
        os.mkdir(cfg.loss_record_dir)
