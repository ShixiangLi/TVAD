import copy
import json
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import trange
from scipy.ndimage import gaussian_filter

from models.model import build_model, build_sampler, build_trainer
from datasets.dataset import fmf_vicu
from utils.parser import parse_args, load_config
from utils.others import get_time
from utils.metrics import evaluate

device = torch.device('cuda:0')

def eval():
    # cfg
    args = parse_args()
    cfg = load_config(args)

    # dataset
    val_dataset = fmf_vicu(cfg, split='test')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True,
                                                 num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)

    # model
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED, map_location=torch.device('cpu'))['ema_model'])
    model.to(device)
    model.eval()

    # sampler
    sampler = build_sampler(cfg, model)
    sampler.to(device)

    # evaluate
    img_acc, img_f1, img_fdr, img_mdr, _ = evaluate(sampler, model, cfg, val_dataloader, device)


if __name__ == '__main__':
    eval()