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

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for current_sample, video_sample, label in iter(dataloader):
            yield video_sample, current_sample, label

def warmup_lr(step, cfg):
    return min(step, cfg.TRAIN.WARMUP) / cfg.TRAIN.WARMUP

def train():
    # cfg
    args = parse_args()
    cfg = load_config(args)

    # dataset
    print('[{}] Loading data set...'.format(get_time()))
    dataset = fmf_vicu(cfg=cfg, split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                             num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
    datalooper = infiniteloop(dataloader)

    val_dataset = fmf_vicu(cfg=cfg, split='test')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True,
                                                 num_workers=cfg.DATA_LOADER.NUM_WORKERS, pin_memory=cfg.DATA_LOADER.PIN_MEMORY)

    # model setup
    net_model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=cfg.TRAIN.LR)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: warmup_lr(step, cfg))
    trainer = build_trainer(cfg, net_model).to(device)
    net_sampler = build_sampler(cfg, net_model).to(device)
    ema_sampler = build_sampler(cfg, ema_model).to(device)
    if cfg.MODEL.PARALLEL:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    if not os.path.exists(cfg.COMMON.LOGDIR):
        os.makedirs(os.path.join(cfg.COMMON.LOGDIR, 'sample'))
    x_T = torch.randn(cfg.TRAIN.BATCH_SIZE, 3, cfg.DATA.IMAGE_SIZE, cfg.DATA.IMAGE_SIZE)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:cfg.DIFFUSION.SAMPLE_SIZE]) + 1) / 2
    writer = SummaryWriter(cfg.COMMON.LOGDIR)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(cfg.COMMON.LOGDIR, "flagfile.txt"), 'w') as f:
        f.write(cfg.dump())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(cfg.TRAIN.TOTAL_STEPS, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, c_0, _ = next(datalooper)
            x_0, c_0 = x_0.to(device), c_0.to(device)
            loss = trainer(x_0, c_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), cfg.TRAIN.GRAD_CLIP)
            optim.step()
            sched.step()
            ema(net_model, ema_model, cfg.TRAIN.EMA_DECAY)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if cfg.DIFFUSION.SAMPLE_STEP > 0 and step % cfg.DIFFUSION.SAMPLE_STEP == 0 and step > 0:
                print("\nSampling...")
                ema_model.eval()
                with torch.no_grad():
                    # x_T is your initial random noise
                    if cfg.DIFFUSION.get('SAMPLER_TYPE', 'DDPM').upper() == 'DDIM':
                        x_0 = ema_sampler(x_T, c_0, 
                                          num_steps=cfg.DIFFUSION.DDIM_NUM_STEPS, 
                                          eta=cfg.DIFFUSION.DDIM_ETA)
                    else: # DDPM
                        x_0 = ema_sampler(x_T, c_0) # Original DDPM call

                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        cfg.COMMON.LOGDIR, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                ema_model.train() # Switch back to train mode

            # save
            if cfg.TRAIN.SAVE_STEP > 0 and step % cfg.TRAIN.SAVE_STEP == 0 and step > 0:
                print("Saving...")
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(cfg.COMMON.CHECKPOINTS, 'ckpt.pt'))

            # evaluate
            if cfg.TEST.EVAL_STEP > 0 and step % cfg.TEST.EVAL_STEP == 0 and step > 0:
                print("Evaluating...")
                # acc, rec, f1, fdr, mdr, _ = evaluate(net_sampler, net_model, cfg, val_dataloader, device)
                ema_acc, ema_rec, ema_f1, ema_fdr, ema_mdr, _ = evaluate(ema_sampler, ema_model, cfg, val_dataloader, device)
                metrics = {
                    # 'ACC': acc,
                    # 'REC': rec,
                    # 'F1': f1,
                    # 'FDR': fdr,
                    # 'MDR': mdr,
                    'EMA_ACC': ema_acc,
                    'EMA_REC': ema_rec,
                    'EMA_F1': ema_f1,
                    'EMA_FDR': ema_fdr,
                    'EMA_MDR': ema_mdr
                }
                pbar.write(
                    "%d/%d " % (step, cfg.TRAIN.TOTAL_STEPS) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(cfg.COMMON.LOGDIR, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

def main():
    train()


if __name__ == '__main__':
    main()