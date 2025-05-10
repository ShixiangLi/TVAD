from .unet import UNet
from .diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIMSampler

def build_model(cfg):
    model = UNet(
        T=cfg.MODEL.T,
        c_in=cfg.MODEL.CONDITION_IN,
        ch=cfg.MODEL.EMBED_DIM,
        ch_mult=cfg.MODEL.CHANNEL_MULT,
        attn=cfg.MODEL.ATTN_BLOCK,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        dropout=cfg.MODEL.DROPOUT
    )
    return model

def build_sampler(cfg, model):
    sampler_type = cfg.DIFFUSION.get('SAMPLER_TYPE', 'DDPM').upper() # Get sampler type, default to DDPM

    if sampler_type == 'DDIM':
        print("Building DDIM Sampler")
        sampler = DDIMSampler(
            model=model,
            beta_1=cfg.DIFFUSION.BETA_1,
            beta_T=cfg.DIFFUSION.BETA_T,
            T=cfg.MODEL.T,
            img_size=cfg.DATA.IMAGE_SIZE,
            mean_type=cfg.DIFFUSION.MEAN_TYPE, # Crucial: DDIM works best if model predicts 'epsilon'
            eta=cfg.DIFFUSION.DDIM_ETA # You'll add this to config
        )
    elif sampler_type == 'DDPM':
        print("Building DDPM Sampler (GaussianDiffusionSampler)")
        sampler = GaussianDiffusionSampler(
            model=model,
            beta_1=cfg.DIFFUSION.BETA_1,
            beta_T=cfg.DIFFUSION.BETA_T,
            T=cfg.MODEL.T,
            img_size=cfg.DATA.IMAGE_SIZE,
            mean_type=cfg.DIFFUSION.MEAN_TYPE,
            var_type=cfg.DIFFUSION.VAR_TYPE
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
        
    return sampler


def build_trainer(cfg, model):
    trainer = GaussianDiffusionTrainer(
        model=model,
        beta_1=cfg.DIFFUSION.BETA_1,
        beta_T=cfg.DIFFUSION.BETA_T,
        T=cfg.MODEL.T
    )
    return trainer