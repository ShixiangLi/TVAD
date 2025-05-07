from .ViCuDataset import ViCuDataset


def fmf_vicu(cfg, split, transform=None):
    dataset = ViCuDataset(
        directory=cfg.DATA.MAT_FILES, split=split
    )
    return dataset
