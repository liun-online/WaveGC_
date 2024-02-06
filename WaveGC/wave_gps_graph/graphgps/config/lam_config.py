from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_lam')
def set_cfg_lam(cfg):
    cfg.lam = CN()
    cfg.lam.nheads = 8
    cfg.lam.n_coe = 5
    cfg.lam.n_scales = 3
    cfg.lam.trans_dropout = 0.1
    cfg.lam.adj_dropout = 0.0
    cfg.lam.thre = 10.
    cfg.lam.low_thre = 0.5
    cfg.lam.high_thre = 10.
    cfg.lam.drop = 0.5
    cfg.lam.tight_use = True
    cfg.lam.ham = False