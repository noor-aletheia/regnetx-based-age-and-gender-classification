import numpy as np
from reglayers import AnyNet, WrappedModel
import torch
import os

REGNET_MODEL_ZOO = {
    'regnet_200m': {
        'config': {'WA': 36.44, 'W0': 24, 'WM': 2.49, 'DEPTH': 13, 'GROUP_W': 8, 'BOT_MUL': 1},
        'feature_dim': 368,
    },
    'regnet_400m': {
        'config': {'WA': 24.48, 'W0': 24, 'WM': 2.54, 'DEPTH': 22, 'GROUP_W': 16, 'BOT_MUL': 1},
        'feature_dim': 384,
    },
    'regnet_600m': {
        'config': {'WA': 36.97, 'W0': 48, 'WM': 2.24, 'DEPTH': 16, 'GROUP_W': 24, 'BOT_MUL': 1},
        'feature_dim': 528,
    },
    'regnet_800m': {
        'config': {'WA': 35.73, 'W0': 56, 'WM': 2.28, 'DEPTH': 16, 'GROUP_W': 16, 'BOT_MUL': 1},
        'feature_dim': 672,
    },
    'regnet_1600m': {
        'config': {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'BOT_MUL': 1},
        'feature_dim': 912,
    },
    'regnet_3200m': {
        'config': {'WA': 26.31, 'W0': 88, 'WM': 2.25, 'DEPTH': 25, 'GROUP_W': 48, 'BOT_MUL': 1},
        'feature_dim': 1008,
    },
    'regnet_6400m': {
        'config': {'WA': 60.83, 'W0': 184, 'WM': 2.07, 'DEPTH': 17, 'GROUP_W': 56, 'BOT_MUL': 1},
        'feature_dim': 1624,
    },
}


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont



class RegNet(AnyNet):
    """RegNet model."""
    def __init__(self, cfg, **kwargs):
        b_ws, num_s, _, _ = generate_regnet(
            cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH']
        )
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [cfg['BOT_MUL'] for _ in range(num_s)]
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        ss = [2 for _ in range(num_s)]
        se_r = None
        STEM_W = 32
        kwargs = {
            "stem_w": STEM_W,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": 1000,
        }
        super(RegNet, self).__init__(**kwargs)


def get_regnet(model_name, pretrained=False, weights_dir="./regnet_weights", **kwargs):
    """
    Create a RegNet model for the given model_name (e.g., 'regnet_1600m').
    Loads weights from local regnet_weights folder only.
    """
    model_name = model_name.lower()
    if model_name not in REGNET_MODEL_ZOO:
        raise ValueError(f"Unknown RegNet model: {model_name}")
    cfg = REGNET_MODEL_ZOO[model_name]['config']
    model = RegNet(cfg, **kwargs)
    if pretrained:
        weights_path = os.path.join(weights_dir, f"{model_name}.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No weights file found for {model_name} in {weights_dir}. Please place the weights manually as {model_name}.pth.")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    return model
