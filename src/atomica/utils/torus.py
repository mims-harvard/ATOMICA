import numpy as np
import tqdm
import os
from pathlib import Path

"""
    Source: https://github.com/gcorso/DiffDock/blob/main/utils/torus.py
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""


def p_torus(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1, desc='torus calculating p'):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1, desc='torus calculating grad'):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

# Lazy loading - only compute when needed
_p_ = None
_score_ = None

def _load_torus_data():
    global _p_, _score_
    if _p_ is not None and _score_ is not None:
        return _p_, _score_
    
    # Get the directory where this file is located (atomica/utils/)
    utils_dir = Path(__file__).parent
    p_file = utils_dir / '.p.npy'
    score_file = utils_dir / '.score.npy'
    
    x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
    sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

    if p_file.exists() and score_file.exists():
        _p_ = np.load(p_file)
        _score_ = np.load(score_file)
    else:
        _p_ = p_torus(x, sigma[:, None], N=100)
        np.save(p_file, _p_)

        _score_ = grad(x, sigma[:, None], N=100) / _p_
        np.save(score_file, _score_)
    
    return _p_, _score_


def score(x, sigma):
    p_, score_ = _load_torus_data()
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    p_, score_ = _load_torus_data()
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


# Lazy loading for score_norm
_score_norm_ = None

def _load_score_norm():
    global _score_norm_
    if _score_norm_ is not None:
        return _score_norm_
    
    x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
    sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi
    
    _score_norm_ = score(
        sample(sigma[None].repeat(10000, 0).flatten()),
        sigma[None].repeat(10000, 0).flatten()
    ).reshape(10000, -1)
    _score_norm_ = (_score_norm_ ** 2).mean(0)
    
    return _score_norm_

def score_norm(sigma):
    score_norm_ = _load_score_norm()
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]