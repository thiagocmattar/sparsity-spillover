"""Ideal analytic rules used only for manuscript CPU verification."""
import numpy as np


def ideal_pressure(u, w, weight=1., budget=1.):
    u, w = np.asarray(u, dtype=float), np.asarray(w, dtype=float)
    q = u @ u
    if q == 0:
        return np.zeros_like(w)
    projected = w - (u @ w)/q*u if u @ w < 0 else w
    norm = np.linalg.norm(projected)
    return min(weight, budget*np.sqrt(q)/norm)*projected if norm else np.zeros_like(w)


def reach(layers, width, tokens, vocabulary):
    d = layers*(12*width+tokens+1)+vocabulary
    return dict(model_products=tokens*width*d, h=4*layers*width/d,
                mh=8*layers*width/d, a4=12*layers*width/d,
                a7=layers*(12*width+tokens+1)/d,
                attention_gap=layers*(tokens+1)/d)
