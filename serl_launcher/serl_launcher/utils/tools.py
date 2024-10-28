import numpy as np


def ema(series, alpha=0.5):
    """
    Exponential moving average
    :param series: the input series
    :param alpha: the smoothing factor
    :return: the smoothed series
    """
    smoothed = np.zeros_like(series, dtype=float)
    smoothed[0] = series[0]
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed
