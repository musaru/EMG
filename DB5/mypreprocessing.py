import scipy.signal
import numpy as np


def lpf(x, f=1., fs=100):
    if len(x) < 3:  # 入力信号の長さをチェック
        return x  # 信号をそのまま返して処理を飛ばす
        
    f = f / (fs / 2)
    x = np.abs(x)
    b, a = scipy.signal.butter(1, f, 'low')
    output = scipy.signal.filtfilt(
        b, a, x, axis=0,
        padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)
    )
    return output
