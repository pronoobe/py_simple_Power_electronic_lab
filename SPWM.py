import queue

import numpy as np
import scipy.signal
from matplotlib import pyplot as plt


def generate_sine_wave(A, freq, sample_num, duration, mirror=False):
    """

    :param freq:
    :param sample_num:
    :param duration: 几个周期
    :param mirror:
    :return:
    """
    x = np.linspace(0, duration * 1 / freq, sample_num, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = A * np.sin((2 * np.pi) * frequencies)
    return x, y


class signal(object):
    def __init__(self, wave, freq, wave_duration, step=0.001):
        self.wave = wave
        self.freq = freq
        self.wave_duration = wave_duration
        self.step = step

    def __repr__(self):
        plt.plot(self.x, self.step)
        plt.show()

    @property
    def x(self):
        return np.arange(0, self.wave_duration, self.step)

    @property
    def y(self):
        return self.wave


class _SPWM_sync(object):
    """
    同步调制的SPWM
    """

    def __init__(self, N: int or float or list or tuple, modulation_wave="tri"):
        self._modulation_wave_type = {"delta"}
        self.input_wave = None  # 输入波形
        self.square = None
        self.N = N  # 调制比
        self.freq = None  # 输入波形的频率
        self.wave_time = None  # 波形的持续时间
        self.step = None

    def generate_tri_wave(self, step=0.001, uc=1):
        """
        它生成一个三角波，频率为“freq”，持续时间为“duration”，步长为“step”

        :param step: 每个样本之间的时间步长。
        :return: 锯齿波绝对值的 numpy 数组。
        """
        duration = self.wave_time
        time = np.arange(0, duration, step)
        tri = scipy.signal.sawtooth(time * self.N * self.freq * 2 * np.pi, 0.5)
        return tri

    def get_wave_and_freq(self, wave, freq, wave_duration, step=0.001):
        """
        此函数接受波形、频率和波持续时间并返回波形和频率

        :param wave: 要播放的波形0
        :param freq: 波的频率
        :param wave_duration: 波的持续时间（以秒为单位）
        """
        """
        此函数接收波和频率，并将它们分配给类的 input_wave 和 freq 属性。

        :param wave: 输入信号的波形
        :param freq: 波的频率。
        """
        self.input_wave = wave
        self.freq = freq
        self.wave_time = wave_duration
        self.step = step

    def _clear_input(self):
        """
        清零输入信号
        :return:
        """
        self.input_wave = []
        self.freq = None
        self.wave_time = None

    def compare_and_generate_pwm(self, step=0.001, uc=1, plot=False):
        """
        比较信号并且得到PWM波
        :return:
        """
        tri_wave = self.generate_tri_wave(uc=2 * uc, step=step)
        self.square = np.zeros((1, self.wave_time * round(1 / step)))[0]
        x = np.arange(0, self.wave_time, step)
        for i in range(len(self.square)):
            if tri_wave[i] > self.input_wave[i]:
                self.square[i] = -1
            else:
                self.square[i] = 1
        self.square = self.square
        if plot:
            plt.subplot(211)
            plt.plot(x, tri_wave)
            plt.plot(x, self.input_wave)
            plt.subplot(212)
            plt.plot(x, self.square)
            plt.show()
        self._clear_input()  # 清除
        return x, self.square[0]


class _SPWM_async(_SPWM_sync):
    def __init__(self, freq):
        super(_SPWM_async, self).__init__(freq)
        self.N_constant = self.N  # 固定不变的N

    def generate_tri_wave(self, step=0.001, uc=1):
        """
        它生成一个三角波，频率为“freq”，持续时间为“duration”，步长为“step”

        :param step: 每个样本之间的时间步长。
        :return: 锯齿波绝对值的 numpy 数组。
        """
        duration = self.wave_time
        time = np.arange(0, duration, step)
        tri = scipy.signal.sawtooth(time * self.N_constant * np.pi, 0.5)
        return tri


class signal_manage(object):
    def __init__(self):
        self.signal = queue.Queue()  # 信号流

    def get_signal_in(self, signal: np.ndarray or list or tuple):
        """
        该函数接收一个信号，可以是列表、元组或 numpy 数组，并将其附加到信号列表

        :param signal: 输入信号。
        :type signal: np.ndarray or list or tuple
        """
        if isinstance(signal, (list, tuple)):
            signal = np.array(signal)
        self.signal.put(signal)

    def load_signal_out(self):
        """
        此函数从用户那里获取信号并将其返回
        :return: 正在返回信号。
        """
        signal = self.signal.get()
        return signal


class SPWM(signal_manage):
    def __new__(cls, mode):
        mixin = {'sync': _SPWM_sync, 'async': _SPWM_async}[mode]
        cls_mixin = type(mixin.__name__[1:], (cls, mixin), {})
        return super(SPWM, cls_mixin).__new__(cls_mixin)

    def __init__(self, mode, *args, **kwargs):
        super(SPWM, self).__init__(*args, **kwargs)
        super(signal_manage, self).__init__()

    def __add__(self, other):
        ...

    __iadd__ = __add__


if __name__ == "__main__":
    x, sinw = generate_sine_wave(1, 2, 4000, 4)
    pwm1 = _SPWM_sync(5)
    pwm2 = _SPWM_async(5)
    pwm1.get_wave_and_freq(wave=sinw, freq=2, wave_duration=4)
    pwm1.compare_and_generate_pwm(plot=True, step=1 / 1000)
    # print(pwm1.square)
