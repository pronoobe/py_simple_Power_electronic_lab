import numpy as np
from matplotlib import pyplot as plt


class X(object):
    def __init__(self):
        self.hello = 'hello'


class Y(object):
    def __init__(self):
        self.moo = 'moo'


class D(object):
    def __init__(self):
        self.aa = 111


class Z(D):
    def __new__(cls, mode):
        mixin = {'X': X, 'Y': Y}[mode]
        cls_mixin = type(mixin.__name__, (cls, mixin), {})
        return super(Z, cls_mixin).__new__(cls_mixin)

    def __init__(self, mode, *args, **kwargs):
        super(Z, self).__init__(*args, **kwargs)
        super(D, self).__init__()


class signal(object):
    def __init__(self, wave, freq, wave_duration, step=0.001):
        self.wave = wave
        self.freq = freq
        self.wave_duration = wave_duration
        self.step = step

    def __repr__(self):
        x = np.arange(0, self.wave_duration, self.step)
        plt.plot(x, self.step)


def aaa(d, *args, **kwargs):
    print(d)
    print(args)
    print(kwargs)





if __name__ == '__main__':
    aaa(1, 2, c=2, e=3)
