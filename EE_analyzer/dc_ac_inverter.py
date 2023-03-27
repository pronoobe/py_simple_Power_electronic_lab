import numba as nb
from ahkab import *
from matplotlib import pyplot as plt

from circult_graph import *
from device import *


class H_Bridge(object):
    def __init__(self, U, motor: DC_Motor, step=0.01, sub_step=1e-4):
        self.motor = motor
        self.U = U
        # 下面初始化两种拓扑下电路
        self.circuit_positive = Circuit('正极性下H桥')
        self.circuit_negative = Circuit('负极性下H桥')

        self.control_signal = None
        self.step = step
        self.now_Id = 0
        self.now_UR = 0
        self.now_UL = 0
        self.y = None
        self.U_res = []
        self.I_res = []
        self.sub_step = sub_step
        self._vector_step = np.vectorize(self._step)

    def load_pwm_wave(self, alpha, freq, time=0.1):
        # alpha: 占空比，0到1之间的小数
        # freq: 频率，单位为赫兹
        # time: 持续时间，单位为秒
        # step: 步长，单位为秒
        # 返回值：pwm信号和时间坐标的数组

        # 计算周期和高电平持续时间
        step = self.step * self.sub_step
        period = 1 / freq  # 周期，单位为秒
        high_time = alpha * period  # 高电平持续时间，单位为秒
        # 创建时间坐标数组
        t = np.arange(0, time + step, step)  # 时间坐标数组，从0到time（包含），步长为step

        # 创建pwm信号数组（使用向量化运算）
        pwm = np.array(t % period < high_time, dtype=np.int64)  # pwm信号数组，根据条件判断生成0或1
        self.control_signal = pwm
        return pwm, t

    def change_step(self, step):
        self.step = step

    def load_signal_in(self, signal):
        self.control_signal = signal

    def change_motor(self, motor: DC_Motor):
        self.motor = motor

    def _step(self, num):
        if num >= 0:
            self.circuit_positive = Circuit('正极性下H桥')
            self.circuit_positive.add_vsource('V', 'n1', self.circuit_positive.gnd, dc_value=1.0 * self.U, ac_value=0.)
            self.circuit_positive.add_resistor('R1', 'n1', 'n2', self.motor.R * 1.0)
            self.circuit_positive.add_inductor('L1', 'n2', 'n3', self.motor.L, ic=self.now_Id)
            self.circuit_positive.add_vsource('E', 'n3', self.circuit_positive.gnd, dc_value=1.0 * self.motor.E,
                                              ac_value=0.)
            dc = new_tran(0, self.step, self.sub_step, x0=None)
            res = run(self.circuit_positive, dc)
            U_R = get_results([('Vn1', "Vn2")], res['tran'])[0][0][-1]
            U_L1 = get_results([('Vn2', "Vn3")], res['tran'])[0][0][-1]
            Ud = U_R + U_L1 - self.motor.E
            I = get_results([('I(L1)', "")], res['tran'])[0][0][-1]
            self.now_UR = U_R
            self.now_UL = U_L1
            self.now_Id = I

        else:
            self.circuit_negative = Circuit('负极性下H桥')
            self.circuit_negative.add_vsource('V', 'n1', self.circuit_negative.gnd, dc_value=-1.0 * self.U, ac_value=0.)

            self.circuit_negative.add_vsource('E', 'n3', self.circuit_negative.gnd, dc_value=1.0 * self.motor.E,
                                              ac_value=0.)
            self.circuit_negative.add_resistor('R1', 'n1', 'n2', self.motor.R * 1.0)
            self.circuit_negative.add_inductor('L1', 'n2', 'n3', self.motor.L, ic=self.now_Id)
            dc = new_tran(0, self.step, self.sub_step, x0=None)
            res = run(self.circuit_negative, dc)
            U_R = get_results([('Vn1', "Vn2")], res['tran'])[0][0][-1]
            U_L1 = get_results([('Vn2', "Vn3")], res['tran'])[0][0][-1]
            Ud = U_R + U_L1 - self.motor.E
            I = get_results([('I(L1)', "")], res['tran'])[0][-1]
            self.now_UR = U_R
            self.now_UL = U_L1
            self.now_Id = I

    def plot_U(self, *args, **kwargs):
        x = np.arange(0, len(self.control_signal), self.step)
        plt.plot(self.U_res, *args, **kwargs)
        plt.show()

    def plot_I(self, *args, **kwargs):
        x = np.arange(0, len(self.control_signal), self.step)
        plt.plot(self.I_res, *args, **kwargs)
        plt.show()

    def calculate(self):
        j = 0
        l = len(self.control_signal)
        for i in self.control_signal:
            self._step(i)
            self.U_res.append(self.now_UR + self.now_UL - self.motor.E)
            self.I_res.append(self.now_Id)
            if j % 1 == 0:
                print("仿真了{}秒的数据，共{}秒，当前Ud为{}".format(j * self.step * self.sub_step, l * self.step * self.sub_step,
                                                      self.now_UR ))
            j += 1


if __name__ == '__main__':
    a = H_Bridge(5, motor=DC_Motor(3, 1, 0.1))
    a.load_pwm_wave(0.3, 20, 0.0001)
    a.calculate()
    print(a.I_res)
    # a.plot_U()
