# > The IGBT class is a Python class that represents an IGBT device

import numpy as np
from matplotlib import pyplot as plt
from numpy import pi

from device import *


# 这是一个包含全桥整流器所有数据的类
class full_bridge(object):
    def __init__(self, U, load_type="RL", alpha=0.0, acc=0.01):
        """
        该函数接受负载类型、电压和模拟精度。然后，它会为 x 轴创建一个值列表，并为每相电压创建一个值列表。

        :param U: 系统电压
        :param load_type: RL、RC、RLC, defaults to RL (optional)
        :param alpha: 负载角度, defaults to 0 (optional)
        :param acc: 模拟的准确性
        """
        self.load_type = load_type
        self.U = U
        self.acc = acc
        self.alpha = alpha
        self.uab = None
        self.uac = None
        self.ubc = None
        self.uba = None
        self.uca = None
        self.ucb = None
        # 为 x 轴创建值列表。
        self.x_axis = np.arange(0, np.pi * 4, acc)
        self.T = []
        self.T.append(IGBT([pi / 6 + alpha, pi / 6 + 2 * pi / 3 + alpha], name="1"))
        self.T.append(IGBT([pi / 6 + alpha + 2 * pi / 3, pi / 6 + alpha + 4 * pi / 3], name="3"))
        self.T.append(IGBT([pi / 6 + alpha + 4 * pi / 3, pi / 6 + alpha + 6 * pi / 3], name="5"))
        self.T.append(IGBT([pi / 6 + alpha, pi / 6 + alpha + pi / 3], name="6"))
        self.T.append(IGBT([pi / 6 + pi + 2 * pi / 3 + alpha, pi / 6 + 2 * pi + alpha], name="6"))
        self.T.append(IGBT([pi / 6 + alpha + pi / 3, pi / 6 + alpha + pi], name="2"))
        self.T.append(IGBT([pi / 6 + alpha + pi, pi / 6 + pi + 2 * pi / 3 + alpha], name="4"))

    @staticmethod
    def _relu(x):
        return x if x >= 0 else 0

    def _uab(self, angel):
        return np.sqrt(3) * self.U * np.sin(angel + np.pi / 6)

    def _uac(self, angel):
        return np.sqrt(3) * self.U * np.sin(angel - np.pi / 3 + np.pi / 6)

    def _ubc(self, angel):
        return np.sqrt(3) * self.U * np.sin(angel - 2 * np.pi / 3 + np.pi / 6)

    def _uba(self, angel):
        return np.sqrt(3) * self.U * np.sin(angel - 3 * np.pi / 3 + np.pi / 6)

    def _uca(self, angel):
        return np.sqrt(3) * self.U * np.sin(angel - 4 * np.pi / 3 + np.pi / 6)

    def _ucb(self, angel):
        return np.sqrt(3) * self.U * np.sin(angel - 5 * np.pi / 3 + np.pi / 6)

    def plot_line_voltage(self, num):
        """
        绘制线电压波形
        :return:
        """
        self.uab = np.sqrt(3) * self.U * np.sin(np.linspace(0, pi * 5, num=num) + np.pi / 6)
        self.uac = np.sqrt(3) * self.U * np.sin(np.linspace(0, pi * 5, num=num) - np.pi / 3 + np.pi / 6)
        self.ubc = np.sqrt(3) * self.U * np.sin(np.linspace(0, pi * 5, num=num) - 2 * np.pi / 3 + np.pi / 6)
        self.uba = np.sqrt(3) * self.U * np.sin(np.linspace(0, pi * 5, num=num) - 3 * np.pi / 3 + np.pi / 6)
        self.uca = np.sqrt(3) * self.U * np.sin(np.linspace(0, pi * 5, num=num) - 4 * np.pi / 3 + np.pi / 6)
        self.ucb = np.sqrt(3) * self.U * np.sin(np.linspace(0, pi * 5, num=num) - 5 * np.pi / 3 + np.pi / 6)
        plt.plot(np.linspace(0, 5 * np.pi, num=num), self.uab, linestyle="--", color="blue")
        plt.plot(np.linspace(0, 5 * np.pi, num=num), self.uac, linestyle="--", color="blue")
        plt.plot(np.linspace(0, 5 * np.pi, num=num), self.ubc, linestyle="--", color="blue")
        plt.plot(np.linspace(0, 5 * np.pi, num=num), self.uba, linestyle="--", color="blue")
        plt.plot(np.linspace(0, 5 * np.pi, num=num), self.uca, linestyle="--", color="blue")
        plt.plot(np.linspace(0, 5 * np.pi, num=num), self.ucb, linestyle="--", color="blue")

    def get_on_voltage_circle(self):
        plt.title("alpha = {}".format((self.alpha / np.pi) * 180) + "°  " + "load_type: " + self.load_type)
        """
        获取一个周期内哪些器件导通了
        :return:
        """
        voltage_x = []
        for angel in np.arange(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6, self.acc):  # 遍历所有时刻
            is_on_single = set()
            for device in self.T:
                device.set_angle(angel)
                if device.is_on:
                    is_on_single.add(device.name)  # 添加导通的器件名字
            if is_on_single == {"1", "6"}:
                voltage_x.append(self._uab(angel))
            elif is_on_single == {"1", "2"}:
                voltage_x.append(self._uac(angel))
            elif is_on_single == {"2", "3"}:
                voltage_x.append(self._ubc(angel))
            elif is_on_single == {"3", "4"}:
                voltage_x.append(self._uba(angel))
            elif is_on_single == {"4", "5"}:
                voltage_x.append(self._uca(angel))
            elif is_on_single == {"5", "6"}:
                voltage_x.append(self._ucb(angel))
        return voltage_x

    def plot_Ud(self, *args, **kwargs):
        plt.cla()
        """
        画出负载曲线
        :param args:
        :param kwargs:
        :return:
        """
        Ud = self.get_on_voltage_circle()
        Ud = Ud * 2
        relu = np.frompyfunc(self._relu, 1, 1)  # 向量化
        if self.load_type == "RL":
            plt.plot(np.linspace(self.alpha + pi / 6, 4 * np.pi + self.alpha + pi / 6, num=len(Ud)), Ud, *args,
                     **kwargs)
            self.plot_line_voltage(len(Ud))
            plt.show()
        else:
            plt.plot(np.linspace(self.alpha + pi / 6, 4 * np.pi + self.alpha + pi / 6, num=len(Ud)), relu(Ud), *args,
                     **kwargs)
            self.plot_line_voltage(len(Ud))
            plt.show()

    def plot_UT(self, ut_num, *args, **kwargs):
        plt.cla()
        plt.title("VT{} ".format(ut_num) + "alpha = {}".format((self.alpha / np.pi) * 180) + "°  ")
        # uab = np.frompyfunc(self._uab, 1, 1)  # 向量化
        # uac = np.frompyfunc(self._uab, 1, 1)  # 向量化
        # ubc = np.frompyfunc(self._uab, 1, 1)  # 向量化
        # uba = np.frompyfunc(self._uab, 1, 1)  # 向量化
        # uca = np.frompyfunc(self._uab, 1, 1)  # 向量化
        # ucb = np.frompyfunc(self._uab, 1, 1)  # 向量化
        if ut_num == "1":
            range1 = np.arange(2 * pi + self.alpha + pi / 6 + 2 * pi / 3, 2 * pi + self.alpha + pi / 6 + 4 * pi / 3,
                               self.acc)
            range2 = np.arange(2 * pi + self.alpha + pi / 6 + 4 * pi / 3, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3,
                               self.acc)
            uab = np.frompyfunc(self._uab, 1, 1)  # 向量化
            uac = np.frompyfunc(self._uac, 1, 1)  # 向量化
            Ut_on = np.arange(self.alpha + pi / 6, 2 * pi / 3 + self.alpha + pi / 6, self.acc) * 0
            Ut_ab = uab(range1)
            Ut_ac = uac(range2)
            Ut = np.hstack((Ut_on, Ut_ab, Ut_ac, Ut_on, Ut_ab, Ut_ac))
            self.plot_line_voltage(len(Ut))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(Ut)), Ut,
                     *args,
                     **kwargs)
            plt.show()
        elif ut_num == "3":
            range1 = np.arange(2 * pi + self.alpha + pi / 6 + 4 * pi / 3, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3,
                               self.acc)
            range2 = np.arange(2 * pi + self.alpha + pi / 6 + 6 * pi / 3, 2 * pi + self.alpha + pi / 6 + 8 * pi / 3,
                               self.acc)
            ubc = np.frompyfunc(self._ubc, 1, 1)  # 向量化
            uba = np.frompyfunc(self._uba, 1, 1)  # 向量化
            Ut_on = np.arange(2 * pi / 3 + self.alpha + pi / 6, 4 * pi / 3 + self.alpha + pi / 6, self.acc) * 0
            Ut_ab = ubc(range1)
            Ut_ac = uba(range2)
            Ut = np.hstack((Ut_ac, Ut_on, Ut_ab, Ut_ac, Ut_on, Ut_ab, Ut_ac))
            self.plot_line_voltage(len(Ut))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 8 * pi / 3, num=len(Ut)), Ut,
                     *args,
                     **kwargs)
            plt.show()
        elif ut_num == "5":
            range1 = np.arange(2 * pi + self.alpha + pi / 6 + 6 * pi / 3, 2 * pi + self.alpha + pi / 6 + 8 * pi / 3,
                               self.acc)
            range2 = np.arange(2 * pi + self.alpha + pi / 6 + 8 * pi / 3, 2 * pi + self.alpha + pi / 6 + 10 * pi / 3,
                               self.acc)
            ubc = np.frompyfunc(self._uca, 1, 1)  # 向量化
            uba = np.frompyfunc(self._ucb, 1, 1)  # 向量化
            Ut_on = np.arange(4 * pi / 3 + self.alpha + pi / 6, 6 * pi / 3 + self.alpha + pi / 6, self.acc) * 0
            Ut_ab = ubc(range1)
            Ut_ac = uba(range2)
            Ut = np.hstack((Ut_ab, Ut_ac, Ut_on, Ut_ab, Ut_ac, Ut_on, Ut_ac))
            self.plot_line_voltage(len(Ut))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 8 * pi / 3, num=len(Ut)), Ut,
                     *args,
                     **kwargs)
            plt.show()
        elif ut_num == "2":
            range1 = np.arange(2 * pi + self.alpha + pi / 6 + 3 * pi / 3, 2 * pi + self.alpha + pi / 6 + 5 * pi / 3,
                               self.acc)
            range2 = np.arange(2 * pi + self.alpha + pi / 6 + 5 * pi / 3, 2 * pi + self.alpha + pi / 6 + 7 * pi / 3,
                               self.acc)
            uab = np.frompyfunc(self._uca, 1, 1)  # 向量化
            uac = np.frompyfunc(self._ucb, 1, 1)  # 向量化
            Ut_on = np.arange(self.alpha + pi / 6 + 2 * pi / 3, 4 * pi / 3 + self.alpha + pi / 6, self.acc) * 0
            Ut_ab = uab(range1)
            Ut_ac = uac(range2)
            Ut = np.hstack((Ut_ac, Ut_on, Ut_ab, Ut_ac, Ut_on, Ut_ab, Ut_ac))
            self.plot_line_voltage(len(Ut))
            plt.plot(np.linspace(self.alpha - pi / 6, 2 * pi + self.alpha + pi / 6 + 7 * pi / 3, num=len(Ut)), Ut,
                     *args,
                     **kwargs)
            plt.show()
        elif ut_num == "6":
            ...

    def plot_err(self, err_device, *args, **kwargs):
        """绘制故障电压"""
        plt.cla()
        uab = np.frompyfunc(self._uab, 1, 1)  # 向量化
        uac = np.frompyfunc(self._uac, 1, 1)  # 向量化
        ubc = np.frompyfunc(self._ubc, 1, 1)  # 向量化
        uba = np.frompyfunc(self._uba, 1, 1)  # 向量化
        uca = np.frompyfunc(self._uca, 1, 1)  # 向量化
        ucb = np.frompyfunc(self._ucb, 1, 1)  # 向量化
        range1 = np.arange(self.alpha + pi / 6, pi / 3 + self.alpha + pi / 6, self.acc)  # 第一部分
        range2 = np.arange(pi / 3 + self.alpha + pi / 6, 2 * pi / 3 + self.alpha + pi / 6, self.acc)  # 第2部分
        range3 = np.arange(2 * pi / 3 + self.alpha + pi / 6, 3 * pi / 3 + self.alpha + pi / 6, self.acc)  # 第3部分
        range4 = np.arange(3 * pi / 3 + self.alpha + pi / 6, 4 * pi / 3 + self.alpha + pi / 6, self.acc)  # 第4部分
        range5 = np.arange(4 * pi / 3 + self.alpha + pi / 6, 5 * pi / 3 + self.alpha + pi / 6, self.acc)  # 第5部分
        range6 = np.arange(5 * pi / 3 + self.alpha + pi / 6, 6 * pi / 3 + self.alpha + pi / 6, self.acc)  # 第6部分
        if err_device == "2":
            plt.title("Ud  alpha = {}".format((self.alpha / np.pi) * 180) + "° " + "   arm{} broken    ".format(
                err_device) + " load type:{}".format(self.load_type))
            range1 = np.arange(self.alpha + pi / 6, 2 * pi / 3 + self.alpha + pi / 6, self.acc)  # 第一部分
            range2 = np.arange(2 * pi / 3 + self.alpha + pi / 6, pi + self.alpha + pi / 6, self.acc) * 0  # 第2部分
            range3 = np.arange(pi + self.alpha + pi / 6, pi + self.alpha + pi / 6 + 1 * pi / 3, self.acc)  # 第3部分
            range4 = np.arange(pi + self.alpha + pi / 6 + 1 * pi / 3, pi + self.alpha + pi / 6 + 2 * pi / 3,
                               self.acc)  # 第4部分
            range5 = np.arange(pi + self.alpha + pi / 6 + 2 * pi / 3, pi + self.alpha + pi / 6 + 3 * pi / 3,
                               self.acc)  # 第5部分
            U_err = np.hstack((uab(range1), range2, uba(range3), uca(range4), ucb(range5), uab(range1), range2,
                               uba(range3), uca(range4), ucb(range5)))  # 拼接故障电压
            if self.load_type == "R":
                U_err = np.frompyfunc(self._relu, 1, 1)(U_err)  # 向量化
            self.plot_line_voltage(len(U_err))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(U_err)), U_err,
                     *args, **kwargs)
        elif err_device == "1":
            plt.title("Ud  alpha = {}".format((self.alpha / np.pi) * 180) + "° " + "   arm{} broken    ".format(
                err_device) + " load type:{}".format(self.load_type))

            U_err = np.hstack((ucb(range1), 0 * range2, ubc(range3), uba(range4), uca(range5), ucb(range6), ucb(range1),
                               0 * range2, ubc(range3), uba(range4), uca(range5), ucb(range6)))  # 拼接故障电压
            if self.load_type == "R":
                U_err = np.frompyfunc(self._relu, 1, 1)(U_err)  # 向量化
            self.plot_line_voltage(len(U_err))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(U_err)), U_err,
                     *args, **kwargs)
        elif err_device == "3":
            plt.title("Ud  alpha = {}".format((self.alpha / np.pi) * 180) + "° " + "   arm{} broken    ".format(
                err_device) + " load type:{}".format(self.load_type))

            U_err = np.hstack((uab(range1), uac(range2), uac(range3), range4 * 0, uca(range5), ucb(range6),
                               uab(range1), uac(range2), uac(range3), range4 * 0, uca(range5), ucb(range6)))  # 拼接故障电压
            if self.load_type == "R":
                U_err = np.frompyfunc(self._relu, 1, 1)(U_err)  # 向量化
            self.plot_line_voltage(len(U_err))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(U_err)), U_err,
                     *args, **kwargs)
        elif err_device == "4":
            plt.title("Ud  alpha = {}".format((self.alpha / np.pi) * 180) + "° " + "   arm{} broken    ".format(
                err_device) + " load type:{}".format(self.load_type))

            U_err = np.hstack((uab(range1), uac(range2), ubc(range3), ubc(range4), 0 * range5, ucb(range6),
                               uab(range1), uac(range2), ubc(range3), ubc(range4), 0 * range5, ucb(range6)))  # 拼接故障电压
            if self.load_type == "R":
                U_err = np.frompyfunc(self._relu, 1, 1)(U_err)  # 向量化
            self.plot_line_voltage(len(U_err))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(U_err)), U_err,
                     *args, **kwargs)
        elif err_device == "5":
            plt.title("Ud  alpha = {}".format((self.alpha / np.pi) * 180) + "° " + "   arm{} broken    ".format(
                err_device) + " load type:{}".format(self.load_type))

            U_err = np.hstack((uab(range1), uac(range2), ubc(range3), uba(range4), uba(range5), 0 * range6,
                               uab(range1), uac(range2), ubc(range3), uba(range4), uba(range5), 0 * range6))  # 拼接故障电压
            if self.load_type == "R":
                U_err = np.frompyfunc(self._relu, 1, 1)(U_err)  # 向量化
            self.plot_line_voltage(len(U_err))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(U_err)), U_err,
                     *args, **kwargs)
        elif err_device == "6":
            plt.title("Ud  alpha = {}".format((self.alpha / np.pi) * 180) + "° " + "   arm{} broken    ".format(
                err_device) + " load type:{}".format(self.load_type))

            U_err = np.hstack((range1 * 0, uac(range2), ubc(range3), uba(range4), uca(range5), uca(range6),
                               range1 * 0, uac(range2), ubc(range3), uba(range4), uca(range5), uca(range6)))  # 拼接故障电压
            if self.load_type == "R":
                U_err = np.frompyfunc(self._relu, 1, 1)(U_err)  # 向量化
            self.plot_line_voltage(len(U_err))
            plt.plot(np.linspace(self.alpha + pi / 6, 2 * pi + self.alpha + pi / 6 + 6 * pi / 3, num=len(U_err)), U_err,
                     *args, **kwargs)
        plt.show()

    def plot_phase_voltage(self):
        plt.cla()
        """
        绘制线电压
        :return:
        """

        def draw_other_degree(degree, U2=1):
            degree = degree + np.pi / 6  # 开始的角度
            a = np.sin(np.linspace(0, np.pi * 4, num=200))
            b = np.sin(np.linspace(0, np.pi * 4, num=200) + 2 / 3 * np.pi)
            c = np.sin(np.linspace(0, np.pi * 4, num=200) - 2 / 3 * np.pi)
            ua_upper = np.sin(np.linspace(degree, degree + np.pi * 2 / 3, num=200))  # 正半轴A相电压
            ub_upper = np.sin(
                np.linspace(degree + np.pi * 2 / 3, degree + np.pi * 4 / 3, num=200) - (2 / 3 * np.pi))  # 正半轴B相电压
            uc_upper = np.sin(
                np.linspace(degree + np.pi * 4 / 3, degree + np.pi * 6 / 3, num=200) + (2 / 3 * np.pi))  # 正半轴C相电压
            uc_upper_init = np.sin(
                np.linspace(0, degree, num=200) + (2 / 3 * np.pi))  # 正半轴C相电压起始值
            ua_lower = -np.sin(np.linspace(degree, degree + np.pi * 2 / 3, num=200))  # 正半轴A相电压
            ub_lower = -np.sin(
                np.linspace(degree + np.pi * 2 / 3, degree + np.pi * 4 / 3, num=200) - (2 / 3 * np.pi))  # 正半轴B相电压
            uc_lower = -np.sin(
                np.linspace(degree + np.pi * 4 / 3, degree + np.pi * 6 / 3, num=200) + (2 / 3 * np.pi))  # 正半轴C相电压

            ua_upper = np.append(ua_upper, ub_upper[0])
            ub_upper = np.append(ub_upper, uc_upper[0])
            uc_upper = np.append(uc_upper, ua_upper[0])
            uc_upper_init = np.append(uc_upper_init, ua_upper[0])
            ua_lower = np.append(ua_lower, ub_lower[0])
            ub_lower = np.append(ub_lower, uc_lower[0])
            uc_lower = np.append(ub_lower, ua_lower[0])
            plt.title("alpha = {}".format((degree / np.pi) * 180 - 30) + "°")
            plt.plot(np.linspace(0, degree, num=len(uc_upper_init)), U2 * uc_upper_init, color="red")
            plt.plot(np.linspace(degree, degree + 2 * np.pi / 3, num=len(ua_upper)), U2 * ua_upper, color="red")
            plt.plot(np.linspace(0, degree, num=len(uc_upper_init)), uc_upper_init, color="red")
            plt.plot(np.linspace(degree + 2 * np.pi / 3, degree + 4 * np.pi / 3, num=len(ub_upper)), U2 * ub_upper,
                     color="red")
            plt.plot(np.linspace(degree + 4 * np.pi / 3, degree + 6 * np.pi / 3, num=len(uc_upper)), U2 * uc_upper,
                     color="red")
            plt.plot(np.linspace(degree + 6 * np.pi / 3, degree + 8 * np.pi / 3, num=len(ua_upper)), U2 * ua_upper,
                     color="red")
            plt.plot(np.linspace(degree + 8 * np.pi / 3, degree + 10 * np.pi / 3, num=len(ub_upper)), U2 * ub_upper,
                     color="red")
            plt.plot(np.linspace(degree + 10 * np.pi / 3, degree + 12 * np.pi / 3, num=len(uc_upper)), U2 * uc_upper,
                     color="red")
            plt.plot(np.linspace(degree - np.pi * 2 / 3 + np.pi / 3, degree + np.pi / 3, num=len(ub_lower)),
                     U2 * ub_lower, color="red")
            plt.plot(np.linspace(degree + np.pi / 3, degree + np.pi * 2 / 3 + np.pi / 3, num=len(ua_lower)),
                     U2 * ua_lower,
                     color="red")
            plt.plot(
                np.linspace(degree + np.pi * 2 / 3 + np.pi / 3, degree + np.pi * 4 / 3 + np.pi / 3, num=len(ub_lower)),
                U2 * ub_lower, color="red")
            plt.plot(
                np.linspace(degree + np.pi * 4 / 3 + np.pi / 3, degree + np.pi * 6 / 3 + np.pi / 3, num=len(uc_lower)),
                U2 * uc_lower, color="red")
            plt.plot(
                np.linspace(degree + np.pi / 3 + np.pi * 6 / 3, degree + np.pi * 8 / 3 + np.pi / 3, num=len(ua_lower)),
                U2 * ua_lower,
                color="red")
            plt.plot(
                np.linspace(degree + np.pi * 8 / 3 + np.pi / 3, degree + np.pi * 10 / 3 + np.pi / 3, num=len(ub_lower)),
                U2 * ub_lower, color="red")
            plt.plot(np.linspace(0, 4 * np.pi, num=len(a)), U2 * a, linestyle="--", color="blue")
            plt.plot(np.linspace(0, 4 * np.pi, num=len(a)), U2 * b, linestyle="--", color="blue")
            plt.plot(np.linspace(0, 4 * np.pi, num=len(a)), U2 * c, linestyle="--", color="blue")
            plt.show()

        draw_other_degree(degree=self.alpha, U2=self.U, load="RL")


if __name__ == '__main__':
        a = full_bridge(1, alpha=pi / 3, load_type="R")
        a.plot_Ud(color="red")
        # a.plot_err("2", color="red")
        # a.plot_UT("2", color="red")
