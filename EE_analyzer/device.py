# > IGBT类是一个Python类，代表一个IGBT设备
# 这是一个代表IGBT的类
from numpy import pi


# > 类`IGBT`是代表一个IGBT的类
class IGBT(object):
    def __init__(self, on_time=None, name="1"):
        if on_time is None:
            on_time = [0, 2 * pi]
        self.on_time = [on_time[0], on_time[1]]
        self.now_angle = 0
        self.name = name
        self.is_break = False  # 是否能正常工作或断开

    def set_on_time(self, on_time):
        """
        此函数将对象的 on_time 属性设置为 on_time 参数的值

        :param on_time: 灯亮的时间。
        """
        self.on_time = on_time

    def set_angle(self, angle):
        """
        *|MARKER_CURSOR|*

        :param angle: IGBT当前电角度。
        """

        self.now_angle = angle

    @property
    def is_on(self):
        """
        如果当前时间开始和结束时间之间，则返回True，否则返回False
        :return: 函数 is_on 返回一个布尔值。
        """
        if self.is_break:
            return False
        if self.on_time[0] < self.on_time[1]:
            if self.on_time[0] <= self.now_angle <= self.on_time[1]:
                return True
            else:
                return False
        else:
            if self.now_angle <= self.on_time[0] or self.now_angle >= self.on_time[1]:
                return True
            else:
                return False

    def __repr__(self):
        return self.name + ":" + str(self.on_time)

    def destroy(self):
        """
        它将 is_break 变量设置为 True。
        """
        self.is_break = True

    def repair(self):
        """
        它修IGBT。
        """
        self.is_break = False


class DC_Motor(object):
    def __init__(self, E, L, R):
        self.E = E
        self.L = L
        self.R = R
