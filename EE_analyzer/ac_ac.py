import abc


class converter(abc.ABC):
    def plot(self, *args, **kwargs):
        pass

    def set_alpha(self, *args, **kwargs):
        pass


class boost_converter(converter):
    ...


class buck_converter(converter):
    ...
