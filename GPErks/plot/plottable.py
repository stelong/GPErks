from abc import ABCMeta, abstractmethod

from GPErks.plot.options import PlotOptions


class Plottable(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def plot(self, plot_options: PlotOptions = PlotOptions()):
        pass
