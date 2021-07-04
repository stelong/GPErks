from abc import ABCMeta, abstractmethod

from GPErks.plot.options import PlotOptions


class Plottable(metaclass=ABCMeta):
    def __init__(self, options: PlotOptions):
        self.plot_options: options

    @abstractmethod
    def plot(self):
        pass
