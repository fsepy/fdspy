# -*- coding: utf-8 -*-

from asciiplot import *


def test_asciiplot():
    size = (80, 25)
    xlim = (-2 * np.pi, 2 * np.pi)
    ylim = (-1.1, 1.1)

    aplot = AsciiPlot(size=size)

    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    aplot.plot(
        x=x,
        y=np.sin(x),
        xlim=xlim,
        ylim=ylim,
    ).show()


if __name__ == '__main__':
    test_asciiplot()
