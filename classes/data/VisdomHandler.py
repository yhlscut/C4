from __future__ import print_function

import numpy as np
import visdom


class VisdomHandler:

    def __init__(self, port: int, env: str):
        self.__vis = visdom.Visdom(port=port, env=env)
        self.__win_curve = self.__vis.line(X=np.array([0]), Y=np.array([0]))

    def update(self, epoch: int, loss: float, name: str):
        try:
            self.__vis.updateTrace(
                X=np.array([epoch]),
                Y=np.array([loss]),
                win=self.__win_curve,
                name=name
            )
        except:
            print('Visdom error!')
