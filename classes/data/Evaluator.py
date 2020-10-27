import numpy as np


class Evaluator:

    def __init__(self):
        self.__errors = []

    def add_error(self, error: float):
        self.__errors.append(error)

    def compute_metrics(self) -> tuple:
        self.__errors = sorted(self.__errors)

        median = self.__g(0.5)
        mean = np.mean(self.__errors)
        trimean = 0.25 * (self.__g(0.25) + 2 * self.__g(0.5) + self.__g(0.75))
        bst25 = np.mean(self.__errors[:int(0.25 * len(self.__errors))])
        wst25 = np.mean(self.__errors[int(0.75 * len(self.__errors)):])
        pct95 = self.__g(0.95)

        return mean, median, trimean, bst25, wst25, pct95

    def __g(self, f: float) -> float:
        return np.percentile(self.__errors, f * 100)
