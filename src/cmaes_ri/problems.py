"""Classic continuous test functions, each with global minimum value 0."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def sphere(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20.0 + np.e)


def griewank(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, x.size + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


@dataclass(frozen=True)
class Problem:
    name: str
    func: Callable[[np.ndarray], float]
    bound: float  # symmetric domain [-bound, bound]
    modality: str  # "unimodal" | "multimodal"
    x_opt_component: float  # value of each coordinate at the global minimizer

    @property
    def L(self) -> float:
        return self.bound


PROBLEMS: dict[str, Problem] = {
    "sphere": Problem("sphere", sphere, 5.12, "unimodal", 0.0),
    "rosenbrock": Problem("rosenbrock", rosenbrock, 2.048, "unimodal", 1.0),
    "rastrigin": Problem("rastrigin", rastrigin, 5.12, "multimodal", 0.0),
    "ackley": Problem("ackley", ackley, 32.768, "multimodal", 0.0),
    "griewank": Problem("griewank", griewank, 600.0, "multimodal", 0.0),
}


def get_problem(name: str) -> Problem:
    return PROBLEMS[name]
