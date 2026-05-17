import numpy as np
import pytest

from cmaes_ri.problems import PROBLEMS, get_problem


@pytest.mark.parametrize("name", ["sphere", "rosenbrock", "rastrigin", "ackley", "griewank"])
def test_optimum_is_zero(name):
    prob = get_problem(name)
    x_opt = np.full(5, prob.x_opt_component)
    assert prob.func(x_opt) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_off_optimum_is_positive(name):
    prob = get_problem(name)
    x = np.full(5, prob.x_opt_component) + 0.3
    assert prob.func(x) > 0.0


def test_domain_symmetric_and_L():
    prob = get_problem("sphere")
    assert prob.bound == 5.12
    assert prob.L == 5.12
    assert prob.modality in {"unimodal", "multimodal"}
