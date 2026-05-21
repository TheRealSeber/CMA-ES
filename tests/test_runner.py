from cmaes_ri.runner import run_single
from cmaes_ri.seeds import make_seed_list


def test_sphere_converges():
    rec = run_single(
        "sphere",
        dim=2,
        sigma0_factor=1.0,
        variant="baseline",
        rep=0,
        seed=make_seed_list(1)[0],
        maxfevals=2000,
    )
    assert rec.final_fbest < 1e-2
    assert len(rec.evals) == len(rec.fbest) == len(rec.sigma) == len(rec.ps_norm)


def test_same_seed_same_run():
    seed = make_seed_list(1)[0]
    a = run_single("sphere", 3, 1.0, "baseline", 0, seed, maxfevals=1500)
    b = run_single("sphere", 3, 1.0, "baseline", 0, seed, maxfevals=1500)
    assert a.fbest == b.fbest
    assert a.total_evals == b.total_evals


def test_same_seed_same_first_gen_across_variants():
    seed = make_seed_list(1)[0]
    a = run_single("sphere", 3, 1.0, "baseline", 0, seed, maxfevals=200)
    b = run_single("sphere", 3, 1.0, "ri", 0, seed, maxfevals=200)
    # same m0 + same pycma seed => identical evaluation budget at first generation
    assert a.evals[0] == b.evals[0]
    # but p_sigma init differs => trajectories diverge in sigma at some point
    assert a.sigma != b.sigma or a.ps_norm[0] != b.ps_norm[0]
