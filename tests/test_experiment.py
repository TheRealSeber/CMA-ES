from cmaes_ri.experiment import build_matrix, run_matrix


def test_matrix_full_size():
    m = build_matrix()
    assert len(m) == 5 * 3 * 4 * 2 * 31 == 3720


def test_small_matrix_runs_and_resumes(tmp_path):
    funcs = ["sphere"]
    dims = [2]
    factors = [1.0]
    reps = 2
    out = tmp_path / "out"
    summary1 = run_matrix(
        out, functions=funcs, dims=dims, factors=factors, reps=reps, maxfevals=800, n_jobs=1
    )
    assert len(summary1) == 1 * 1 * 1 * 2 * 2  # 4 runs
    summary2 = run_matrix(
        out, functions=funcs, dims=dims, factors=factors, reps=reps, maxfevals=800, n_jobs=1
    )
    assert len(summary2) == len(summary1)
