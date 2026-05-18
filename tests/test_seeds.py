import numpy as np

from cmaes_ri.seeds import derive_streams, make_seed_list


def test_seed_list_is_deterministic_and_sized():
    a = make_seed_list(31)
    b = make_seed_list(31)
    assert a == b
    assert len(a) == 31
    assert len(set(a)) == 31


def test_streams_independent_of_variant_for_mean_and_pycma():
    base = 12345
    s_base = derive_streams(base, variant="baseline")
    s_ri = derive_streams(base, variant="ri")
    m_base = s_base.mean_rng.standard_normal(4)
    m_ri = s_ri.mean_rng.standard_normal(4)
    assert np.array_equal(m_base, m_ri)
    assert s_base.pycma_seed == s_ri.pycma_seed


def test_ri_perturbation_stream_is_usable():
    s = derive_streams(777, variant="ri")
    z = s.perturb_rng.standard_normal((3, 4))
    assert z.shape == (3, 4)
