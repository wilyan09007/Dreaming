import numpy as np
from dreaming.gate import certify, confidence, cp_upper, realized


def test_cp_upper_matches_worked_example():
    # certification-deferral.md: n=500, budget 5%, delta 5% -> at most 16 errors certify
    assert cp_upper(16, 500, 0.05) <= 0.05 < cp_upper(17, 500, 0.05)
    assert cp_upper(0, 0, 0.05) == 1.0


def test_certificate_holds_on_fresh_rows():
    """Scores correlate with correctness; the cutoff found on calibration rows keeps the realized error under budget on new rows."""
    rng = np.random.default_rng(1)

    def draw(n: int):
        conf = rng.random(n)
        correct = rng.random(n) < np.where(conf > 0.5, 0.99, 0.6)  # confident rows are right 99% of the time
        return conf, correct

    conf, correct = draw(2000)
    g = certify(conf, correct, budget=0.05)
    assert g.certified and 0 < g.coverage < 1 and g.bound <= 0.05
    cov, err = realized(*draw(20000), g.cutoff)
    assert err <= 0.05 and abs(cov - g.coverage) < 0.1


def test_nothing_certifies_when_errors_are_uniform():
    rng = np.random.default_rng(2)
    g = certify(rng.random(300), rng.random(300) > 0.2, budget=0.05)
    assert not g.certified


def test_confidence_is_quantized_top2_gap():
    c = confidence(np.array([[0.7, 0.2, 0.1], [0.5, 0.49999999, 0.0]]))
    assert c.tolist() == [0.5, 0.0]


def test_constant_confidence_pays_delta_once():
    """A router's predictor has one confidence level: the gate tests one event, not forty (B2)."""
    conf = np.ones(500)
    correct = np.ones(500, dtype=bool)
    correct[:23] = False
    g = certify(conf, correct, budget=0.07)
    assert g.certified and g.coverage == 1.0 and g.total == 500 and g.n == 500
    assert g.certified and abs(g.bound - cp_upper(23, 500, 0.05)) < 1e-12
    assert not certify(conf, correct, budget=0.064).certified and certify(conf, correct, budget=0.065).certified


def test_empty_calibration_is_uncertified_not_an_error():
    g = certify(np.zeros(0), np.zeros(0, dtype=bool), budget=0.05)
    assert not g.certified and g.total == 0


def test_audits_needed_before_a_revocation():
    from dreaming.gate import audits_needed

    assert audits_needed(0.05, 0.05) == 59 and cp_upper(0, 58, 0.05) > 0.05 >= cp_upper(0, 59, 0.05)
