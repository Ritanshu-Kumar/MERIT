from merit.research.m8_policy import M8Policy


def test_score_buy():
    policy = M8Policy()

    score = policy.score(
        relative_spread=0.002,
        imbalance=0.5,
        side="BUY",
    )

    assert score == 0.1


def test_score_sell():
    policy = M8Policy()

    score = policy.score(
        relative_spread=0.002,
        imbalance=0.5,
        side="SELL",
    )

    assert score == -0.1


def test_selection():
    policy = M8Policy()

    assert policy.select(
        relative_spread=0.01,
        imbalance=0.2,
        side="BUY",
    )

    assert not policy.select(
        relative_spread=0.0002,
        imbalance=0.2,
        side="BUY",
    )