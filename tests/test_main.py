"""Tests for main.py, the command-line entry point."""

import pytest

from main import ORDER_FINDERS, describe_order, main
from shor.classical import classical_order


def test_factors_989(capsys):
    assert main(["989", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Factoring N = 989 (n = 10 bits)" in out
    assert out.strip().endswith("989 = 23 × 43")


@pytest.mark.parametrize("seed", range(20))
def test_factors_15_for_any_seed(seed, capsys):
    assert main(["15", "--seed", str(seed)]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("15 = 3 × 5")
    # Either an order was found and shown, or the base was lucky; never neither.
    assert ("order r =" in out) != ("no order finding needed" in out)


def test_runs_without_a_seed(capsys):
    assert main(["21"]) == 0
    assert capsys.readouterr().out.strip().endswith("21 = 3 × 7")


@pytest.mark.parametrize("N", ["13", "16", "27", "1"])
def test_invalid_N_exits_with_usage_error(N, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([N])
    assert exit_info.value.code == 2
    assert f"N = {N}" in capsys.readouterr().err


def test_unknown_method_is_rejected():
    with pytest.raises(SystemExit):
        main(["15", "--method", "nonsense"])


def test_every_registered_finder_is_callable_as_find_order():
    assert set(ORDER_FINDERS) == {"classical", "n15"}
    for make_finder, description in ORDER_FINDERS.values():
        find_order = make_finder(3)
        assert find_order(7, 15) == 4
        assert description


@pytest.mark.parametrize("seed", range(6))
def test_n15_method_factors_15(seed, capsys):
    assert main(["15", "--method", "n15", "--seed", str(seed)]) == 0
    out = capsys.readouterr().out
    assert "Order finder: n15" in out
    assert out.strip().endswith("15 = 3 × 5")


def test_n15_method_is_repeatable_with_a_seed(capsys):
    main(["15", "--method", "n15", "--seed", "4"])
    first = capsys.readouterr().out
    main(["15", "--method", "n15", "--seed", "4"])
    assert capsys.readouterr().out == first


def test_n15_method_rejects_other_N(capsys):
    # Checked up front: a lucky base would otherwise factor 21 without ever
    # reaching the order finder that cannot handle it.
    with pytest.raises(SystemExit) as exit_info:
        main(["21", "--method", "n15"])
    assert exit_info.value.code == 2
    assert "only factors N = 15" in capsys.readouterr().err


def test_describe_order_cases():
    assert describe_order(7, classical_order(7, 15), 15) == (
        "order r = 4, a^(r/2) ≡ 4 (mod N), gcd(4 ± 1, N) = 3 and 5"
    )
    assert "odd" in describe_order(4, 3, 21)  # 4 has order 3 mod 21
    assert "≡ −1" in describe_order(5, 6, 21)  # 5^3 = 125 = -1 mod 21
    assert "failed" in describe_order(7, None, 15)
    assert "≢ 1" in describe_order(7, 5, 15)
    assert "no factor" in describe_order(7, 8, 15)  # even multiple of the order
