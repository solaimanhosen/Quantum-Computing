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
    assert set(ORDER_FINDERS) == {"classical", "n15", "oracle"}
    for make_finder, description, supports in ORDER_FINDERS.values():
        find_order = make_finder(3)
        assert find_order(7, 15) == 4
        assert description
        assert supports(15)


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
    assert "--method n15 cannot handle N = 21" in capsys.readouterr().err


@pytest.mark.parametrize("N, factors", [(21, "3 × 7"), (35, "5 × 7"), (33, "3 × 11")])
def test_oracle_method_factors_N(N, factors, capsys):
    for seed in range(3):
        assert main([str(N), "--method", "oracle", "--seed", str(seed)]) == 0
        out = capsys.readouterr().out
        assert "Order finder: oracle" in out
        assert f"circuit: {3 * N.bit_length()} qubits" in out
        assert out.strip().endswith(f"{N} = {factors}")


def test_oracle_method_is_repeatable_with_a_seed(capsys):
    main(["21", "--method", "oracle", "--base", "2", "--seed", "1"])
    first = capsys.readouterr().out
    assert "a = 2: order r = 6" in first
    main(["21", "--method", "oracle", "--base", "2", "--seed", "1"])
    assert capsys.readouterr().out == first


@pytest.mark.parametrize("method", ["classical", "oracle"])
def test_base_option_runs_the_order_finder_on_that_base(method, capsys):
    # 2 has order 6 mod 21; 2^3 = 8; gcd(7, 21) = 7 and gcd(9, 21) = 3.
    assert main(["21", "--method", method, "--base", "2", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "a = 2: order r = 6, a^(r/2) ≡ 8 (mod N), gcd(8 ± 1, N) = 7 and 3" in out
    assert out.strip().endswith("21 = 3 × 7")


def test_base_option_with_the_n15_method(capsys):
    assert main(["15", "--method", "n15", "--base", "7", "--seed", "0"]) == 0
    assert "a = 7: order r = 4" in capsys.readouterr().out


def test_base_that_fails_is_followed_by_random_bases(capsys):
    # 4 has the odd order 3 mod 21, so the driver must move on.
    assert main(["21", "--base", "4", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "a = 4: order r = 3 is odd; trying another base" in out
    assert out.strip().endswith("21 = 3 × 7")


def test_base_that_shares_a_factor_needs_no_order_finding(capsys):
    assert main(["21", "--base", "6"]) == 0
    assert "no order finding needed" in capsys.readouterr().out


@pytest.mark.parametrize("base", ["1", "20", "0", "-3", "50"])
def test_base_out_of_range_is_rejected(base, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["21", "--base", base])
    assert exit_info.value.code == 2
    assert "--base must be between 2 and N - 2 = 19" in capsys.readouterr().err


def test_oracle_method_rejects_N_above_its_limit(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["143", "--method", "oracle"])
    assert exit_info.value.code == 2
    assert "--method oracle cannot handle N = 143" in capsys.readouterr().err


def test_invalid_N_is_reported_as_such_for_every_method(capsys):
    # 13 is within every method's range except n15; the reason given must be
    # that it is prime, not something about the method.
    for method in ("classical", "oracle"):
        with pytest.raises(SystemExit):
            main(["13", "--method", method])
        assert "prime power" in capsys.readouterr().err


def test_describe_order_cases():
    assert describe_order(7, classical_order(7, 15), 15) == (
        "order r = 4, a^(r/2) ≡ 4 (mod N), gcd(4 ± 1, N) = 3 and 5"
    )
    assert "odd" in describe_order(4, 3, 21)  # 4 has order 3 mod 21
    assert "≡ −1" in describe_order(5, 6, 21)  # 5^3 = 125 = -1 mod 21
    assert "failed" in describe_order(7, None, 15)
    assert "≢ 1" in describe_order(7, 5, 15)
    assert "no factor" in describe_order(7, 8, 15)  # even multiple of the order
