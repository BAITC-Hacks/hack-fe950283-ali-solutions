"""Согласование числительных в обоснованиях."""
import pytest

from moneygraph.fmt import PAYERS, PAYERS_GEN, plural


@pytest.mark.parametrize("n, expected", [
    (1, "1 плательщик"), (2, "2 плательщика"), (5, "5 плательщиков"), (11, "11 плательщиков"),
    (14, "14 плательщиков"), (21, "21 плательщик"), (24, "24 плательщика"), (111, "111 плательщиков"), (0, "0 плательщиков"),
])
def test_plural_nominative(n, expected):
    assert plural(n, PAYERS) == expected


def test_plural_genitive_after_preposition():
    assert f"от {plural(1, PAYERS_GEN)}" == "от 1 плательщика"
    assert f"от {plural(3, PAYERS_GEN)}" == "от 3 плательщиков"
