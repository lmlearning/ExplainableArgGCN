import pytest

from graph_io import parse_extension_union


@pytest.mark.parametrize("text,expected", [
    ("[[a,b],[c]]", {"a", "b", "c"}),
    (" [ [ 1, 2 ], [ 2, 3 ] ] \n", {"1", "2", "3"}),
    ("[a,b]", {"a", "b"}),
    ("[[]]", set()), ("[]", set()), ("[[first]]", {"first"}),
])
def test_extension_union_preserves_first_argument(text, expected):
    assert parse_extension_union(text) == expected


@pytest.mark.parametrize("text", ["", "a,b", "[[a]", "[[a],garbage]", "[[a,,b]]"])
def test_malformed_solution_is_rejected(text):
    with pytest.raises(ValueError):
        parse_extension_union(text)
