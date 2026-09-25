import pytest

from graph_io import parse_tgf


def parse(tmp_path, text):
    path = tmp_path / "graph.tgf"
    path.write_text(text, encoding="utf-8")
    return parse_tgf(path)


def test_whitespace_does_not_create_nodes_or_attacks(tmp_path):
    assert parse(tmp_path, "\nalpha\n\nbeta\nisolated\n # \n\nalpha\t beta\nbeta  beta\n\n") == (
        ["alpha", "beta", "isolated"], [("alpha", "beta"), ("beta", "beta")]
    )


def test_argument_order_is_preserved(tmp_path):
    assert parse(tmp_path, "10\n2\n1\n#\n2 10\n1 2\n") == (
        ["10", "2", "1"], [("2", "10"), ("1", "2")]
    )


def test_empty_framework(tmp_path):
    assert parse(tmp_path, "\n#\n") == ([], [])


@pytest.mark.parametrize("text,message", [
    ("a\n", "missing TGF"),
    ("a\n#\n#\n", "duplicate TGF"),
    ("a\na\n#\n", "duplicate argument"),
    ("a label\n#\n", "unlabelled"),
    ("a\n#\na\n", "two attack endpoints"),
    ("a\n#\na a extra\n", "two attack endpoints"),
    ("a\n#\na b\n", "not a declared argument"),
])
def test_invalid_graphs_fail_with_filename_context(tmp_path, text, message):
    with pytest.raises(ValueError, match="graph.tgf.*" + message):
        parse(tmp_path, text)
