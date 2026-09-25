"""Dependency-free input parsing shared by the graph-training entry points."""


def parse_tgf(path):
    """Read the unlabelled TGF subset used by the argumentation datasets.

    Preserve argument/attack order, ignore blank lines, and reject malformed
    records before NetworkX can silently create undeclared arguments.
    """
    arguments, attacks = [], []
    declared = set()
    seen_separator = False
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            context = f"{path}:{line_number}: "
            if line == "#":
                if seen_separator:
                    raise ValueError(context + "duplicate TGF separator")
                seen_separator = True
                continue
            fields = line.split()
            if not seen_separator:
                if len(fields) != 1:
                    raise ValueError(context + "expected one unlabelled argument identifier")
                if line in declared:
                    raise ValueError(context + "duplicate argument identifier")
                arguments.append(line)
                declared.add(line)
            else:
                if len(fields) != 2:
                    raise ValueError(context + "expected two attack endpoints")
                if any(endpoint not in declared for endpoint in fields):
                    raise ValueError(context + "attack endpoint is not a declared argument")
                attacks.append(tuple(fields))
    if not seen_separator:
        raise ValueError(f"{path}: missing TGF '#' separator")
    return arguments, attacks
