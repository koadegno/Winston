import ast
from pathlib import Path


def test_all_stream_recorder_functions_have_docstrings():
    """Every production function, including private helpers, must have a docstring."""
    root = Path("scripts/stream_recorder")
    missing: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
                missing.append(f"{path.name}:{node.lineno}:{node.name}")
    assert not missing, "Missing docstrings: " + ", ".join(missing)
