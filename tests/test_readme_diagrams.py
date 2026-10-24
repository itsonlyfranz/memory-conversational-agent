"""The README's flow diagram is generated from the compiled graph; fail if it drifts."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("render_graph", ROOT / "scripts" / "render_graph.py")
render_graph = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render_graph)


def test_readme_flow_diagram_matches_compiled_graph():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert render_graph.flow_block() in readme, (
        "README flow diagram is stale; run `python scripts/render_graph.py`"
    )


def test_render_graph_check_passes_on_current_readme():
    assert render_graph.main(["--check"]) == 0


def test_flow_diagram_contains_every_graph_node():
    source = render_graph.render_flow_mermaid()
    for node in ("recall", "generate", "update_buffer", "extract", "store"):
        assert node in source
