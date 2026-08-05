"""Judge response cache.

Scoring 50 items across five metrics is ~250 judge calls against a free tier that exhausted
repeatedly during the v1.0 evaluation — twice mid-sweep, which is why two comparisons ended up at
n=30 against n=47 for reasons of quota rather than design. The cache makes a re-score free, so an
analysis bug no longer costs a scoring budget.

These tests exercise the cache wrapper directly; they never call a judge.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
from langchain_core.outputs import Generation

# eval/ is a directory of scripts, not an importable package, so load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "run_ragas", Path(__file__).resolve().parents[1] / "eval" / "run_ragas.py"
)
run_ragas = importlib.util.module_from_spec(_SPEC)
sys.modules["run_ragas"] = run_ragas
_SPEC.loader.exec_module(run_ragas)

pytestmark = pytest.mark.asyncio


def _value(text: str = "cached verdict"):
    """SQLiteCache stores one row per Generation, so an empty list writes nothing and would look
    indistinguishable from a miss. Cache a real generation."""
    return [Generation(text=text)]


@pytest.fixture
def cache(tmp_path):
    from langchain_core.globals import set_llm_cache

    c = run_ragas._install_judge_cache(tmp_path / "j.sqlite")
    yield c
    set_llm_cache(None)  # never leak a cache into another test's LLM calls


async def test_miss_then_hit_is_counted(cache):
    """The counters are the only signal that a re-score actually cost nothing, so they have to be
    right — a cache that silently reported hits it never served would be worse than none."""
    assert await cache.alookup("p", "llm") is None
    assert (cache.hits, cache.misses) == (0, 1)

    await cache.aupdate("p", "llm", _value())
    assert await cache.alookup("p", "llm") is not None
    assert (cache.hits, cache.misses) == (1, 1)


async def test_async_path_is_counted(cache):
    """RAGAS drives the judge asynchronously, so alookup is the path that actually runs. Counting
    only the sync path would report 0 hits on a fully cached run."""
    await cache.aupdate("async-prompt", "llm", _value())
    await cache.alookup("async-prompt", "llm")
    assert cache.hits == 1


async def test_key_includes_the_model_parameters(cache):
    """llm_string carries the judge model and its parameters. Changing the judge must invalidate
    the entry rather than silently reuse another model's verdict."""
    await cache.aupdate("same prompt", "judge-a", _value("judge-a said this"))

    assert await cache.alookup("same prompt", "judge-a") is not None
    assert await cache.alookup("same prompt", "judge-b") is None, (
        "a different judge must not read another judge's cached answer"
    )


async def test_key_includes_the_prompt(cache):
    """The prompt carries the question, the answer, the contexts and the metric's own template, so
    distinct scoring work must not collide."""
    await cache.aupdate("prompt one", "llm", _value())
    assert await cache.alookup("prompt two", "llm") is None


async def test_cache_persists_across_instances(tmp_path):
    """A re-score is a fresh process. If entries did not survive that, the cache would only ever
    help within a single run, which is not where the cost is."""
    from langchain_core.globals import set_llm_cache

    path = tmp_path / "persist.sqlite"
    first = run_ragas._install_judge_cache(path)
    await first.aupdate("durable", "llm", _value())

    second = run_ragas._install_judge_cache(path)
    try:
        assert await second.alookup("durable", "llm") is not None
        assert second.hits == 1
    finally:
        set_llm_cache(None)
