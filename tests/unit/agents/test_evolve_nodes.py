import pytest

from fedotllm.agents.evolve import nodes


def test_explore_is_not_silently_routed_to_qa():
    with pytest.raises(ValueError, match="research-only"):
        nodes._wants_evolve({"evolve_mode": "explore", "messages": []})


def test_default_repo_cache_is_independent_of_working_directory():
    assert nodes.REPO_CACHE.is_absolute()
