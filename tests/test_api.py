import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_reports_ablation_flags(client):
    """The eval harness records these to verify a run's --config label against the configuration
    it was actually collected under. A `full` run was once collected with router_mode=classifier
    — the `full-clf` configuration — and nothing in the output revealed it."""
    data = (await client.get("/health")).json()
    flags = data["flags"]
    assert set(flags) == {"router_mode", "router_model", "decompose_enabled", "critic_mode"}
    assert isinstance(flags["decompose_enabled"], bool)
    assert flags["router_model"], "router_model must resolve to a concrete model name"
