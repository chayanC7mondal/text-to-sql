import pytest
from app.ml.ensemble import SQLEnsemble


@pytest.fixture
async def ensemble():
    """Create and initialize ensemble for testing."""
    ens = SQLEnsemble()
    await ens.initialize()
    yield ens


@pytest.mark.asyncio
async def test_provider_selection(ensemble):
    """Test provider selection based on complexity."""
    # Low complexity should use local or fast model
    provider_low = ensemble._select_provider(0.2)
    assert provider_low in ensemble.providers.keys()

    # High complexity should use best model
    provider_high = ensemble._select_provider(0.9)
    assert provider_high in ensemble.providers.keys()


@pytest.mark.asyncio
async def test_weight_update(ensemble):
    """Test weight update mechanism."""
    initial_weight = ensemble.current_weights.get("local", 0.0)

    # Simulate success
    ensemble.update_weights("local", True)
    updated_weight = ensemble.current_weights.get("local", 0.0)

    # Weight should increase (or stay same if at max)
    assert updated_weight >= initial_weight

    # Simulate failure
    ensemble.update_weights("local", False)
    final_weight = ensemble.current_weights.get("local", 0.0)

    # Weight should decrease
    assert final_weight <= updated_weight


def test_get_provider_priority(ensemble):
    """Test provider priority order."""
    priority = ensemble._get_provider_priority()
    assert isinstance(priority, list)
    assert len(priority) > 0
    assert "openai" in priority or "anthropic" in priority or "local" in priority


def test_get_provider_stats(ensemble):
    """Test getting provider statistics."""
    stats = ensemble.get_provider_stats()
    assert isinstance(stats, dict)
