"""Tests for the model module."""

from src.core.model import EssayModel


class TestEssayModel:
    """Tests for EssayModel class."""

    def test_model_init(self) -> None:
        """Test model initialization."""
        model = EssayModel()
        assert model is not None
        assert model.model is None
