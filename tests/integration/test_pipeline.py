"""Integration tests for the full pipeline."""

from src.core.config import PATH_CONFIG


class TestPipeline:
    """Integration tests for the essay scoring pipeline."""

    def test_config_paths_exist(self) -> None:
        """Test that configured paths can be created."""
        PATH_CONFIG.ensure_dirs()
        assert PATH_CONFIG.data_dir.exists()
        assert PATH_CONFIG.models_dir.exists()
