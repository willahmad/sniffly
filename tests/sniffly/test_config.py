"""
Tests for configuration functions, specifically the Claude config directory.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from sniffly.config import get_claude_config_dir, get_claude_projects_dir


class TestClaudeConfigDir:
    """Test the get_claude_config_dir function."""

    def test_default_returns_home_claude(self):
        """Test that default config dir is ~/.claude."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                # Remove CLAUDE_CONFIG_DIR if it exists
                os.environ.pop("CLAUDE_CONFIG_DIR", None)

                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)

                    result = get_claude_config_dir()
                    assert result == Path(temp_dir) / ".claude"

    def test_respects_claude_config_dir_env(self):
        """Test that CLAUDE_CONFIG_DIR environment variable is respected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config_dir = Path(temp_dir) / "custom-claude-config"

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(custom_config_dir)}):
                result = get_claude_config_dir()
                assert result == custom_config_dir

    def test_env_var_takes_precedence_over_default(self):
        """Test that env var takes precedence over default ~/.claude."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config_dir = Path(temp_dir) / "my-claude-dir"

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(custom_config_dir)}):
                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path("/some/other/home")

                    result = get_claude_config_dir()
                    # Should use env var, not home
                    assert result == custom_config_dir
                    assert result != Path("/some/other/home") / ".claude"


class TestClaudeProjectsDir:
    """Test the get_claude_projects_dir function."""

    def test_default_returns_home_claude_projects(self):
        """Test that default projects dir is ~/.claude/projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("CLAUDE_CONFIG_DIR", None)

                with patch("pathlib.Path.home") as mock_home:
                    mock_home.return_value = Path(temp_dir)

                    result = get_claude_projects_dir()
                    assert result == Path(temp_dir) / ".claude" / "projects"

    def test_respects_claude_config_dir_env(self):
        """Test that CLAUDE_CONFIG_DIR environment variable affects projects dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_config_dir = Path(temp_dir) / "custom-claude-config"

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(custom_config_dir)}):
                result = get_claude_projects_dir()
                assert result == custom_config_dir / "projects"


class TestIntegrationWithLogFinder:
    """Integration tests to verify log finder uses configurable paths."""

    def test_log_finder_respects_claude_config_dir(self):
        """Test that log finder functions respect CLAUDE_CONFIG_DIR."""
        from sniffly.utils.log_finder import get_all_projects_with_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up custom config directory
            custom_config_dir = Path(temp_dir) / "my-claude-config"
            projects_dir = custom_config_dir / "projects"
            projects_dir.mkdir(parents=True)

            # Create a project
            project = projects_dir / "-Users-test-project"
            project.mkdir()
            (project / "log.jsonl").write_text('{"type": "user"}\n')

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(custom_config_dir)}):
                result = get_all_projects_with_metadata()

                assert len(result) == 1
                assert result[0]["dir_name"] == "-Users-test-project"
                assert str(projects_dir) in result[0]["log_path"]

    def test_find_claude_logs_respects_claude_config_dir(self):
        """Test that find_claude_logs respects CLAUDE_CONFIG_DIR."""
        from sniffly.utils.log_finder import find_claude_logs

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up custom config directory
            custom_config_dir = Path(temp_dir) / "my-claude-config"
            projects_dir = custom_config_dir / "projects"

            # Create the project log directory (path: /Users/test/myproject -> -Users-test-myproject)
            project_log_dir = projects_dir / "-Users-test-myproject"
            project_log_dir.mkdir(parents=True)
            (project_log_dir / "log.jsonl").write_text('{"type": "user"}\n')

            with patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(custom_config_dir)}):
                result = find_claude_logs("/Users/test/myproject")

                assert result is not None
                assert result == str(project_log_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
