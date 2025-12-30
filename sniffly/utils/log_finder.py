"""
Utility to find Claude logs for a given project path
"""

import logging
import os
from pathlib import Path

from sniffly.config import get_claude_projects_dir

logger = logging.getLogger(__name__)


def find_claude_logs(project_path: str) -> str | None:
    """
    Find Claude logs for a given project path.

    Claude stores logs at ~/.claude/projects/[converted-project-path]/
    where the project path has slashes replaced with dashes and starts with a dash.

    Example:
        /Users/john/dev/myapp -> ~/.claude/projects/-Users-john-dev-myapp/

    Args:
        project_path: The project directory path

    Returns:
        Path to Claude logs directory if found, None otherwise
    """
    # Normalize the project path
    project_path = os.path.abspath(project_path)

    # Remove trailing slash if present
    if project_path.endswith("/"):
        project_path = project_path[:-1]

    # Convert to Claude log format
    # Replace all / with -
    converted_path = project_path.replace("/", "-")

    # Construct the Claude log path
    claude_base = get_claude_projects_dir()
    log_path = claude_base / converted_path

    # Check if it exists
    if log_path.exists() and log_path.is_dir():
        # Verify it contains JSONL files
        jsonl_files = list(log_path.glob("*.jsonl"))
        if jsonl_files:
            return str(log_path)

    # Try without leading dash (older format)
    if converted_path.startswith("-"):
        alt_path = claude_base / converted_path[1:]
        if alt_path.exists() and alt_path.is_dir():
            jsonl_files = list(alt_path.glob("*.jsonl"))
            if jsonl_files:
                return str(alt_path)

    return None


def list_all_claude_projects() -> list:
    """
    List all Claude projects found on the system.

    Returns:
        List of tuples (project_path, log_path)
    """
    projects = []
    claude_base = get_claude_projects_dir()

    if not claude_base.exists():
        return projects

    for log_dir in claude_base.iterdir():
        if log_dir.is_dir():
            # Convert back from log format to project path
            dir_name = log_dir.name

            # Handle leading dash
            if dir_name.startswith("-"):
                project_path = "/" + dir_name[1:].replace("-", "/")
            else:
                project_path = dir_name.replace("-", "/")

            # Verify it has JSONL files
            jsonl_files = list(log_dir.glob("*.jsonl"))
            if jsonl_files:
                projects.append((project_path, str(log_dir)))

    return projects


def validate_project_path(project_path: str) -> tuple[bool, str]:
    """
    Validate a project path and return status with message.

    Returns:
        (is_valid, message)
    """
    if not project_path:
        return False, "Project path cannot be empty"

    if not os.path.exists(project_path):
        return False, f"Project path does not exist: {project_path}"

    if not os.path.isdir(project_path):
        return False, f"Project path must be a directory: {project_path}"

    # Check if logs exist
    log_path = find_claude_logs(project_path)
    if not log_path:
        return False, f"No Claude logs found for project: {project_path}"

    return True, f"Found logs at: {log_path}"


def get_all_projects_with_metadata() -> list[dict]:
    """
    Get all Claude projects with metadata for fast display.

    Returns metadata without reading file contents for performance.

    Returns:
        List of dictionaries containing:
        - dir_name: Directory name in .claude/projects
        - log_path: Full path to log directory
        - file_count: Number of JSONL files
        - total_size_mb: Total size of JSONL files in MB
        - last_modified: Unix timestamp of most recent modification
        - first_seen: Unix timestamp of earliest file (approximation of first use)
        - display_name: Human-readable project name
    """
    projects = []
    claude_base = get_claude_projects_dir()

    if not claude_base.exists():
        return projects

    try:
        for log_dir in claude_base.iterdir():
            if log_dir.is_dir():
                jsonl_files = list(log_dir.glob("*.jsonl"))
                if jsonl_files:
                    # Get metadata without reading file contents
                    total_size = sum(f.stat().st_size for f in jsonl_files)

                    # Get modification times
                    mtimes = [f.stat().st_mtime for f in jsonl_files]
                    latest_mtime = max(mtimes)
                    earliest_mtime = min(mtimes)

                    # Use directory name as display name
                    # Don't convert dashes to slashes as we can't distinguish
                    # between dashes that were originally in the name vs path separators
                    dir_name = log_dir.name
                    display_name = dir_name

                    projects.append(
                        {
                            "dir_name": dir_name,
                            "log_path": str(log_dir),
                            "file_count": len(jsonl_files),
                            "total_size_mb": round(total_size / (1024 * 1024), 2),
                            "last_modified": latest_mtime,
                            "first_seen": earliest_mtime,
                            "display_name": display_name,
                        }
                    )
    except Exception as e:
        # Log error but continue - don't fail completely
        logger.info(f"Error reading project metadata: {e}")

    return projects
