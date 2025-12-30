import asyncio
import logging
from pathlib import Path

from sniffly.config import Config, get_claude_projects_dir
from sniffly.core.processor import ClaudeLogProcessor

logger = logging.getLogger(__name__)

# Get config instance
_config = Config()
cache_warm_on_startup = _config.get("cache_warm_on_startup")


# Background tasks
async def warm_recent_projects(
    cache_service, memory_cache, current_log_path, exclude_current: bool = False, limit: int = None
):
    """Preload recent projects into memory cache in background"""
    if limit is None:
        limit = cache_warm_on_startup

    try:
        # Get recent projects
        log_dirs = []
        claude_base = get_claude_projects_dir()

        if claude_base.exists():
            # Get all directories with their most recent JSONL modification times
            for d in claude_base.iterdir():
                if d.is_dir():
                    jsonl_files = list(d.glob("*.jsonl"))
                    if jsonl_files:
                        # Skip current project if requested
                        if exclude_current and str(d) == current_log_path:
                            continue

                        # Find the most recently modified JSONL file
                        most_recent_mtime = max(f.stat().st_mtime for f in jsonl_files)
                        log_dirs.append((d, most_recent_mtime))

            # Sort by most recent JSONL modification time (most recent first)
            log_dirs.sort(key=lambda x: x[1], reverse=True)

            # Take only the most recent projects
            recent_dirs = log_dirs[:limit]

            logger.debug(f"Starting to warm {len(recent_dirs)} recent projects")

            for log_dir, _ in recent_dirs:
                log_path = str(log_dir)

                # Skip if already in memory cache
                if memory_cache.get(log_path):
                    logger.info(f"{log_dir.name} already in memory cache")
                    continue

                # Yield to other tasks
                await asyncio.sleep(0.1)

                try:
                    # Process the project
                    processor = ClaudeLogProcessor(log_path)
                    messages, stats = processor.process_logs()

                    # Save to file cache first (creates metadata)
                    cache_service.save_cached_stats(log_path, stats)
                    cache_service.save_cached_messages(log_path, messages)

                    # Then store in memory cache (force=True for initial warming)
                    if memory_cache.put(log_path, messages, stats, force=True):
                        logger.debug(f"Successfully warmed {log_dir.name}")
                    else:
                        logger.debug(f"Failed to cache {log_dir.name} (too large)")

                except Exception as e:
                    logger.info(f"Error processing {log_dir.name}: {e}")

                # Longer delay between projects to avoid hogging resources
                await asyncio.sleep(0.5)

        logger.debug(f"Completed warming {len(recent_dirs)} projects")

    except Exception as e:
        logger.info(f"Error in warm_recent_projects: {e}")
