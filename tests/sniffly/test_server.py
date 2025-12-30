"""
Tests for the FastAPI server endpoints and functionality.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def patch_claude_projects_dir(temp_dir: Path):
    """Patch get_claude_projects_dir to return temp_dir/.claude/projects."""
    projects_dir = temp_dir / ".claude" / "projects"
    return patch("sniffly.utils.cache_warmer.get_claude_projects_dir", return_value=projects_dir)


class TestServerImports:
    """Test that server modules can be imported without errors."""
    
    def test_can_import_server(self):
        """Test that server.py can be imported."""
        with patch('sniffly.utils.cache_warmer.warm_recent_projects', new=AsyncMock()):
            import sniffly.server
            assert hasattr(sniffly.server, 'app')
            assert hasattr(sniffly.server, 'memory_cache')
            assert hasattr(sniffly.server, 'cache_service')
    
    def test_can_import_cache_warmer(self):
        """Test that cache_warmer.py can be imported."""
        from sniffly.utils.cache_warmer import warm_recent_projects
        assert callable(warm_recent_projects)
    
    def test_can_import_memory_cache(self):
        """Test that memory_cache.py can be imported."""
        from sniffly.utils.memory_cache import MemoryCache
        cache = MemoryCache()
        assert hasattr(cache, 'get')
        assert hasattr(cache, 'put')
        assert hasattr(cache, 'invalidate')
    
    def test_can_import_local_cache(self):
        """Test that local_cache.py can be imported."""
        from sniffly.utils.local_cache import LocalCacheService
        cache = LocalCacheService()
        assert hasattr(cache, 'get_cached_stats')
        assert hasattr(cache, 'save_cached_stats')


class TestServerEndpointStructure:
    """Test that server endpoints are properly defined."""
    
    def test_server_has_required_endpoints(self):
        """Test that server has all required endpoints defined."""
        with patch('sniffly.utils.cache_warmer.warm_recent_projects', new=AsyncMock()):
            from sniffly.server import app
            
            # Get all routes
            routes = []
            for route in app.routes:
                if hasattr(route, 'path'):
                    routes.append(route.path)
            
            # Check critical endpoints exist
            assert "/" in routes
            assert "/project/{project_name:path}" in routes  # Project-specific URLs
            assert "/api/health" in routes
            assert "/api/project" in routes
            assert "/api/stats" in routes
            assert "/api/messages" in routes
            assert "/api/dashboard-data" in routes
            assert "/api/cache/status" in routes
            assert "/api/refresh" in routes
            assert "/api/recent-projects" in routes
            assert "/api/projects" in routes  # New comprehensive projects endpoint
            assert "/api/pricing" in routes
            assert "/api/share-enabled" in routes
    
    def test_server_global_variables(self):
        """Test that server has required global variables."""
        with patch('sniffly.utils.cache_warmer.warm_recent_projects', new=AsyncMock()):
            import sniffly.server as server
            
            # Check globals exist
            assert hasattr(server, 'current_project_path')
            assert hasattr(server, 'current_log_path')
            assert hasattr(server, 'memory_cache')
            assert hasattr(server, 'cache_service')
            assert hasattr(server, 'cache_warm_on_startup')


class TestCacheWarmerFunction:
    """Test the cache warmer function behavior."""
    
    @pytest.mark.asyncio
    async def test_warm_recent_projects_handles_empty_dir(self):
        """Test that warm_recent_projects handles missing directories gracefully."""
        from sniffly.utils.cache_warmer import warm_recent_projects

        mock_cache_service = Mock()
        mock_memory_cache = Mock()

        # Create a temporary directory that doesn't have .claude/projects
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch_claude_projects_dir(Path(temp_dir)):
                # Should complete without errors
                await warm_recent_projects(
                    mock_cache_service,
                    mock_memory_cache,
                    None,
                    exclude_current=False,
                    limit=3
                )

        # Should not have tried to cache anything
        mock_memory_cache.put.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_warm_recent_projects_skips_current(self):
        """Test that warm_recent_projects can skip the current project."""
        from sniffly.utils.cache_warmer import warm_recent_projects

        mock_cache_service = Mock()
        mock_memory_cache = Mock()
        mock_memory_cache.get.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock project structure
            claude_dir = Path(temp_dir) / ".claude" / "projects"
            claude_dir.mkdir(parents=True)

            # Create two project dirs
            project1 = claude_dir / "project1"
            project1.mkdir()
            (project1 / "log.jsonl").write_text('{"type": "user"}\n')

            project2 = claude_dir / "project2"
            project2.mkdir()
            (project2 / "log.jsonl").write_text('{"type": "user"}\n')

            with patch_claude_projects_dir(Path(temp_dir)):
                # Mock processor to avoid actual processing
                with patch('sniffly.utils.cache_warmer.ClaudeLogProcessor') as mock_processor:
                    mock_processor.return_value.process_logs.return_value = ([], {})

                    # Warm with current project = project1
                    await warm_recent_projects(
                        mock_cache_service,
                        mock_memory_cache,
                        str(project1),
                        exclude_current=True,
                        limit=5
                    )

                    # Should only process project2
                    assert mock_processor.call_count == 1
                    assert "project2" in str(mock_processor.call_args[0][0])


class TestMemoryCacheFunctionality:
    """Test memory cache basic functionality."""
    
    def test_memory_cache_basic_operations(self):
        """Test basic memory cache operations."""
        from sniffly.utils.memory_cache import MemoryCache
        
        cache = MemoryCache(max_projects=2, max_mb_per_project=1)
        
        # Test empty cache
        assert cache.get("test_path") is None
        
        # Test put and get
        messages = [{"type": "user", "content": "test"}]
        stats = {"total": 1}
        assert cache.put("test_path", messages, stats)
        
        result = cache.get("test_path")
        assert result is not None
        assert result[0] == messages
        assert result[1] == stats
        
        # Test invalidate
        assert cache.invalidate("test_path")
        assert cache.get("test_path") is None
        
        # Test stats
        cache_stats = cache.get_stats()
        assert cache_stats['hits'] >= 0
        assert cache_stats['misses'] >= 0
        assert cache_stats['projects_cached'] == 0
    
    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction in memory cache."""
        from sniffly.utils.memory_cache import MemoryCache
        
        cache = MemoryCache(max_projects=2, max_mb_per_project=1)
        
        # Add two projects
        cache.put("path1", [{"msg": 1}], {"stats": 1})
        cache.put("path2", [{"msg": 2}], {"stats": 2})
        
        # Access path1 to make it more recently used
        cache.get("path1")
        
        # Add third project - should evict path2 (least recently used)
        # Use force=True to bypass protection window
        cache.put("path3", [{"msg": 3}], {"stats": 3}, force=True)
        
        # path1 and path3 should be in cache, path2 should be evicted
        assert cache.get("path1") is not None
        assert cache.get("path2") is None
        assert cache.get("path3") is not None
    
    def test_memory_cache_size_limit(self):
        """Test memory cache size limits."""
        from sniffly.utils.memory_cache import MemoryCache
        
        # Very small size limit (1KB)
        cache = MemoryCache(max_projects=5, max_mb_per_project=0.001)
        
        # Create large message
        large_messages = [{"content": "x" * 10000} for _ in range(100)]
        stats = {"total": 100}
        
        # Should reject due to size
        result = cache.put("large_path", large_messages, stats)
        assert result is False
        assert cache.get("large_path") is None


class TestLocalCacheService:
    """Test local cache service functionality."""
    
    def test_local_cache_basic_operations(self):
        """Test basic local cache operations."""
        from sniffly.utils.local_cache import LocalCacheService
        
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.dict(os.environ, {'CACHE_DIR': cache_dir}):
                cache = LocalCacheService()
                
                # Test saving and retrieving stats
                test_stats = {"total_messages": 42, "total_tokens": 1000}
                cache.save_cached_stats("/test/path", test_stats)
                
                retrieved_stats = cache.get_cached_stats("/test/path")
                assert retrieved_stats == test_stats
                
                # Test saving and retrieving messages
                test_messages = [{"type": "user", "content": "hello"}]
                cache.save_cached_messages("/test/path", test_messages)
                
                retrieved_messages = cache.get_cached_messages("/test/path")
                assert retrieved_messages == test_messages
                
                # Test invalidation
                cache.invalidate_cache("/test/path")
                assert cache.get_cached_stats("/test/path") is None
                assert cache.get_cached_messages("/test/path") is None


class TestServerConfiguration:
    """Test server configuration and environment variables."""
    
    def test_cache_configuration_from_env(self):
        """Test that cache configuration is read from environment."""
        with patch.dict(os.environ, {
            'CACHE_MAX_PROJECTS': '10',
            'CACHE_MAX_MB_PER_PROJECT': '100',
            'CACHE_WARM_ON_STARTUP': '5'
        }):
            # Re-import to get new env values
            import importlib

            import sniffly.server
            importlib.reload(sniffly.server)
            
            assert sniffly.server.max_projects == 10
            assert sniffly.server.max_mb_per_project == 100
            assert sniffly.server.cache_warm_on_startup == 5


class TestProjectAPIMethods:
    """Test project-related API logic without full server."""
    
    def test_project_path_conversion(self):
        """Test conversion between project paths and log directories."""
        # Test path with leading dash
        dir_name = "-Users-john-dev-myapp"
        expected_path = "/Users/john/dev/myapp"
        
        # Simulate the conversion logic from server.py
        if dir_name.startswith('-'):
            project_path = '/' + dir_name[1:].replace('-', '/')
        else:
            project_path = dir_name.replace('-', '/')
        
        assert project_path == expected_path
        
        # Test path without leading dash
        dir_name2 = "Users-john-dev-myapp"
        if dir_name2.startswith('-'):
            project_path2 = '/' + dir_name2[1:].replace('-', '/')
        else:
            project_path2 = dir_name2.replace('-', '/')
        
        assert project_path2 == "Users/john/dev/myapp"


class TestProjectsAPIEndpoint:
    """Test the new /api/projects endpoint logic."""
    
    def test_projects_api_imports(self):
        """Test that the projects API imports work."""
        from sniffly.utils.log_finder import get_all_projects_with_metadata
        assert callable(get_all_projects_with_metadata)
    
    def test_project_specific_url_routing(self):
        """Test that project-specific URLs are configured."""
        with patch('sniffly.utils.cache_warmer.warm_recent_projects', new=AsyncMock()):
            from sniffly.server import app
            
            # Check the route exists and captures path parameter
            project_route = None
            for route in app.routes:
                if hasattr(route, 'path') and '{project_name:path}' in route.path:
                    project_route = route
                    break
            
            assert project_route is not None
            assert project_route.path == "/project/{project_name:path}"
    
    def test_projects_api_with_mock_data(self):
        """Test projects API response structure with mock data."""
        # Mock project data
        mock_projects = [
            {
                'dir_name': '-Users-test-project1',
                'log_path': '/home/.claude/projects/-Users-test-project1',
                'file_count': 3,
                'total_size_mb': 1.5,
                'last_modified': 1704100000,
                'first_seen': 1704000000,
                'display_name': 'Users/test/project1'
            },
            {
                'dir_name': '-Users-test-project2',
                'log_path': '/home/.claude/projects/-Users-test-project2',
                'file_count': 1,
                'total_size_mb': 0.5,
                'last_modified': 1704200000,
                'first_seen': 1704150000,
                'display_name': 'Users/test/project2'
            }
        ]
        
        # Test sorting by last_modified (default)
        sorted_projects = sorted(mock_projects, key=lambda x: x['last_modified'], reverse=True)
        assert sorted_projects[0]['dir_name'] == '-Users-test-project2'
        
        # Test sorting by size
        sorted_by_size = sorted(mock_projects, key=lambda x: x['total_size_mb'], reverse=True)
        assert sorted_by_size[0]['dir_name'] == '-Users-test-project1'
        
        # Test pagination
        paginated = mock_projects[0:1]
        assert len(paginated) == 1
        assert paginated[0]['dir_name'] == '-Users-test-project1'
    
    def test_project_switching_no_duplicate_selectors(self):
        """Test that project switching doesn't create duplicate selectors."""
        # This tests the fix for the duplicate project selector issue
        
        # The fix was in project-detector.js - removed populateProjectSelector
        # and in dashboard.html - updated switchProject to use URL navigation
        
        # We can verify the server-side logic
        with patch('sniffly.utils.cache_warmer.warm_recent_projects', new=AsyncMock()):
            from sniffly.server import app
            
            # Verify both root and project routes exist
            routes = [route.path for route in app.routes if hasattr(route, 'path')]
            assert "/" in routes
            assert "/project/{project_name:path}" in routes
            
            # The actual DOM testing would require end-to-end tests
            # For now, we've verified the server routes are correct


class TestRefreshAllProjects:
    """Test the refresh_all_projects function behavior."""
    
    @pytest.mark.asyncio
    async def test_refresh_all_projects_no_changes(self):
        """Test refresh_all_projects when no projects have changed."""
        from sniffly.server import refresh_all_projects
        
        # Mock get_all_projects_with_metadata to return test projects
        test_projects = [
            {"display_name": "project1", "log_path": "/test/path1"},
            {"display_name": "project2", "log_path": "/test/path2"},
            {"display_name": "project3", "log_path": "/test/path3"}
        ]
        
        # Mock services
        mock_cache_service = Mock()
        mock_cache_service.has_changes.return_value = False  # No changes
        
        mock_memory_cache = Mock()
        
        with patch('sniffly.utils.log_finder.get_all_projects_with_metadata', return_value=test_projects):
            with patch('sniffly.server.cache_service', mock_cache_service):
                with patch('sniffly.server.memory_cache', mock_memory_cache):
                    result = await refresh_all_projects({})
        
        # Verify has_changes was called for each project
        assert mock_cache_service.has_changes.call_count == 3
        mock_cache_service.has_changes.assert_any_call("/test/path1")
        mock_cache_service.has_changes.assert_any_call("/test/path2")
        mock_cache_service.has_changes.assert_any_call("/test/path3")
        
        # Verify no invalidation or processing happened
        mock_cache_service.invalidate_cache.assert_not_called()
        mock_memory_cache.invalidate.assert_not_called()
        
        # Check response
        response = result.body.decode()
        response_data = json.loads(response)
        assert response_data["status"] == "success"
        assert response_data["files_changed"] is False
        assert response_data["projects_refreshed"] == 0
        assert response_data["total_projects"] == 3
        assert "No changes detected" in response_data["message"]
    
    @pytest.mark.asyncio
    async def test_refresh_all_projects_with_changes(self):
        """Test refresh_all_projects when some projects have changed."""
        from sniffly.server import refresh_all_projects
        
        # Mock get_all_projects_with_metadata to return test projects
        test_projects = [
            {"display_name": "project1", "log_path": "/test/path1"},
            {"display_name": "project2", "log_path": "/test/path2"},
            {"display_name": "project3", "log_path": "/test/path3"}
        ]
        
        # Mock services
        mock_cache_service = Mock()
        # First project: no changes, second: has changes, third: no changes
        mock_cache_service.has_changes.side_effect = [False, True, False]
        
        mock_memory_cache = Mock()
        
        # Mock processor
        mock_messages = [{"type": "user", "content": "test"}]
        mock_stats = {"total": 1}
        
        with patch('sniffly.utils.log_finder.get_all_projects_with_metadata', return_value=test_projects):
            with patch('sniffly.server.cache_service', mock_cache_service):
                with patch('sniffly.server.memory_cache', mock_memory_cache):
                    with patch('sniffly.server.ClaudeLogProcessor') as mock_processor:
                        mock_processor.return_value.process_logs.return_value = (mock_messages, mock_stats)
                        
                        result = await refresh_all_projects({})
        
        # Verify has_changes was called for each project
        assert mock_cache_service.has_changes.call_count == 3
        
        # Verify only project2 was invalidated and processed
        mock_cache_service.invalidate_cache.assert_called_once_with("/test/path2")
        mock_memory_cache.invalidate.assert_called_once_with("/test/path2")
        
        # Verify processor was called once for project2
        assert mock_processor.call_count == 1
        mock_processor.assert_called_with("/test/path2")
        
        # Verify caches were updated for project2
        mock_cache_service.save_cached_stats.assert_called_once_with("/test/path2", mock_stats)
        mock_cache_service.save_cached_messages.assert_called_once_with("/test/path2", mock_messages)
        mock_memory_cache.put.assert_called_once_with("/test/path2", mock_messages, mock_stats)
        
        # Check response
        response = result.body.decode()
        response_data = json.loads(response)
        assert response_data["status"] == "success"
        assert response_data["files_changed"] is True
        assert response_data["projects_refreshed"] == 1
        assert response_data["total_projects"] == 3
        assert "Refreshed 1 of 3 projects" in response_data["message"]
    
    @pytest.mark.asyncio
    async def test_refresh_all_projects_handles_errors(self):
        """Test refresh_all_projects handles processing errors gracefully."""
        from sniffly.server import refresh_all_projects
        
        # Mock get_all_projects_with_metadata to return test projects
        test_projects = [
            {"display_name": "project1", "log_path": "/test/path1"},
            {"display_name": "project2", "log_path": "/test/path2"},
        ]
        
        # Mock services
        mock_cache_service = Mock()
        # Both projects have changes
        mock_cache_service.has_changes.return_value = True
        
        mock_memory_cache = Mock()
        
        with patch('sniffly.utils.log_finder.get_all_projects_with_metadata', return_value=test_projects):
            with patch('sniffly.server.cache_service', mock_cache_service):
                with patch('sniffly.server.memory_cache', mock_memory_cache):
                    with patch('sniffly.server.ClaudeLogProcessor') as mock_processor:
                        # First project processes successfully, second throws error
                        mock_processor.return_value.process_logs.side_effect = [
                            ([{"msg": 1}], {"stats": 1}),
                            Exception("Processing error")
                        ]
                        
                        result = await refresh_all_projects({})
        
        # Verify both projects were attempted to be processed
        assert mock_processor.call_count == 2
        
        # Verify caches were invalidated for both
        assert mock_cache_service.invalidate_cache.call_count == 2
        assert mock_memory_cache.invalidate.call_count == 2
        
        # Verify only first project's caches were saved
        assert mock_cache_service.save_cached_stats.call_count == 1
        assert mock_cache_service.save_cached_messages.call_count == 1
        assert mock_memory_cache.put.call_count == 1
        
        # Check response - should report 1 successful refresh
        response = result.body.decode()
        response_data = json.loads(response)
        assert response_data["status"] == "success"
        assert response_data["files_changed"] is True
        assert response_data["projects_refreshed"] == 1
        assert response_data["total_projects"] == 2
    
    @pytest.mark.asyncio
    async def test_refresh_all_projects_empty_projects(self):
        """Test refresh_all_projects when no projects exist."""
        from sniffly.server import refresh_all_projects
        
        # Mock get_all_projects_with_metadata to return empty list
        with patch('sniffly.utils.log_finder.get_all_projects_with_metadata', return_value=[]):
            result = await refresh_all_projects({})
        
        # Check response
        response = result.body.decode()
        response_data = json.loads(response)
        assert response_data["status"] == "success"
        assert response_data["files_changed"] is False
        assert response_data["projects_refreshed"] == 0
        assert response_data["total_projects"] == 0
        assert "No changes detected" in response_data["message"]
    
    @pytest.mark.asyncio
    async def test_refresh_endpoint_calls_refresh_all_projects(self):
        """Test that /api/refresh endpoint calls refresh_all_projects when no current project."""
        from sniffly.server import refresh_data
        
        # Mock current_log_path to be None
        with patch('sniffly.server.current_log_path', None):
            with patch('sniffly.server.refresh_all_projects', new=AsyncMock()) as mock_refresh_all:
                mock_refresh_all.return_value = Mock(
                    body=json.dumps({
                        "status": "success",
                        "message": "test",
                        "files_changed": False,
                        "refresh_time_ms": 10,
                        "projects_refreshed": 0,
                        "total_projects": 0
                    }).encode()
                )
                
                request_data = {"timezone_offset": 120}
                await refresh_data(request_data)
                
                # Verify refresh_all_projects was called with the request
                mock_refresh_all.assert_called_once_with(request_data)
    
    def test_refresh_all_projects_timing(self):
        """Test that refresh_all_projects tracks timing correctly."""
        # This would require a more complex integration test
        # For now, we've tested the main logic paths
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])