"""Pytest configuration and shared fixtures."""

import pytest
from app import config


@pytest.fixture
def project_config():
    """Fixture providing application configuration."""
    return config
