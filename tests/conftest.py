"""Pytest configuration for regression fixtures."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--write-goldens",
        "--write",
        action="store_true",
        default=False,
        help="rewrite golden npz fixtures instead of checking them",
    )


@pytest.fixture
def write_goldens(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("write_goldens"))
