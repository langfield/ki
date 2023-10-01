#!/usr/bin/env python3
"""Test that imports work okay."""
# pylint: disable=unused-import
import pytest
import ki


@pytest.mark.skip
def test_package():
    """Dummy test."""
    assert True
