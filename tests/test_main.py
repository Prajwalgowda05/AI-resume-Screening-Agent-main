"""Smoke tests for application entry point and CLI interface."""

import io
import sys
import unittest
from unittest.mock import patch
from app.main import main
from app import config


class TestAppFoundation(unittest.TestCase):
    """Test suite for validating the main CLI entry point."""

    def test_main_execution(self):
        """Test that app.main executes and returns 0 exit code."""
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        try:
            sys.stdout = captured_output
            with patch("sys.argv", ["app.main"]):
                exit_code = main()
        finally:
            sys.stdout = original_stdout

        self.assertEqual(exit_code, 0)
        output = captured_output.getvalue()
        self.assertIn("AI Resume Screening Agent", output)

    def test_config_paths(self):
        """Test that basic paths in config are resolved."""
        self.assertTrue(config.BASE_DIR.exists())


if __name__ == "__main__":
    unittest.main()
