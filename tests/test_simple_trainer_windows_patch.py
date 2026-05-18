from __future__ import annotations

import builtins
import importlib
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock


class SimpleTrainerWindowsPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.gsplat_runner_root = str(cls.repo_root / "gsplat_runner")
        sys.path.insert(0, cls.gsplat_runner_root)
        cls.simple_trainer = importlib.import_module("simple_trainer")

    @classmethod
    def tearDownClass(cls):
        try:
            sys.path.remove(cls.gsplat_runner_root)
        except ValueError:
            pass

    def test_patch_windows_cpp_extension_decode_warns_when_cpp_extension_import_fails(self):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch.utils" and "cpp_extension" in fromlist:
                raise ImportError("missing cpp_extension")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(self.simple_trainer.os, "name", "nt"), mock.patch(
            "builtins.__import__", side_effect=fake_import
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.simple_trainer._patch_windows_cpp_extension_decode()

        self.assertEqual(len(caught), 1)
        self.assertIn(
            "Failed to patch torch Windows JIT subprocess decode path",
            str(caught[0].message),
        )


if __name__ == "__main__":
    unittest.main()
