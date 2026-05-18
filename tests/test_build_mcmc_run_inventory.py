import unittest

from scripts import build_mcmc_run_inventory


class BuildMcmcRunInventoryTests(unittest.TestCase):
    def test_safe_float_rejects_nan_and_inf(self):
        self.assertIsNone(build_mcmc_run_inventory._safe_float("nan"))
        self.assertIsNone(build_mcmc_run_inventory._safe_float("inf"))
        self.assertIsNone(build_mcmc_run_inventory._safe_float(float("nan")))
        self.assertIsNone(build_mcmc_run_inventory._safe_float(float("inf")))
        self.assertEqual(build_mcmc_run_inventory._safe_float("3.5"), 3.5)

    def test_safe_int_rejects_overflow_and_bad_values(self):
        self.assertIsNone(build_mcmc_run_inventory._safe_int(float("inf")))
        self.assertIsNone(build_mcmc_run_inventory._safe_int(float("-inf")))
        self.assertIsNone(build_mcmc_run_inventory._safe_int("abc"))
        self.assertEqual(build_mcmc_run_inventory._safe_int("7"), 7)


if __name__ == "__main__":
    unittest.main()
