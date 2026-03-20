"""Runner script that discovers and executes all KSE unit tests.

Usage:
    python -m test.srt.kse.run_all_kse_tests          (from repo root)
    python test/srt/kse/run_all_kse_tests.py           (from repo root)
"""

import sys
import unittest


def main():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_modules = [
        "test.srt.kse.test_types_and_config",
        "test.srt.kse.test_registry",
        "test.srt.kse.test_controller",
        "test.srt.kse.test_quest_policy",
        "test.srt.kse.test_streaming_llm_policy",
        "test.srt.kse.test_triton_adapter",
        "test.srt.kse.test_flashattention_adapter",
    ]

    for mod_name in test_modules:
        try:
            suite.addTests(loader.loadTestsFromName(mod_name))
        except Exception as e:
            print(f"ERROR loading {mod_name}: {e}", file=sys.stderr)
            return 1

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
