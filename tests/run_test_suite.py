"""
Master Test Suite Runner for FCIL-AndMal2020.
Executes all unit, integration, model, method, and federated tests.
"""

import sys
import unittest

sys.dont_write_bytecode = True

if __name__ == "__main__":
    print("=" * 80)
    print("  RUNNING COMPLETE SCIENTIFIC TEST SUITE FOR FCIL-AndMal-2020")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="./tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if result.wasSuccessful():
        print("\n" + "=" * 80)
        print("  ✅ ALL UNIT AND INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print(f"  ❌ TEST SUITE FAILED: {len(result.failures)} Failures, {len(result.errors)} Errors")
        print("=" * 80)
        sys.exit(1)
