import importlib
import sys
import types
import unittest
import warnings
from unittest.mock import patch


_TESTING_UTILS = {}


def _load_testing_util(platform_name=None):
    cache_key = platform_name or "__default__"
    if cache_key in _TESTING_UTILS:
        return _TESTING_UTILS[cache_key]

    fake_pyext = types.ModuleType("pyext")

    class FakeRuntimeModule:
        @staticmethod
        def from_string(*args, **kwargs):
            return types.SimpleNamespace()

    fake_pyext.RuntimeModule = FakeRuntimeModule

    import_patches = [patch.dict(sys.modules, {"pyext": fake_pyext})]
    if platform_name is not None:
        import_patches.append(patch("platform.system", return_value=platform_name))

    with import_patches[0]:
        active_context = import_patches[1] if len(import_patches) > 1 else None
        if active_context is not None:
            active_context.__enter__()
        try:
            sys.modules.pop("src.prime_code.testing_util", None)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The NumPy module was reloaded.*")
                testing_util = importlib.import_module("src.prime_code.testing_util")
        finally:
            if active_context is not None:
                active_context.__exit__(None, None, None)

    _TESTING_UTILS[cache_key] = testing_util
    return testing_util


class PrimeCodeTestingUtilTest(unittest.TestCase):
    def test_extract_prime_code_error_location_from_traceback(self):
        testing_util = _load_testing_util()
        traceback_text = """Traceback (most recent call last):
  File "/root/projects/IMO/comb_bench/src/prime_code/utils.py", line 31, in _temp_run
    res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
  File "/root/projects/IMO/comb_bench/src/prime_code/testing_util.py", line 122, in run_test
    _set_timeout_alarm(timeout)
RuntimeError: boom
"""

        location = testing_util.extract_prime_code_error_location(traceback_text)

        self.assertEqual(
            {
                "file": "src/prime_code/testing_util.py",
                "line": 122,
                "function": "run_test",
                "code": "_set_timeout_alarm(timeout)",
            },
            location,
        )

    def test_clean_traceback_preserves_non_string_tracebacks(self):
        testing_util = _load_testing_util()
        traceback_text = """Traceback (most recent call last):
  File "/root/projects/IMO/comb_bench/src/prime_code/testing_util.py", line 122, in run_test
    _set_timeout_alarm(timeout)
RuntimeError: boom
"""

        self.assertEqual(traceback_text, testing_util.clean_traceback(traceback_text))

    def test_set_timeout_alarm_skips_signal_alarm_on_windows(self):
        testing_util = _load_testing_util(platform_name="Windows")

        self.assertTrue(testing_util.IS_WINDOWS)

        with patch.object(testing_util.signal, "alarm", create=True) as mock_alarm:
            testing_util._set_timeout_alarm(5)

        mock_alarm.assert_not_called()
