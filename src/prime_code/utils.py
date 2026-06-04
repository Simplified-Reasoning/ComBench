# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import sys
import traceback
from typing import Optional

from .testing_util import extract_prime_code_error_location, run_test


def _format_exitcode(exitcode: Optional[int]) -> str:
    if exitcode is None:
        return "unknown"
    if exitcode < 0:
        return f"signal {-exitcode}"
    return str(exitcode)


def _build_worker_exception_metadata(exc: BaseException) -> dict[str, object]:
    error_traceback = traceback.format_exc()
    message = str(exc).strip()
    detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    metadata = {
        "status": "internal_error",
        "error": repr(exc),
        "error_message": f"prime_code worker crashed with {detail}",
        "exception_type": type(exc).__name__,
        "traceback": error_traceback,
    }
    location = extract_prime_code_error_location(error_traceback)
    if location is not None:
        metadata["prime_code_error_location"] = location
    return metadata


def _build_missing_result_metadata(timeout: int, exitcode: Optional[int], timed_out: bool) -> dict[str, object]:
    if timed_out:
        return {
            "status": "timeout",
            "error": f"worker exceeded timeout={timeout}s",
            "error_message": f"prime_code worker exceeded timeout={timeout}s and was killed",
            "process_exitcode": exitcode,
            "timeout_seconds": timeout,
        }

    return {
        "status": "worker_crash",
        "error": f"worker exited before reporting results (exitcode={_format_exitcode(exitcode)})",
        "error_message": "prime_code worker exited before reporting results",
        "process_exitcode": exitcode,
        "timeout_seconds": timeout,
    }


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except BaseException as exc:
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append(_build_worker_exception_metadata(exc))


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    timed_out = p.is_alive()
    if p.is_alive():
        p.kill()
        p.join()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list.append(_build_missing_result_metadata(timeout=timeout, exitcode=p.exitcode, timed_out=timed_out))
    return result[0], metadata_list
