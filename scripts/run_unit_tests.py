from __future__ import annotations

import importlib
import inspect
import traceback


def run_module_tests(module_name: str) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        results.append((module_name, False, traceback.format_exc()))
        return results

    for name, obj in inspect.getmembers(mod):
        if name.startswith("test_") and inspect.isfunction(obj):
            try:
                obj()
                results.append((f"{module_name}.{name}", True, ""))
            except Exception:  # AssertionError or other failures
                results.append((f"{module_name}.{name}", False, traceback.format_exc()))
    return results


def main() -> int:
    # Only run unit tests; skip integration package
    modules = [
        "tests.test_types",
        "tests.test_vectorstore",
    ]
    failed = 0
    for m in modules:
        for test_name, ok, err in run_module_tests(m):
            status = "PASS" if ok else "FAIL"
            print(f"{status} - {test_name}")
            if not ok:
                print(err)
                failed += 1
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
