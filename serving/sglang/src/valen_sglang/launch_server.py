"""Launch the stock engine with mandatory source, runtime, and plugin checks."""

import os
import sys

from valen_sglang.compatibility import verify_engine, verify_runtime


def main():
    verify_engine()
    verify_runtime()
    allowlist = os.environ.get("SGLANG_PLUGINS")
    if allowlist and "valen" not in {value.strip() for value in allowlist.split(",")}:
        raise RuntimeError("SGLANG_PLUGINS excludes valen; include valen or unset it")
    from sglang.srt.plugins import load_plugins

    load_plugins()
    from valen_sglang.plugin import assert_hooks_applied

    assert_hooks_applied()
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
