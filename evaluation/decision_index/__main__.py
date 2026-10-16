"""Forward commands to the pinned upstream runner and scorers."""
from ._compat import ensure_file_digest


def main():
    ensure_file_digest()
    try:
        from decision_index.cli import main as upstream_main
    except ModuleNotFoundError as exc:
        if exc.name == "decision_index":
            raise SystemExit('Install the optional dependency: python -m pip install -e ".[decision-index]"') from exc
        raise
    upstream_main()


if __name__ == "__main__":
    main()
