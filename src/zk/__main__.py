"""`python -m zk ...` runs the CLI, for hooks and scheduled tasks that call the venv's python."""

from zk.cli import main

raise SystemExit(main())
