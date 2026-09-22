# OpenDleto Copilot Instructions

## Notebook Output Policy

When editing or adding Jupyter notebooks (`*.ipynb`) in this repository:

1. Remove all computed cell outputs before commit/push.
2. Do not commit execution counts.
3. Keep notebooks source-focused so diffs remain reviewable.

### Required workflow

- Run `pre-commit run --all-files` before pushing notebook changes.
- If `pre-commit` is not installed, install and enable it:
  - `pip install pre-commit`
  - `pre-commit install`

### Enforcement

- CI workflow `notebook-output-hygiene.yml` is authoritative and will fail if notebook outputs are committed.
- If CI fails on notebook hygiene, strip outputs and recommit.
