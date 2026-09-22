# AGENTS Guidance for OpenDleto

## Always Keep Notebooks Clean

For all changes to Jupyter notebooks (`*.ipynb`):

- Strip computed outputs before commit.
- Avoid committing execution counts.
- Keep notebook diffs limited to source/content changes.

## Before Push

Run:

```bash
pre-commit run --all-files
```

If needed:

```bash
pip install pre-commit
pre-commit install
```

## CI Policy

This repository enforces notebook hygiene in `.github/workflows/notebook-output-hygiene.yml`.
Any push/PR with notebook outputs is expected to fail until outputs are stripped.
