# Releasing kalbee

How to cut a release. From kalbee 1.0.0 onward, the public API is stable:
releases follow semantic versioning, so `MAJOR` changes break the
`predict`/`update` contract or the documented exports, `MINOR` adds features
backwards-compatibly, and `PATCH` is for fixes only.

## Preflight (must all pass)

```bash
uv sync
uv run ruff check .
uv run ruff format --check .      # must be clean
uv run pytest tests/ --cov=kalbee --cov-fail-under=90
uv run mkdocs build               # docs must build with no warnings
```

## Bump the version

The version lives in exactly two places — keep them in sync:

1. `kalbee/constants.py` → `__version__`
2. `pyproject.toml` → `[project] version`

Regenerate any version string in prose that names a specific release (the
paper cite in `.github/docs/kalbee.tex` is the usual one) and recompile the PDF:

```bash
cd .github/docs && pdflatex -interaction=nonstopmode -halt-on-error kalbee.tex
```

Add a `CHANGELOG.md` entry under a new `## [X.Y.Z]` heading.

## Tag and publish

```bash
git add -A
git commit -m "release: kalbee X.Y.Z"
git push origin main
git tag -a vX.Y.Z -m "kalbee X.Y.Z"
git push origin vX.Y.Z
```

Build and upload the package (requires a PyPI token in `~/.pypirc` or
`TWINE_PASSWORD`):

```bash
uv run python -m pip install --upgrade build twine
uv run python -m build
uv run python -m twine check dist/*
uv run python -m twine upload dist/*
```

Then create a GitHub Release from the `vX.Y.Z` tag (title `kalbee X.Y.Z`,
body: the CHANGELOG entry) and link the compiled `kalbee.pdf` if relevant.

## After publishing

- Sanity-check the PyPI page and run one fresh install in a clean venv:
  `pip install kalbee && kalbee --version`.
- Bump to the next development version (`X.Y.(Z+1)dev`) in both version
  locations so accidental local installs are clearly pre-release.
