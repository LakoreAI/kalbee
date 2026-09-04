#!/usr/bin/env bash
#
# Build and publish kalbee to PyPI.
#
# Usage:
#   ./scripts/publish.sh            # build, check, and upload to PyPI
#   ./scripts/publish.sh --test     # upload to TestPyPI instead
#   ./scripts/publish.sh --no-upload # build + check only (dry run)
#
# Credentials are read from a local .env file (TWINE_USERNAME / TWINE_PASSWORD)
# if present, otherwise from the environment. Uploads use --skip-existing, so
# re-running for an already-published version is a safe no-op.
#
# Remember to bump `version` in pyproject.toml AND __version__ in
# kalbee/__init__.py before publishing a new release — PyPI never allows
# overwriting an existing version.

set -euo pipefail

# Always operate from the repository root, regardless of where we're invoked.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPOSITORY="pypi"
DO_UPLOAD=1

for arg in "$@"; do
  case "$arg" in
    --test) REPOSITORY="testpypi" ;;
    --no-upload) DO_UPLOAD=0 ;;
    -h|--help)
      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

VERSION="$(grep -m1 '^version' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')"
echo "==> Publishing kalbee $VERSION (repository: $REPOSITORY)"

# Sanity check: pyproject.toml and __init__.py versions must agree.
INIT_VERSION="$(grep -m1 '__version__' kalbee/__init__.py | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ "$VERSION" != "$INIT_VERSION" ]]; then
  echo "ERROR: version mismatch — pyproject.toml=$VERSION, __init__.py=$INIT_VERSION" >&2
  exit 1
fi

# 1. Clean previous build artifacts so we only ship the current version.
echo "==> Cleaning dist/"
rm -f dist/*.whl dist/*.tar.gz

# 2. Build sdist + wheel.
echo "==> Building distributions"
uv build

# 3. Validate metadata renders correctly on PyPI.
echo "==> Checking distributions"
uv run --with twine twine check dist/*

if [[ "$DO_UPLOAD" -eq 0 ]]; then
  echo "==> --no-upload set; built and checked only. Artifacts in dist/:"
  ls -1 dist/
  exit 0
fi

# 4. Load credentials from .env if available (never printed).
if [[ -f .env ]]; then
  echo "==> Loading credentials from .env"
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# 5. Upload. --skip-existing makes re-runs of an already-published version safe.
echo "==> Uploading to $REPOSITORY"
uv run --with twine twine upload --repository "$REPOSITORY" --skip-existing dist/*

echo "==> Done. https://pypi.org/project/kalbee/$VERSION/"
