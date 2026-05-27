.PHONY: requirements test lock sync sync-cpu sync-cuda

# ---- Local dev ----

# Install/refresh the local venv. Default: CUDA torch (for GPU dev machines).
# On Mac, the CUDA wheel resolves to the regular Mac torch (no CUDA on Apple).
sync: sync-cuda

sync-cuda:
	uv sync --extra cuda --frozen

sync-cpu:
	uv sync --extra cpu --frozen

# Refresh uv.lock from pyproject.toml without installing.
lock:
	uv lock

# Run the test suite (uses whatever extra is currently installed).
test:
	uv run pytest tests/

# ---- HF Space deploy ----

# Regenerate requirements.txt for HF Space (always CPU torch).
# The workflow does the same on every main push.
requirements:
	uv export --extra cpu --no-hashes --no-dev --no-emit-project \
		--format requirements-txt -o requirements.txt
	{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; cat requirements.txt; } > requirements.txt.tmp
	mv requirements.txt.tmp requirements.txt
