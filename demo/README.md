# MEMO interactive demo

A self-contained, static gallery for the MEMO surface-flux emulator. Pick a
held-out test date (2013–2014), a flux variable, and a model, and compare the
**SST input → model prediction → CESM truth → error** side by side, with the
pooled test-set R² for each model.

Everything here is static — `index.html` + `manifest.json` + JPEGs in
`assets/`. No server, build step, or dependencies are needed to view it.

## View locally

Because the page uses `fetch()` to load `manifest.json`, open it through a
tiny local server rather than `file://`:

```bash
cd demo
python -m http.server 8000
# then open http://localhost:8000
```

## Deploy on GitHub Pages

Two options:

**A. GitHub Actions (recommended, serves the demo at the site root)**
1. Push this repo (the workflow `.github/workflows/deploy-demo.yml` is included).
2. Repo → **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. On the next push to `master`, the demo deploys to
   `https://<user>.github.io/<repo>/`.

**B. Serve straight from the branch (no Actions)**
1. Repo → **Settings → Pages → Source: Deploy from a branch**, branch
   `master`, folder `/ (root)`.
2. The demo is then at `https://<user>.github.io/<repo>/demo/`.

## Regenerate

The gallery is produced by [`../scripts/build_demo.py`](../scripts/build_demo.py),
which loads the three trained checkpoints, runs them on the same 2013–2014
timesteps from the `couple_cache_mem24h` cache, and renders the composite maps.
Run it on a machine with the project's `atm` conda env (CPU is fine):

```bash
python scripts/build_demo.py
```
