# Publishing MonoDOR on GitHub

The source release is prepared before the publisher page is online. It includes code, environment recipes, instructions and small result tables. The paper, review correspondence, dataset and checkpoints are separate artifacts.

## Export the current source

From the development checkout:

```bash
python scripts/release/check_source.py
python scripts/release/export_source.py
python release-dist/source/scripts/release/check_source.py
```

The exporter creates `release-dist/source/` and refuses to overwrite an existing destination. For a later snapshot, supply a fresh path with `--output release-dist/source-v2`.

It copies existing Git-tracked source files from the current working tree, plus the explicit public documentation, configuration, dependency files, Docker files and release scripts. Therefore it includes current edits and skips deleted files. Arbitrary untracked model code is not included automatically; review and track new implementation files before exporting them.

`RELEASE_MANIFEST.json` records the base commit, whether there are working-tree changes, and a SHA-256 hash of every copied file. The base commit does not by itself identify a modified snapshot.

The exported `.gitignore` adds exact exceptions for the included YAML/shell files. This preserves legacy configurations when initializing a new repository, without exposing unrelated ignored experiments from the development checkout.

The export excludes:

- Git history and IDE configuration.
- Local manuscript PDFs, revision notes, review responses and revision assets.
- Data, generated experiment output, model weights and compiled binaries.
- Local `.env` values; the path-only example file is included.

The exporter is a scoped source-packaging tool, not a general secret scanner. Review the generated files before publishing. `.gitignore` also excludes local research material, but ignore rules do not remove previously committed files from Git history.

## Publish the source snapshot

For a new empty GitHub repository, initialize Git inside the exported directory. Replace the remote with the repository you own:

```bash
cd release-dist/source
git init -b main
git add .
git diff --cached --stat
git commit -m "Prepare T-IV code and reproducible environment release"
# Set the actual remote URL, then push using your normal GitHub authentication.
# git remote add origin https://github.com/ho-tae/MonoDOR.git
# git push -u origin main
```

If updating the existing repository, review and commit the intended source/configuration/documentation changes in that checkout. Preserve ongoing experimental changes deliberately; do not replace an existing Git history with the snapshot just to upload the setup files.

No commit, push, GitHub release or container publication is performed by these scripts.

## Repository presentation

Repository name: **MonoDOR**. Publish it as a new repository owned by the personal GitHub account.

Repository description:

> MonoDOR: Depth- and Occupancy-Aware Representation Learning for Monocular 3D Object Detection. Accepted to IEEE T-IV.

Suggested topics: `monocular-3d-detection`, `vmamba`, `kitti`, `pytorch`, `autonomous-driving`.

The README already states acceptance and forthcoming online publication. Add the publisher link and final BibTeX once official metadata is available. Check the summary tables against the published version at the same time. Add a model download only when the matching detector checkpoint, configuration, source snapshot and evaluation log can be provided together.

## Continuous integration

`.github/workflows/release-checks.yml` validates release Python/shell syntax, configuration inheritance, local documentation links and the exported source structure. These checks run without a GPU on pushes and pull requests.

The workflow can also be launched manually with `build_image=true` to build the CUDA image. This optional job imports the compiled modules inside the image but does not execute GPU kernels. It does not publish an image. A full Docker build was not available on the development host used to prepare this release.

GPU checks, data preparation and benchmark runs remain explicit steps on a machine with the required GPU/data assets. The [validation record](VALIDATION.md) states which checks have actually run.
