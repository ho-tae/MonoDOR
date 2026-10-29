"""Export a source-only snapshot without Git history, papers, data or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DOCS = {"README.md", "docs/REPRODUCING.md", "docs/README_ko.md",
               "docs/RESULTS.md", "docs/RELEASE.md", "docs/VALIDATION.md"}
ROOT_FILES = {"LICENSE", "setup.py", "requirements.txt", ".gitignore", ".dockerignore"}
EXCLUDED_SUFFIXES = {".pdf", ".docx", ".pth", ".pt", ".pkl", ".bin", ".zip", ".so", ".pyc"}


def allowed(name):
    path = Path(name)
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if any(part in {"__pycache__", "build", "dist", ".git"} or part.endswith(".egg-info") for part in path.parts):
        return False
    if path.name.startswith(".env") and path.name != ".env.example":
        return False
    return (name in PUBLIC_DOCS | ROOT_FILES
            or name.startswith(("pcdet/", "VMamba/", "tools/", "configs/release/",
                                "scripts/release/", "requirements/"))
            or name in {"docker/release.Dockerfile", "docker/compose.yaml", "docker/.env.example", "docker/README.md",
                        ".github/workflows/release-checks.yml"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release-dist/source")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; select a new --output path")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    candidates = set(tracked) | PUBLIC_DOCS | ROOT_FILES
    for directory in ("configs/release", "scripts/release", "requirements", "docker", ".github/workflows"):
        candidates.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob("*") if p.is_file())
    selected = [name for name in sorted(candidates) if name and allowed(name) and (ROOT / name).is_file()]
    for name in selected:
        if (ROOT / name).is_symlink():
            raise ValueError(f"Source symlink requires explicit review: {name}")
    destination.mkdir(parents=True)
    hashes = {}
    for name in selected:
        source, target = ROOT / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if name == ".gitignore":
            # Previously tracked YAML/shell files would otherwise disappear
            # when a user initializes a fresh Git repository from the export.
            exceptions = ["!/" + item for item in selected if Path(item).suffix in {".yaml", ".sh"}]
            with target.open("a") as handle:
                handle.write("\n# Preserve exported configuration/scripts in a new repository.\n")
                handle.write("\n".join(exceptions) + "\n")
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "includes_worktree_changes": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "description": "Current working-tree sources; base commit alone does not identify this snapshot.",
        "sha256": hashes,
    }
    (destination / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(selected)} files to {destination}; no commit or push performed.")


if __name__ == "__main__":
    main()
