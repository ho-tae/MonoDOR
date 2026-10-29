"""CPU-only release checks for portable configs, entry points and documentation."""
import ast
from pathlib import Path
import re
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[2]


def check_mmconfig(path, seen=None):
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return
    seen.add(path)
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_base_" for t in node.targets):
            bases = ast.literal_eval(node.value)
            for base in ([bases] if isinstance(bases, str) else bases):
                check_mmconfig(path.parent / base, seen)


def main():
    for path in (ROOT / "scripts/release").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for path in (ROOT / "scripts/release").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    cfg = yaml.safe_load((ROOT / "configs/release/vmamba_kitti.yaml").read_text())
    assert "_BASE_CONFIG_" not in cfg["DATA_CONFIG"], "Public dataset configuration must be self-contained"
    assert cfg["OPTIMIZATION"]["NUM_EPOCHS"] == 80
    assert cfg["MODEL"]["VFE"]["FFN"]["DISCRETIZE"]["num_bins"] == 150
    args = cfg["MODEL"]["VFE"]["FFN"]["DDN"]["ARGS"]
    for path in (cfg["DATA_CONFIG"]["DATA_PATH"], args["pretrained_path"], args["config_path"]):
        assert not Path(path).is_absolute(), path
    check_mmconfig(ROOT / "tools" / args["config_path"])
    compose = yaml.safe_load((ROOT / "docker/compose.yaml").read_text())
    assert compose["services"]["research"]["gpus"] == "all"
    from export_source import PUBLIC_DOCS
    for name in PUBLIC_DOCS | {"docker/README.md"}:
        path = ROOT / name
        for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            target = (path.parent / link.split("#")[0]).resolve()
            assert target.exists(), f"Broken local documentation link: {name}: {link}"
    print("Release source, shell syntax, config inheritance and documentation links passed.")


if __name__ == "__main__":
    main()
