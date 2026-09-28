import argparse
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def records(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    descriptions = list(records(root / "data/docci_descriptions.jsonlines"))
    assert Counter(x["split"] for x in descriptions)["train"] == 9647
    clusters = {}
    for row in records(root / "data/docci_metadata.jsonlines"):
        clusters[row["example_id"]] = row.get("cluster_id")
    train_rows = [x for x in descriptions if x["split"] == "train"]
    def cluster_key(row):
        cluster = clusters.get(row["example_id"])
        return str(cluster) if cluster is not None else row["example_id"]
    groups = defaultdict(list)
    for row in train_rows:
        groups[cluster_key(row)].append(row)
    rng = random.Random(args.seed)
    group_keys = list(groups)
    rng.shuffle(group_keys)
    val = []
    val_keys = set()
    for key in group_keys:
        if len(val) >= args.val_count:
            break
        if len(groups[key]) <= args.val_count - len(val):
            val.extend(groups[key])
            val_keys.add(key)
        else:
            remaining = args.val_count - len(val)
            val.extend(groups[key][:remaining])
            val_keys.add(key)
            break
    assert len(val) == args.val_count, len(val)
    pool = [row for key in group_keys if key not in val_keys for row in groups[key]]
    assert len(pool) >= args.train_count, len(pool)
    train = rng.sample(pool, args.train_count)
    assert not {x["example_id"] for x in train} & {x["example_id"] for x in val}
    assert not {cluster_key(x) for x in train} & {cluster_key(x) for x in val}
    selected = {x["image_file"] for x in train + val}
    image_dir = root / "images"
    image_dir.mkdir(exist_ok=True)
    if args.manifest_only:
        found = {p.name for p in image_dir.glob("*.jpg")}
    else:
        found = set()
        with tarfile.open(root / "data/docci_images.tar.gz", "r:gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if name not in selected or not member.isfile():
                    continue
                source = archive.extractfile(member)
                assert source is not None
                target = image_dir / name
                with target.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                found.add(name)
    assert found == selected, {"missing": sorted(selected - found)[:20]}
    verify_rows = (train + val)[:32] if args.manifest_only else train + val
    for row in verify_rows:
        path = image_dir / row["image_file"]
        with Image.open(path) as image:
            image.verify()
    manifest = {
        "seed": args.seed,
        "source": "https://storage.googleapis.com/docci/data/",
        "train": [{k: x[k] for k in ("example_id", "image_file", "description")} for x in train],
        "val": [{k: x[k] for k in ("example_id", "image_file", "description")} for x in val],
        "official_test_used": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": len(train), "val": len(val), "image_files": len(found), "train_clusters": len({cluster_key(x) for x in train}), "val_clusters": len(val_keys)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
