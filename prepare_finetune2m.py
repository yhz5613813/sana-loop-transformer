"""Index one text-to-image-2M WebDataset shard without extracting images."""

import argparse
import hashlib
import json
import random
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--shard", default="data/finetune2m/data_512_2M/data_000000.tar")
    parser.add_argument("--val-count", type=int, default=500)
    args = parser.parse_args()
    root = args.root
    shard = root / args.shard
    assert shard.is_file(), shard
    images = {}
    prompts = {}
    with tarfile.open(shard) as archive:
        for member in archive:
            stem, suffix = Path(member.name).stem, Path(member.name).suffix.lower()
            if suffix in (".jpg", ".jpeg", ".png", ".webp"):
                assert stem not in images, stem
                images[stem] = {"image_file": member.name, "tar_file": args.shard,
                                "offset": member.offset_data, "size": member.size}
            elif suffix == ".json":
                with archive.extractfile(member) as file:
                    prompt = json.load(file).get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    prompts[stem] = prompt.strip()
    rows = []
    for stem in sorted(images.keys() & prompts.keys()):
        rows.append({"example_id": stem, "description": prompts[stem], **images[stem],
                     "prompt_sha256": hashlib.sha256(prompts[stem].encode()).hexdigest()})
    assert len(rows) > args.val_count + 8, len(rows)
    groups = {}
    for row in rows:
        groups.setdefault(row["prompt_sha256"], []).append(row)
    keys = list(groups)
    random.Random(20260928).shuffle(keys)
    val_keys = set()
    count = 0
    for key in keys:
        if count >= args.val_count:
            break
        val_keys.add(key)
        count += len(groups[key])
    train = [row for row in rows if row["prompt_sha256"] not in val_keys]
    val = [row for row in rows if row["prompt_sha256"] in val_keys]
    train = train[:len(train) // 8 * 8]
    assert not ({r["prompt_sha256"] for r in train} & {r["prompt_sha256"] for r in val})
    manifest = {"source": "jackyhate/text-to-image-2M", "shard": args.shard,
                "seed": 20260928, "train": train, "val": val,
                "indexed_images": len(images), "indexed_prompts": len(prompts)}
    out = root / "manifest_finetune2m.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"train": len(train), "val": len(val), "images": len(images),
                      "prompts": len(prompts), "unique_prompt_groups": len(groups),
                      "manifest": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


