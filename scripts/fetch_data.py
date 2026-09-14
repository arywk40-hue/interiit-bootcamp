"""Download pinned public data, keeping source text out of Git.

Run from the repository root: python3 scripts/fetch_data.py
Only data is downloaded; no downloaded Python code is executed.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote


def at(base, *names):
    return [(base + quote(name), Path(name).name) for name in names]


SOURCES = {
    "sentiment": at(
        "https://huggingface.co/datasets/RTT1/SentiMix/resolve/205f0391fc1f10320ec3c10708eaa27e88db04c7/",
        "train_14k_split_conll.txt", "dev_3k_split_conll.txt",
        "Hindi_test_unalbelled_conll_updated.txt", "test_labels_hinglish.txt", "README.md",
    ),
    "intent": at(
        "https://raw.githubusercontent.com/google-research-datasets/Hinglish-TOP-Dataset/fdd3998a6573130659bfa1ce4b1ebe698df2bf3a/",
        "LICENSE.md", *[f"Dataset/Human Annotated Data/{split}.tsv"
                        for split in ("train", "validation", "test")],
    ),
    "qa": at(
        "https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/3489317a587da66c92e44d5778b20e5461de695f/",
        "COPYING", "README", "code_mixed_qa_train.json",
    ),
    "pretrain": [
        ("https://huggingface.co/datasets/LingoIITGN/PHINC/resolve/44b5471df75e508b9d6fcdbcc93df04bb1e84056/PHINC.csv",
         "PHINC.csv"),
        ("https://huggingface.co/datasets/LingoIITGN/PHINC/resolve/44b5471df75e508b9d6fcdbcc93df04bb1e84056/README.md",
         "README.md"),
        ("https://huggingface.co/datasets/saidutta69/Gupshup/resolve/2289a2761c7725b700394636c7a8962c0ad724a6/data/train.jsonl",
         "gupshup_train.jsonl"),
        ("https://huggingface.co/datasets/saidutta69/Gupshup/resolve/2289a2761c7725b700394636c7a8962c0ad724a6/README.md",
         "GUPSHUP_README.md"),
    ],
}


def main():
    manifest_path = Path("data/source_manifest.json")
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for task, files in SOURCES.items():
        folder = Path("data/raw") / task
        folder.mkdir(parents=True, exist_ok=True)
        manifest[task] = {}
        for url, target in files:
            path = folder / target
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".part")
                subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "120",
                                url, "-o", str(temporary)], check=True)
                temporary.replace(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = previous.get(task, {}).get(path.name, {}).get("sha256")
            if expected and digest != expected:
                raise ValueError(f"Checksum mismatch: {path}; restore the pinned source file")
            manifest[task][path.name] = {"url": url, "sha256": digest, "bytes": path.stat().st_size}
            print(f"Verified {path}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Official GupShup h2h summaries remain request-only; the public Gupshup chat file is MLM text only.")


if __name__ == "__main__":
    main()
