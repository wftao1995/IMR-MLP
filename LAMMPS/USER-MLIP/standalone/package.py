#!/usr/bin/env python3
"""Export only manifest-listed standalone sources and record their SHA-256.

Refuses to overwrite an existing destination. No compiler, GPU, or IMR-MLP library
is needed. A package can be exported again using its bundled manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory/"manifest.json").read_text())
    bundled = (directory/"vendor/mtp_metadata.h").is_file()
    root = directory.parent if bundled else directory.parents[2]
    destination = args.output.resolve()
    if destination.exists(): raise SystemExit(f"Refusing to overwrite {destination}")
    paths = []
    seen = set()
    for entry in manifest["files"]:
        relative = entry["destination"]
        source = root/(relative if bundled else entry["source"])
        if Path(relative).is_absolute() or ".." in Path(relative).parts or relative in seen:
            raise ValueError(f"Invalid or duplicate destination: {relative}")
        if not source.is_file(): raise FileNotFoundError(source)
        seen.add(relative); paths.append((source,relative))
    destination.mkdir(parents=True)
    hashes = {}
    for source,relative in paths:
        target = destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    revision = None
    if bundled and (root/"source-record.json").is_file():
        revision = json.loads((root/"source-record.json").read_text()).get("source_revision")
    if not bundled:
        query = subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],text=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        if query.returncode==0: revision=query.stdout.strip()
    record = {"format":"mtp-standalone-source-v1","source_revision":revision,"sha256":hashes}
    (destination/"source-record.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"package":str(destination),"files":len(hashes),"source_revision":revision},indent=2))

if __name__ == "__main__": main()
