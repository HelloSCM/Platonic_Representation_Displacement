#!/usr/bin/env python3
"""Export the paper's original PDF figures as resolution-independent SVGs."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--latex", type=Path, default=ROOT.parent / "latex")
args = parser.parse_args()
source = args.latex.resolve()
manifest = []
for pdf in sorted((source / "figures").rglob("*.pdf")):
    name = pdf.stem
    svg = ROOT / "assets" / "figures" / f"{name}.svg"
    original = ROOT / "assets" / "pdf" / pdf.name
    svg.parent.mkdir(parents=True, exist_ok=True)
    original.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
    shutil.copy2(pdf, original)
    manifest.append({"name": name, "source": str(pdf.relative_to(source)),
                     "svg": str(svg.relative_to(ROOT)), "pdf": str(original.relative_to(ROOT))})
(ROOT / "assets" / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

# Preserve Table 1 exactly, including missing entries and each model variant.
table = (source / "floats" / "table_01_shadowclip_results.tex").read_text()
columns = ["Food101", "CIFAR10", "CIFAR100", "SUN397", "Cars", "Aircraft", "DTD",
           "Pets", "Caltech101", "Flowers102", "ImageNet-1K", "COCO I2T", "COCO T2I",
           "Flickr30k I2T", "Flickr30k T2I", "MMVP"]
results = {}
pattern = re.compile(r"^(CLIP-FT|CLIP|DreamLIP|LiT|ShareLock|SAIL-NV2|w/o RKD|\{\\methodname\})\s*\n(.*?)(?=\\\\)", re.M | re.S)
for variant, block in [("base", table.split("ViT-Base")[1].split("ViT-Large")[0]),
                       ("large", table.split("ViT-Large")[1])]:
    rows = []
    for match in pattern.finditer(block):
        name = match[1].replace(r"{\methodname}", "ShadowCLIP")
        values = re.sub(r"\\textbf\{([^}]+)\}", r"\1", match[2]).strip().lstrip("&").split("&")
        values = [None if v.strip() == "-" else float(v.strip()) for v in values]
        if len(values) != len(columns):
            raise ValueError(f"Unexpected column count: {variant} {name}")
        rows.append({"method": name, "values": values})
    if not rows or rows[-1]["method"] != "ShadowCLIP":
        raise ValueError(f"Could not parse Table 1: {variant}")
    results[variant] = rows
(ROOT / "assets" / "results.json").write_text(json.dumps({"columns": columns, "variants": results}, indent=2) + "\n")
print(f"Exported {len(manifest)} original figures and {sum(map(len, results.values()))} table rows.")
