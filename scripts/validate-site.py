#!/usr/bin/env python3
"""Validate local assets, fragment links, original PDFs, and result-table data."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.refs = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            if values["id"] in self.ids:
                self.errors.append(f"Duplicate id: {values['id']}")
            self.ids.add(values["id"])
        for attr in ("src", "href"):
            if attr in values:
                self.refs.append(values[attr])
        if tag == "img" and "alt" not in values:
            self.errors.append("Image without alt text")

document = Document()
document.feed((ROOT / "index.html").read_text())
errors = document.errors
for ref in document.refs:
    parts = urlsplit(ref)
    if parts.scheme or parts.netloc:
        continue
    if parts.path and not (ROOT / unquote(parts.path)).is_file():
        errors.append(f"Missing file: {ref}")
    if not parts.path and parts.fragment and parts.fragment not in document.ids:
        errors.append(f"Missing fragment: {ref}")
for path in re.findall(r"url\(['\"]?([^)'\"]+)", (ROOT / "styles.css").read_text()):
    if not (ROOT / path).is_file():
        errors.append(f"Missing CSS asset: {path}")
manifest = json.loads((ROOT / "assets" / "figure-manifest.json").read_text())
for item in manifest:
    ET.parse(ROOT / item["svg"])
    with (ROOT / item["pdf"]).open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            errors.append(f"Invalid PDF: {item['pdf']}")
data = json.loads((ROOT / "assets" / "results.json").read_text())
for variant, rows in data["variants"].items():
    for row in rows:
        if len(row["values"]) != len(data["columns"]):
            errors.append(f"Table column mismatch: {variant} {row['method']}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {len(document.refs)} local/external references, {len(document.ids)} unique IDs, "
      f"{len(manifest)} SVG/PDF figure pairs, two result-table variants, and all CSS assets.")
