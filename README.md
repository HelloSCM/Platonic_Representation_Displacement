# Displacement Geometry — Project Page

The English project page for **Displacement Geometry Captures Platonic Shared Reality Across Models and Modalities**, accepted at NeurIPS 2026.

- [Paper](https://arxiv.org/abs/2609.24209)
- [Code](https://github.com/HelloSCM/Platonic_Representation_Displacement)
- [Project page](https://helloscm.github.io/Platonic_Representation_Displacement/)

The website lives on `gh-pages`; experiment code lives on `main`. The two branches are maintained separately.

## Local preview

There are no frontend dependencies or build steps. From the root of a `gh-pages` checkout, run:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open [the local preview](http://127.0.0.1:8000/). Use HTTP rather than opening the HTML file directly so the browser can load the result-table JSON.

## Files

- `index.html`: paper overview, authors, resource links, figures, methods, results, and citation.
- `styles.css`: responsive layout, local fonts, navigation, figures, and background callout.
- `app.js`: Base/Large result switching, full-table view, and citation copying.
- `assets/figures/`: 27 vector SVG figures exported from the original paper PDFs.
- `assets/pdf/`: the 27 original figure PDFs retained as source assets; the page displays SVGs.
- `assets/results.json`: all 16 benchmark columns extracted from LaTeX Table 1.
- `assets/figure-manifest.json`: mappings between displayed figures and their LaTeX source files.
- `assets/fonts/`: local Inter fonts and the SIL Open Font License.
- `assets/metaphor.png`: the map illustration.
- `citation.bib`: BibTeX citation.
- `scripts/social-card.html`: source for the 1200 × 630 social preview at `assets/social-preview.png`.

## Updating figures and results

Install Poppler's `pdftocairo` and use Python 3 and Node.js for the checks below:

```bash
python scripts/prepare-assets.py --latex ../latex
python scripts/validate-site.py
node --check app.js
```

Replace `../latex` with the paper's LaTeX source directory. The export script converts the original PDFs without redrawing curves and extracts Table 1, preserving missing entries and the Base/Large variants. The `w/o LG` row denotes the ablation without the Language Guider.

When updating results, also synchronize the static ViT-Large summary in `index.html`. Update the asset version parameters in `index.html` and the JSON fetch URL in `app.js` as needed so browsers load the new content.

The page follows Metaphor → Geometry → Experiments → Application → Citation. Experiments contains Parallelism, Orthogonality, Modality, Dynamics, and Decomposition. A short background callout beside the map links to the [Platonic Representation Hypothesis project](https://phillipi.github.io/prh/).

## GitHub Pages

Publish changes from the `gh-pages` checkout:

```bash
git add index.html styles.css app.js assets citation.bib README.md scripts robots.txt sitemap.xml .nojekyll .gitignore
git commit -m "Update project page"
git push origin gh-pages
```

For the initial deployment, configure **Settings → Pages → Build and deployment**:

- Source: **Deploy from a branch**
- Branch: **gh-pages**
- Folder: **/(root)**

Subsequent pushes to `gh-pages` trigger deployment. `.nojekyll` enables direct serving of the static files.

The page includes canonical URLs, search metadata, social preview metadata, and scholarly citation metadata. If the repository or site URL changes, update `index.html`, `robots.txt`, and `sitemap.xml` accordingly.

## Content and design notes

The layout takes inspiration from [Visual Abstention](https://visual-abstention.github.io/), using this paper's own text, figures, and interactions. The site has no frontend framework, analytics trackers, or external runtime dependencies.

On desktop, the map appears on the left and the explanation and PRH background callout appear on the right. On smaller screens, these elements stack vertically. Bold opening findings flow directly into their supporting explanations in the same paragraph, and metric names stay together when text wraps.

The ShadowCLIP description distinguishes training from inference. Compute comparisons distinguish cached training from total computation including precomputation, and explicitly note that inference still uses DINO. Results are presented in an interactive table with a static fallback.

Rights to the paper's figures and research content remain with their authors. Font licensing is documented in `assets/fonts/OFL.txt`.
