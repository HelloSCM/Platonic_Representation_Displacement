# Displacement Geometry — Project Page

论文 **Displacement Geometry Captures Platonic Shared Reality Across Models and Modalities** 的英文项目介绍页。

- 论文：https://arxiv.org/abs/2609.24209
- 代码：https://github.com/HelloSCM/Platonic_Representation_Displacement
- GitHub Pages 目标：https://helloscm.github.io/Platonic_Representation_Displacement/
- 页面分支：`gh-pages`；论文代码分支：`main`。网页与正在整理的代码独立维护。

## 本地预览

无需安装前端依赖，也没有构建步骤：

```bash
cd /home/scm25/research/platonic/demo
python -m http.server 8000 --bind 127.0.0.1
```

浏览器打开 `http://127.0.0.1:8000/`。请使用 HTTP 预览，以便浏览器读取结果表格的 JSON 数据。

## 文件

- `index.html`：论文介绍、作者、资源链接、图表、方法、结果和引用。
- `styles.css`：响应式样式、本地字体、移动端导航、图表放大视图。
- `app.js`：Base／Large 结果切换、完整表格、图表放大、引用复制。
- `assets/figures/`：从论文 PDF 原图导出的 27 份矢量 SVG。
- `assets/pdf/`：27 份原始 PDF 图，可从页面或放大视图下载。
- `assets/results.json`：从 LaTeX Table 1 提取的数据，包含全部 16 个 benchmark。
- `assets/figure-manifest.json`：图表与原始 LaTeX 文件的对应关系。
- `assets/fonts/`：本地 Inter 字体及其 SIL Open Font License；页面不依赖外部字体服务。
- `citation.bib`：BibTeX 引用。
- `scripts/social-card.html`：1200 × 630 分享预览图的源文件；生成后的 PNG 位于 `assets/social-preview.png`。

## 更新图表和结果

需要 Poppler 的 `pdftocairo` 和 Python 标准库：

```bash
python scripts/prepare-assets.py --latex ../latex
python scripts/validate-site.py
node --check app.js
```

脚本直接转换 PDF，保留矢量内容；不对论文曲线重绘。页面按 Metaphor → Geometry → Experiments → Application → Citation 组织。Experiments 包括 Parallelism、Orthogonality、Modality、Dynamics、Decomposition 五组。`assets/metaphor.png` 为地图比喻插图。

## GitHub Pages

此目录自身是 `gh-pages` 分支的独立工作目录。后续更新：

```bash
git add index.html styles.css app.js assets citation.bib README.md scripts robots.txt sitemap.xml .nojekyll .gitignore
git commit -m "Update project page"
git push origin gh-pages
```

首次发布时，在仓库 **Settings → Pages → Build and deployment** 中设置：

- Source：**Deploy from a branch**
- Branch：**gh-pages**
- Folder：**/(root)**

保存后，GitHub 会自动部署；之后推送 `gh-pages` 即更新网页。`.nojekyll` 保证静态文件直接发布。

页面已配置正式 URL、搜索引擎元数据、社交分享图和学术引用元数据。若以后更换仓库或 URL，同时修改 `index.html` 的 canonical / Open Graph、`robots.txt` 和 `sitemap.xml`。

## 内容与设计说明

版式参考 [Visual Abstention](https://visual-abstention.github.io/)，以本论文的文字、图表和交互重新实现，页脚保留设计致谢。项目页没有框架、分析追踪器或外部运行时依赖。


性能说明区分缓存后的训练阶段与包含缓存的总计算量，也说明推理时仍使用 DINO。所有论文图表与研究内容的权利归原作者所有；字体许可见 `assets/fonts/OFL.txt`。
