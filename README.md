# AutoAnnotSC

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python)
![AI](https://img.shields.io/badge/AI-Anthropic%20Claude-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Scanpy-blue?style=flat-square)
![Data](https://img.shields.io/badge/Data-Synapse-lightgrey?style=flat-square)
![Literature](https://img.shields.io/badge/Validation-PubMed%20MCP-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

Automated scRNA-seq cell type annotation pipeline. Downloads data from Synapse or accepts local files, runs standard preprocessing, annotates clusters using [CellAnnotator](https://github.com/LucasESBS/cell-annotator) + Claude, and validates marker genes against PubMed — all from a single command.

---

## Pipeline overview

```
Input (Synapse ID / local file / pbmc3k)
  → [1] Download / load
  → [2] QC & preprocessing (filter, normalize, PCA, UMAP, Leiden clustering)
  → [3] Cell type annotation  (CellAnnotator + Claude Sonnet)
  → [4] Marker gene validation (PubMed MCP)
  → [5] Save outputs (.h5ad, .json report, UMAP .png, summary .txt)
```

---

## Requirements

### Python packages

```
scanpy
anndata
cell-annotator
synapseclient
python-dotenv
requests
leidenalg
```

Install via conda/pip:

```bash
conda create -n cellannotator python=3.11
conda activate cellannotator
pip install scanpy anndata cell-annotator synapseclient python-dotenv requests leidenalg
```

### API keys

Create a `.env` file in the directory where you run the script:

```
ANTHROPIC_API_KEY=sk-ant-...
SYNAPSE_AUTH_TOKEN=eyJ...     # only needed for --synapse downloads
```

- **Anthropic API key**: get one at [console.anthropic.com](https://console.anthropic.com)
- **Synapse personal access token**: generate one at [synapse.org](https://www.synapse.org) → your profile → Access Tokens

---

## Usage

```
python scrna_cellranger_annotation.py [--synapse SYN_ID | --input FILE]
                                      [--tissue TISSUE]
                                      [--species SPECIES]
                                      [--resolution FLOAT]
```

| Argument | Default | Description |
|---|---|---|
| `--synapse` | — | Synapse entity ID (project, folder, or file) |
| `--input` | — | Path to local `.h5ad`, `.h5`, or 10x MTX folder |
| `--tissue` | `blood` | Tissue type passed to CellAnnotator |
| `--species` | `human` | Species passed to CellAnnotator (`human` or `mouse`) |
| `--resolution` | `0.5` | Leiden clustering resolution |

`--synapse` and `--input` are mutually exclusive. If neither is provided, the pipeline runs on the built-in `pbmc3k` dataset (useful for testing).

---

## Example commands

### 1. Built-in PBMC3k test (no data needed)

```bash
python scrna_cellranger_annotation.py --tissue blood --species human
```

Downloads the 2,700-cell PBMC3k dataset from scanpy's built-in cache and runs the full pipeline. Good for verifying your environment is set up correctly.

---

### 2. Local `.h5ad` file

```bash
python scrna_cellranger_annotation.py \
  --input data/my_sample.h5ad \
  --tissue lung \
  --species human
```

---

### 3. Local 10x CellRanger `.h5` file

```bash
python scrna_cellranger_annotation.py \
  --input data/filtered_feature_bc_matrix.h5 \
  --tissue brain \
  --species mouse
```

---

### 4. Synapse file entity

```bash
python scrna_cellranger_annotation.py \
  --synapse syn12345678 \
  --tissue blood \
  --species human
```

Authenticates with Synapse using `SYNAPSE_AUTH_TOKEN`, downloads the file to `data/`, and runs the pipeline.

---

### 5. Synapse project or folder (auto-discovers 10x MTX data)

```bash
# Human blood — project with multiple samples
python scrna_cellranger_annotation.py \
  --synapse syn22255433 \
  --tissue blood \
  --species human

# Mouse blood — folder containing MTX files
python scrna_cellranger_annotation.py \
  --synapse syn22255436 \
  --tissue blood \
  --species mouse
```

When given a project or folder ID, the script recursively searches for the first 10x MTX trio (`barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz`). If the folder name contains the species string, it is preferred over others. Only MTX files are downloaded — BAM files and FASTQs are skipped automatically.

---

### 6. Adjust clustering resolution

```bash
python scrna_cellranger_annotation.py \
  --input data/my_sample.h5ad \
  --tissue liver \
  --species mouse \
  --resolution 0.8
```

Higher resolution → more clusters. Default is `0.5`.

---

## Outputs

All outputs are written relative to the working directory, prefixed with `{synapse_id}_{species}` (or `{filename}_{species}` for local inputs):

| File | Description |
|---|---|
| `outputs/{run_id}_annotated.h5ad` | Processed AnnData with `cell_type_predicted` in `.obs` |
| `outputs/{run_id}_report.json` | JSON report: cell counts, clusters, marker genes, PubMed notes |
| `outputs/{run_id}_summary.txt` | Plain-text summary |
| `figures/umap_{run_id}.png` | UMAP coloured by Leiden cluster and predicted cell type |

Example for `--synapse syn22255436 --species mouse`:
```
outputs/syn22255436_mouse_annotated.h5ad
outputs/syn22255436_mouse_report.json
outputs/syn22255436_mouse_summary.txt
figures/umap_syn22255436_mouse.png
```

---

## Notes

- Mouse mitochondrial genes are detected with the `MT-` prefix (human) or `mt-` prefix (mouse). QC removes cells with >20% mitochondrial reads.
- CellAnnotator queries Claude Sonnet for cell type labels; each run makes several API calls.
- PubMed validation uses Anthropic's remote MCP client (`mcp-client-2025-04-04` beta) to search literature for each marker gene set.
- Downloaded Synapse files are cached in `data/` and reused on subsequent runs.
