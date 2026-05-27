# AutoAnnotSC

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python)
![AI](https://img.shields.io/badge/AI-LangChain%20%2B%20Claude-orange?style=flat-square)
![Validation](https://img.shields.io/badge/Validation-HuggingFace%20BioBERT%2FBioGPT-yellow?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Scanpy-blue?style=flat-square)
![Data](https://img.shields.io/badge/Data-Synapse-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

Automated scRNA-seq cell type annotation pipeline. Downloads data from Synapse or accepts local files, runs standard preprocessing, annotates clusters using [CellAnnotator](https://github.com/LucasESBS/cell-annotator) + **LangChain/Claude**, and validates marker genes locally with a **HuggingFace biomedical model (BioBERT / BioGPT)** — all from a single command.

---

## Pipeline overview

```
Input (Synapse ID / local file / pbmc3k)
  → [1] Download / load
  → [2] QC & preprocessing (filter, normalize, PCA, UMAP, Leiden clustering)
  → [3] Cell type annotation  (CellAnnotator + LangChain ChatAnthropic)
  → [4] Marker gene validation (HuggingFace BioBERT or BioGPT — local, no API call)
  → [5] Save outputs (.h5ad, .json report, UMAP .png, summary .txt)
```

---

## What changed from v1

| Component | Before | After |
|---|---|---|
| LLM client | Raw `requests` to Anthropic REST API | **LangChain** `ChatAnthropic` + LCEL chain |
| Marker validation | Claude via PubMed MCP (API call) | **HuggingFace** BioBERT / BioGPT (local inference) |
| Env vars needed | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` (Synapse token unchanged) |

---

## Requirements

### Python packages

```
scanpy
anndata
cell-annotator
synapseclient
python-dotenv
leidenalg
langchain-anthropic
langchain-core
transformers
torch
```

Install via conda/pip:

```bash
conda create -n cellannotator python=3.11
conda activate cellannotator
pip install scanpy anndata cell-annotator synapseclient python-dotenv leidenalg \
            langchain-anthropic langchain-core transformers torch
```

> **GPU optional** — the HuggingFace model runs on CPU by default; set `device=0` in `BiomedicalMarkerValidator.__init__` or ensure CUDA is available for faster inference.

### API keys

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
SYNAPSE_AUTH_TOKEN=eyJ...          # only needed for --synapse downloads
HF_BIOMEDICAL_MODEL=dmis-lab/biobert-base-cased-v1.2   # optional override
```

- **Anthropic API key**: [console.anthropic.com](https://console.anthropic.com)
- **Synapse token**: [synapse.org](https://www.synapse.org) → profile → Access Tokens

---

## Supported biomedical models

| Model | HuggingFace ID | Task | Notes |
|---|---|---|---|
| **BioBERT** (default) | `dmis-lab/biobert-base-cased-v1.2` | fill-mask | Fast, good gene-symbol recall |
| BioBERT Large | `dmis-lab/biobert-large-cased-v1.1` | fill-mask | Higher accuracy, more RAM |
| SciBERT | `allenai/scibert_scivocab_cased` | fill-mask | Broader science coverage |
| **BioGPT** | `microsoft/BioGPT` | text-generation | Generative; good for context |
| BioGPT-Large | `microsoft/BioGPT-Large` | text-generation | Best generative quality |

Set via `--hf-model` flag or `HF_BIOMEDICAL_MODEL` env var.

---

## Usage

```
python scrna_cellranger_annotation.py [--synapse SYN_ID | --input FILE]
                                      [--tissue TISSUE]
                                      [--species SPECIES]
                                      [--resolution FLOAT]
                                      [--hf-model HF_MODEL_ID]
```

| Argument | Default | Description |
|---|---|---|
| `--synapse` | — | Synapse entity ID |
| `--input` | — | Path to local `.h5ad`, `.h5`, or 10x MTX folder |
| `--tissue` | `blood` | Tissue type passed to CellAnnotator |
| `--species` | `human` | Species (`human` or `mouse`) |
| `--resolution` | `0.5` | Leiden clustering resolution |
| `--hf-model` | env / BioBERT | HuggingFace model ID for marker validation |

---

## Example commands

### 1. Built-in PBMC3k test (BioBERT validation)

```bash
python scrna_cellranger_annotation.py --tissue blood --species human
```

### 2. Use BioGPT-Large instead of BioBERT

```bash
python scrna_cellranger_annotation.py \
  --input data/my_sample.h5ad \
  --tissue lung \
  --hf-model microsoft/BioGPT-Large
```

### 3. Local 10x CellRanger `.h5` file

```bash
python scrna_cellranger_annotation.py \
  --input data/filtered_feature_bc_matrix.h5 \
  --tissue brain \
  --species mouse
```

### 4. Synapse project (auto-discovers MTX data)

```bash
python scrna_cellranger_annotation.py \
  --synapse syn22255433 \
  --tissue blood \
  --species human
```

### 5. Higher clustering resolution + SciBERT

```bash
python scrna_cellranger_annotation.py \
  --input data/my_sample.h5ad \
  --tissue liver \
  --species mouse \
  --resolution 0.8 \
  --hf-model allenai/scibert_scivocab_cased
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/{run_id}_annotated.h5ad` | Processed AnnData with `cell_type_predicted` in `.obs` |
| `outputs/{run_id}_report.json` | JSON report: cell counts, clusters, marker genes, HF validation scores |
| `outputs/{run_id}_summary.txt` | Plain-text summary with validation verdict per cell type |
| `figures/umap_{run_id}.png` | UMAP coloured by Leiden cluster and predicted cell type |

### Validation verdicts

Each annotated cell type receives a verdict based on the mean model confidence across its top-3 marker genes:

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | Markers are well-supported by the biomedical model's training corpus |
| `UNCERTAIN` | Moderate support — review manually |
| `FLAGGED` | Low support — potential mis-annotation or novel marker set |

---

## Notes

- Mouse mitochondrial genes use `mt-` prefix; human use `MT-`. QC removes cells with >20% mitochondrial reads.
- Cell annotation calls Claude Sonnet via **LangChain `ChatAnthropic`**; each run makes several API calls to Anthropic.
- Marker validation runs **fully locally** via HuggingFace Transformers — no extra API key or network call required after the model downloads from the Hub.
- Downloaded HuggingFace models are cached in `~/.cache/huggingface/hub` and reused on subsequent runs.
- Downloaded Synapse files are cached in `data/` and reused on subsequent runs.
