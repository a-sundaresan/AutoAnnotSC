"""
scrna_cellranger_annotation.py — scRNA-seq Automated Analysis Journal
----------------------------------------------------------------------
Accepts:
  - Synapse ID  (downloads automatically via synapseclient)
  - Local .h5ad file
  - Local .h5   file (10x CellRanger format)

Stack:
  - LangChain (langchain-anthropic) for cluster annotation via CellAnnotator
  - HuggingFace Transformers (BioBERT / BioGPT) for local marker-gene validation
  - Scanpy for all preprocessing / clustering

Usage:
  conda activate cellannotator
  python scrna_cellranger_annotation.py --tissue blood --species human
  python scrna_cellranger_annotation.py --input data/myfile.h5ad --tissue lung
  python scrna_cellranger_annotation.py --synapse syn26522055 --tissue breast
"""

import os, sys, json, argparse, datetime, warnings
import scanpy as sc
import synapseclient
from dotenv import load_dotenv

# ── LangChain imports ────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── HuggingFace imports ──────────────────────────────────────────────
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")
SYNAPSE_TOKEN  = os.getenv("SYNAPSE_AUTH_TOKEN")
HF_MODEL       = os.getenv("HF_BIOMEDICAL_MODEL", "dmis-lab/biobert-base-cased-v1.2")
# Set to "microsoft/BioGPT-Large" to use BioGPT instead of BioBERT

os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)


# ── ARGUMENT PARSING ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="scRNA-seq Automated Analysis Journal")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synapse",    type=str,   help="Synapse ID e.g. syn26522055")
    src.add_argument("--input",      type=str,   help="Path to local .h5ad or .h5 file")
    p.add_argument("--tissue",       type=str,   default="blood", help="Tissue type (default: blood)")
    p.add_argument("--species",      type=str,   default="human", help="Species (default: human)")
    p.add_argument("--resolution",   type=float, default=0.5,     help="Leiden resolution (default: 0.5)")
    p.add_argument("--hf-model",     type=str,   default=None,
                   help="HuggingFace biomedical model ID (overrides HF_BIOMEDICAL_MODEL env var)")
    return p.parse_args()


# ── ENV CHECK ─────────────────────────────────────────────────────────

def check_env(use_synapse: bool):
    missing = []
    if not ANTHROPIC_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if use_synapse and not SYNAPSE_TOKEN:
        missing.append("SYNAPSE_AUTH_TOKEN")
    if missing:
        raise EnvironmentError(
            f"Missing required env vars: {', '.join(missing)}\n"
            f"Add them to your .env file and try again."
        )
    print("✅ Environment OK")


# ── LANGCHAIN LLM SETUP ──────────────────────────────────────────────

def build_llm_chain(system_prompt: str) -> object:
    """Return a LangChain LCEL chain: prompt | ChatAnthropic | StrOutputParser."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=ANTHROPIC_KEY,
        max_tokens=2000,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  "{user_input}"),
    ])
    return prompt | llm | StrOutputParser()


# ── HUGGINGFACE BIOMEDICAL VALIDATOR ─────────────────────────────────

class BiomedicalMarkerValidator:
    """
    Uses a local HuggingFace biomedical language model (BioBERT or BioGPT)
    to score and validate marker genes for annotated cell types.

    Strategy
    --------
    For each (cell_type, gene_list) pair we construct a cloze/fill-mask
    probe sentence and use masked-LM token probabilities (BioBERT) or
    text-generation perplexity (BioGPT) to produce a confidence score
    that reflects how well the model's biomedical training corpus associates
    the gene with the stated cell type.
    """

    BIOBERT_MODELS = {
        "dmis-lab/biobert-base-cased-v1.2",
        "dmis-lab/biobert-large-cased-v1.1",
        "allenai/scibert_scivocab_cased",
        "allenai/scibert_scivocab_uncased",
    }
    BIOGPT_MODELS = {
        "microsoft/BioGPT",
        "microsoft/BioGPT-Large",
        "microsoft/BioGPT-Large-PubMedQA",
    }

    def __init__(self, model_id: str = "dmis-lab/biobert-base-cased-v1.2"):
        self.model_id  = model_id
        self.is_biogpt = any(m in model_id for m in self.BIOGPT_MODELS)
        print(f"\n   Loading HuggingFace model: {model_id}")
        print(f"   Mode: {'text-generation (BioGPT)' if self.is_biogpt else 'fill-mask (BioBERT)'}")
        device = 0 if torch.cuda.is_available() else -1
        if self.is_biogpt:
            self._pipe = pipeline(
                "text-generation",
                model=model_id,
                device=device,
                max_new_tokens=50,
                do_sample=False,
            )
        else:
            self._pipe = pipeline(
                "fill-mask",
                model=model_id,
                device=device,
                top_k=50,
            )
        print("   Model ready ✅")

    # ------------------------------------------------------------------
    def _score_biobert(self, cell_type: str, gene: str) -> float:
        """
        Probe: '[MASK] is a marker gene of {cell_type} cells.'
        Return the probability that [MASK] is filled with the gene symbol.
        If the gene is not in the top-50 candidates, return 0.
        """
        probe = f"[MASK] is a well-known marker gene of {cell_type} cells."
        try:
            results = self._pipe(probe)
            for item in results:
                if item["token_str"].strip().upper() == gene.upper():
                    return round(item["score"], 4)
        except Exception:
            pass
        return 0.0

    def _score_biogpt(self, cell_type: str, gene: str) -> float:
        """
        Prompt BioGPT with a partial sentence and check if the gene symbol
        appears in the generated continuation.  Returns 1.0 if found, 0.0 if not.
        (Simple heuristic; for production use compute token log-probs instead.)
        """
        prompt = (
            f"In single-cell RNA sequencing studies, {gene} is a marker gene "
            f"for {cell_type}"
        )
        try:
            out = self._pipe(prompt)[0]["generated_text"]
            affirmative_phrases = [
                "is a marker", "marker gene", "highly expressed",
                "specifically expressed", "canonical marker",
            ]
            score = float(any(p in out.lower() for p in affirmative_phrases))
            return score
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def validate(self, marker_genes: dict, top_n: int = 3) -> dict:
        """
        Parameters
        ----------
        marker_genes : {cell_type: [gene, ...], ...}
        top_n        : how many top markers to test per cell type

        Returns
        -------
        validation : {
            cell_type: {
                "genes":        [str, ...],
                "scores":       {gene: float, ...},
                "mean_score":   float,
                "verdict":      "confirmed" | "uncertain" | "flagged"
            }, ...
        }
        """
        validation = {}
        for cell_type, genes in marker_genes.items():
            genes_to_test = genes[:top_n]
            scores = {}
            for gene in genes_to_test:
                if self.is_biogpt:
                    s = self._score_biogpt(cell_type, gene)
                else:
                    s = self._score_biobert(cell_type, gene)
                scores[gene] = s

            if scores:
                mean_score = round(sum(scores.values()) / len(scores), 4)
            else:
                mean_score = 0.0

            if self.is_biogpt:
                # BioGPT scores are binary 0/1
                verdict = (
                    "confirmed"  if mean_score >= 0.67 else
                    "uncertain"  if mean_score >= 0.33 else
                    "flagged"
                )
            else:
                # BioBERT probabilities are typically small (0-0.3 range for rare tokens)
                verdict = (
                    "confirmed"  if mean_score >= 0.01 else
                    "uncertain"  if mean_score >= 0.001 else
                    "flagged"
                )

            validation[cell_type] = {
                "genes":      genes_to_test,
                "scores":     scores,
                "mean_score": mean_score,
                "verdict":    verdict,
            }
        return validation

    # ------------------------------------------------------------------
    def summary_text(self, validation: dict) -> str:
        lines = [
            f"HuggingFace Biomedical Marker Validation  ({self.model_id})",
            "-" * 60,
        ]
        for cell_type, info in validation.items():
            gene_scores = ", ".join(
                f"{g}={s:.4f}" for g, s in info["scores"].items()
            )
            lines.append(
                f"{cell_type:<35s}  "
                f"mean={info['mean_score']:.4f}  "
                f"[{info['verdict'].upper()}]  "
                f"({gene_scores})"
            )
        confirmed = sum(1 for v in validation.values() if v["verdict"] == "confirmed")
        flagged   = sum(1 for v in validation.values() if v["verdict"] == "flagged")
        lines += [
            "-" * 60,
            f"Summary: {confirmed} confirmed, "
            f"{len(validation)-confirmed-flagged} uncertain, {flagged} flagged",
        ]
        return "\n".join(lines)


# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────

MTX_FILES = {
    "barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz",
    "barcodes.tsv",    "features.tsv",    "matrix.mtx",
}


def _find_mtx_folders(syn, parent_id, results=None):
    if results is None:
        results = []
    children = list(syn.getChildren(parent_id, includeTypes=["file", "folder"]))
    names = {c["name"].lower() for c in children if "FileEntity" in c["type"]}
    if any("barcodes" in n for n in names) and any("matrix" in n for n in names):
        results.append((parent_id, children))
    for c in children:
        if "Folder" in c["type"]:
            _find_mtx_folders(syn, c["id"], results)
    return results


def load_from_synapse(synapse_id: str, species: str = "human") -> sc.AnnData:
    print(f"\n[1/5] Downloading {synapse_id} from Synapse...")
    syn = synapseclient.Synapse()
    syn.login(authToken=SYNAPSE_TOKEN, silent=True)
    entity = syn.get(synapse_id, downloadFile=False)
    concrete = entity.get("concreteType", "")

    if "Project" in concrete or "Folder" in concrete:
        print(f"   '{entity['name']}' is a {concrete.split('.')[-1]} — searching for 10x MTX data...")
        all_mtx = _find_mtx_folders(syn, synapse_id)
        if not all_mtx:
            raise ValueError(f"No 10x MTX folder found under {synapse_id}")
        chosen_id, chosen_children = all_mtx[0]
        for fid, fchildren in all_mtx:
            folder_entity = syn.get(fid, downloadFile=False)
            if species.lower() in folder_entity["name"].lower():
                chosen_id, chosen_children = fid, fchildren
                break
        folder_entity = syn.get(chosen_id, downloadFile=False)
        dest = os.path.join("data", folder_entity["name"])
        os.makedirs(dest, exist_ok=True)
        print(f"   Downloading MTX folder: {folder_entity['name']} ({chosen_id})")
        for child in chosen_children:
            if "FileEntity" not in child["type"]:
                continue
            if child["name"].lower() not in MTX_FILES:
                continue
            syn.get(child["id"], downloadLocation=dest, ifcollision="overwrite.local")
        return load_file(dest)
    else:
        dl = syn.get(synapse_id, downloadLocation="data", ifcollision="overwrite.local")
        path = dl["path"] if "path" in dl else os.path.join("data", dl["name"])
        print(f"   Downloaded: {path}")
        return load_file(path)


def load_file(path: str) -> sc.AnnData:
    if os.path.isdir(path):
        print(f"   Format: 10x MTX folder — {path}")
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=True)
        adata.var_names_make_unique()
        adata.obs["sample"] = os.path.basename(path)
        print(f"   Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        return adata
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".h5ad":
        print("   Format: AnnData (.h5ad)")
        adata = sc.read_h5ad(path)
    elif ext in (".h5", ".hdf5"):
        print("   Format: 10x CellRanger (.h5) — converting to AnnData")
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
    elif ext in (".mtx", ".gz"):
        folder = os.path.dirname(path)
        print(f"   Format: 10x MTX folder — {folder}")
        adata = sc.read_10x_mtx(folder, var_names="gene_symbols", cache=True)
        adata.var_names_make_unique()
    else:
        raise ValueError(f"Unsupported file format: {ext}\nSupported: .h5ad, .h5, .hdf5, .mtx")
    adata.obs["sample"] = os.path.basename(path)
    print(f"   Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata


# ── STEP 2: QC & PREPROCESS ──────────────────────────────────────────

def preprocess(adata: sc.AnnData, resolution: float) -> sc.AnnData:
    print("\n[2/5] QC and preprocessing...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    before = adata.n_obs
    adata  = adata[adata.obs.pct_counts_mt < 20].copy()
    print(f"   Cells after QC: {adata.n_obs:,} (removed {before - adata.n_obs:,} high-mito)")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_pcs=20)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)
    n = adata.obs["leiden"].nunique()
    print(f"   Clusters: {n} (resolution={resolution})")
    return adata


# ── STEP 3: ANNOTATE (LangChain + CellAnnotator) ─────────────────────

def annotate(adata: sc.AnnData, tissue: str, species: str) -> tuple:
    """
    Cell type annotation using CellAnnotator backed by a LangChain
    ChatAnthropic chain rather than a raw requests call.
    """
    print("\n[3/5] Annotating cell types with CellAnnotator + LangChain/Claude...")
    from cell_annotator import CellAnnotator

    # CellAnnotator supports passing a pre-built LangChain LLM object
    # via the `llm` kwarg (>= cell-annotator 0.3).  For older versions
    # that only accept provider/model/api_key, we fall back gracefully.
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=ANTHROPIC_KEY,
        max_tokens=2000,
    )
    try:
        cell_ann = CellAnnotator(
            adata,
            species=species,
            tissue=tissue,
            cluster_key="leiden",
            sample_key=None,
            llm=llm,          # LangChain ChatModel interface
        )
    except TypeError:
        # Older cell-annotator — fall back to provider kwarg
        cell_ann = CellAnnotator(
            adata,
            species=species,
            tissue=tissue,
            cluster_key="leiden",
            sample_key=None,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key=ANTHROPIC_KEY,
        )

    cell_ann.annotate_clusters()

    cluster_labels = (
        adata.obs.groupby("leiden")["cell_type_predicted"]
        .first().to_dict()
    )

    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", n_genes=5)
    marker_genes = {}
    for cluster in sorted(adata.obs["leiden"].unique(), key=int):
        genes = (
            sc.get.rank_genes_groups_df(adata, group=cluster)
            .head(5)["names"].tolist()
        )
        label = cluster_labels.get(cluster, f"Cluster {cluster}")
        marker_genes[label] = genes
        print(f"   Cluster {cluster:>2} → {label:35s} | {', '.join(genes[:3])}")

    return adata, marker_genes


# ── STEP 4: HUGGINGFACE MARKER VALIDATION ────────────────────────────

def validate_with_hf(marker_genes: dict, hf_model_id: str) -> tuple:
    """
    Validate marker genes locally using a HuggingFace biomedical model.
    Returns (validation_dict, summary_text).
    """
    print(f"\n[4/5] Validating marker genes with HuggingFace ({hf_model_id})...")
    validator  = BiomedicalMarkerValidator(model_id=hf_model_id)
    validation = validator.validate(marker_genes, top_n=3)
    summary    = validator.summary_text(validation)
    print("\n" + summary)
    return validation, summary


# ── STEP 5: SAVE OUTPUTS ─────────────────────────────────────────────

def save_outputs(
    adata: sc.AnnData,
    marker_genes: dict,
    hf_validation: dict,
    hf_summary: str,
    run_id: str,
    tissue: str,
    species: str,
    source: str,
    hf_model_id: str,
):
    print("\n[5/5] Saving outputs...")

    sc.pl.umap(
        adata,
        color=["leiden", "cell_type_predicted"],
        legend_loc="on data",
        legend_fontsize=7,
        save=f"_{run_id}.png",
        show=False,
    )
    print(f"   UMAP   → figures/umap_{run_id}.png")

    h5ad_path = f"outputs/{run_id}_annotated.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"   AnnData → {h5ad_path}")

    report = {
        "run_id":               run_id,
        "date":                 datetime.date.today().isoformat(),
        "source":               source,
        "tissue":               tissue,
        "species":              species,
        "n_cells":              adata.n_obs,
        "n_genes":              adata.n_vars,
        "n_clusters":           adata.obs["leiden"].nunique(),
        "cell_types":           list(marker_genes.keys()),
        "marker_genes":         marker_genes,
        "hf_model":             hf_model_id,
        "hf_validation":        hf_validation,
    }
    report_path = f"outputs/{run_id}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Report  → {report_path}")

    txt_path = f"outputs/{run_id}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Run:     {run_id}\n")
        f.write(f"Source:  {source}\n")
        f.write(f"Tissue:  {tissue} | Species: {species}\n")
        f.write(f"Cells:   {adata.n_obs:,} | Clusters: {adata.obs['leiden'].nunique()}\n")
        f.write(f"Types:   {', '.join(marker_genes.keys())}\n\n")
        f.write("--- HuggingFace Biomedical Marker Validation ---\n")
        f.write(hf_summary + "\n")
    print(f"   Summary → {txt_path}")


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    use_synapse = bool(args.synapse)
    source      = args.synapse if use_synapse else args.input or "pbmc3k"
    hf_model_id = args.hf_model or HF_MODEL

    dataset_id = args.synapse if use_synapse else os.path.splitext(os.path.basename(source))[0]
    run_id     = f"{dataset_id}_{args.species}"

    print("=" * 60)
    print("  scRNA-seq Automated Analysis Journal")
    print(f"  Run:      {run_id}")
    print(f"  Source:   {source}")
    print(f"  Tissue:   {args.tissue} | Species: {args.species}")
    print(f"  HF model: {hf_model_id}")
    print("=" * 60)

    check_env(use_synapse)

    # ── Load data
    if use_synapse:
        adata = load_from_synapse(args.synapse, species=args.species)
    elif args.input:
        print(f"\n[1/5] Loading local file: {args.input}")
        adata = load_file(args.input)
    else:
        print("\n[1/5] No input specified — using built-in pbmc3k for testing")
        adata = sc.datasets.pbmc3k()
        adata.obs["sample"] = "pbmc3k"
        print(f"   Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    adata               = preprocess(adata, args.resolution)
    adata, markers      = annotate(adata, args.tissue, args.species)
    hf_val, hf_summary  = validate_with_hf(markers, hf_model_id)
    save_outputs(
        adata, markers,
        hf_val, hf_summary,
        run_id, args.tissue, args.species, source, hf_model_id,
    )

    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print(f"  outputs/{run_id}_annotated.h5ad")
    print(f"  outputs/{run_id}_report.json")
    print(f"  figures/umap_{run_id}.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
