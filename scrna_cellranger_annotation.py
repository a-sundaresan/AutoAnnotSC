"""
scrna_journal.py — scRNA-seq Automated Analysis Journal
---------------------------------------------------------
Accepts:
  - Synapse ID  (downloads automatically via Synapse MCP)
  - Local .h5ad file
  - Local .h5   file (10x CellRanger format)

Usage inside Claude Code:
  conda activate cellannotator
  claude
  → Run scrna_journal.py

Or override defaults from command line:
  python scrna_journal.py --input data/myfile.h5ad --tissue lung
  python scrna_journal.py --synapse syn26522055 --tissue breast
"""

import os, sys, json, argparse, datetime
import scanpy as sc
import requests
import synapseclient
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
SYNAPSE_TOKEN = os.getenv("SYNAPSE_AUTH_TOKEN")

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-sonnet-4-20250514"
HEADERS = {
    "Content-Type":      "application/json",
    "anthropic-version": "2023-06-01",
    "anthropic-beta":    "mcp-client-2025-04-04",
}

os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)


# ── ARGUMENT PARSING ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="scRNA-seq Automated Analysis Journal")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--synapse", type=str, help="Synapse ID e.g. syn26522055")
    src.add_argument("--input",   type=str, help="Path to local .h5ad or .h5 file")
    p.add_argument("--tissue",   type=str, default="blood",  help="Tissue type (default: blood)")
    p.add_argument("--species",  type=str, default="human",  help="Species (default: human)")
    p.add_argument("--resolution", type=float, default=0.5,  help="Leiden resolution (default: 0.5)")
    return p.parse_args()


# ── ENV CHECK ────────────────────────────────────────────────────────

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


# ── CLAUDE + MCP HELPER ─────────────────────────────────────────────

def call_claude(prompt: str, mcps: list[str] = None, max_tokens: int = 2000) -> str:
    MCP_URLS = {
        "synapse": {"type": "url", "url": "https://mcp.synapse.org/mcp",      "name": "synapse", "authorization_token": SYNAPSE_TOKEN},
        "pubmed":  {"type": "url", "url": "https://pubmed.mcp.claude.com/mcp","name": "pubmed"},
    }
    body = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if mcps:
        body["mcp_servers"] = [MCP_URLS[m] for m in mcps]

    resp = requests.post(
        API_URL,
        headers={**HEADERS, "x-api-key": ANTHROPIC_KEY},
        json=body
    )
    if not resp.ok:
        print(f"   API error {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    data  = resp.json()
    texts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(texts)


# ── STEP 1: LOAD DATA ────────────────────────────────────────────────

MTX_FILES = {"barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz",
             "barcodes.tsv",    "features.tsv",    "matrix.mtx"}


def _find_mtx_folders(syn, parent_id, results=None):
    """Recursively find all folders containing a 10x MTX trio."""
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
    """Download from Synapse using synapseclient and load."""
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

        # Prefer folders whose parent path contains the species name
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
                continue  # skip BAM and other large non-MTX files
            syn.get(child["id"], downloadLocation=dest, ifcollision="overwrite.local")
        return load_file(dest)
    else:
        dl = syn.get(synapse_id, downloadLocation="data", ifcollision="overwrite.local")
        path = dl["path"] if "path" in dl else os.path.join("data", dl["name"])
        print(f"   Downloaded: {path}")
        return load_file(path)


def load_file(path: str) -> sc.AnnData:
    """Load .h5ad, .h5, or 10x MTX directory automatically."""
    if os.path.isdir(path):
        print(f"   Format: 10x MTX folder — {path}")
        adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=True)
        adata.var_names_make_unique()
        adata.obs["sample"] = os.path.basename(path)
        print(f"   Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        return adata

    ext = os.path.splitext(path)[-1].lower()

    if ext == ".h5ad":
        print(f"   Format: AnnData (.h5ad)")
        adata = sc.read_h5ad(path)

    elif ext in (".h5", ".hdf5"):
        print(f"   Format: 10x CellRanger (.h5) — converting to AnnData")
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()

    elif ext in (".mtx", ".gz"):
        # folder with barcodes/features/matrix
        folder = os.path.dirname(path)
        print(f"   Format: 10x MTX folder — {folder}")
        adata = sc.read_10x_mtx(folder, var_names="gene_symbols", cache=True)
        adata.var_names_make_unique()

    else:
ращ        raise ValueError(
            f"Unsupported file format: {ext}\n"
            f"Supported: .h5ad, .h5, .hdf5, .mtx"
        )

    adata.obs["sample"] = os.path.basename(path)
    print(f"   Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata


# ── STEP 2: QC & PREPROCESS ─────────────────────────────────────────

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


# ── STEP 3: ANNOTATE ─────────────────────────────────────────────────

def annotate(adata: sc.AnnData, tissue: str, species: str) -> tuple[sc.AnnData, dict]:
    print("\n[3/5] Annotating cell types with CellAnnotator + Claude...")
    from cell_annotator import CellAnnotator

    cell_ann = CellAnnotator(
        adata,
        species=species,
        tissue=tissue,
        cluster_key="leiden",
        sample_key=None,
        provider="anthropic",
        model="claude-sonnet-4-6",
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


# ── STEP 4: PUBMED VALIDATION ────────────────────────────────────────

def validate_with_pubmed(marker_genes: dict, tissue: str, species: str) -> str:
    print("\n[4/5] Validating marker genes via PubMed MCP...")
    top = {k: v[:3] for k, v in list(marker_genes.items())[:6]}

    result = call_claude(
        f"""I identified these marker genes for cell types in a {species}
        {tissue} scRNA-seq dataset:
        {json.dumps(top, indent=2)}

        Search PubMed to verify these are well-established markers for
        each cell type. Flag any unexpected or novel markers.
        Return a concise validation summary (3-5 sentences).""",
        mcps=["pubmed"]
    )
    print(f"   {result[:200]}...")
    return result


# ── STEP 5: SAVE OUTPUTS ─────────────────────────────────────────────

def save_outputs(adata: sc.AnnData, marker_genes: dict,
                 pubmed_val: str, run_id: str,
                 tissue: str, species: str, source: str):
    print("\n[5/5] Saving outputs...")

    sc.pl.umap(
        adata,
        color=["leiden", "cell_type_predicted"],
        legend_loc="on data",
        legend_fontsize=7,
        save=f"_{run_id}.png",
        show=False
    )
    print(f"   UMAP   → figures/umap_{run_id}.png")

    h5ad_path = f"outputs/{run_id}_annotated.h5ad"
    adata.write_h5ad(h5ad_path)
    print(f"   AnnData → {h5ad_path}")

    report = {
        "run_id":       run_id,
        "date":         datetime.date.today().isoformat(),
        "source":       source,
        "tissue":       tissue,
        "species":      species,
        "n_cells":      adata.n_obs,
        "n_genes":      adata.n_vars,
        "n_clusters":   adata.obs["leiden"].nunique(),
        "cell_types":   list(marker_genes.keys()),
        "marker_genes": marker_genes,
        "pubmed_notes": pubmed_val,
    }
    report_path = f"outputs/{run_id}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   Report  → {report_path}")

    txt_path = f"outputs/{run_id}_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"Run:      {run_id}\n")
        f.write(f"Source:   {source}\n")
        f.write(f"Tissue:   {tissue} | Species: {species}\n")
        f.write(f"Cells:    {adata.n_obs:,} | Clusters: {adata.obs['leiden'].nunique()}\n")
        f.write(f"Types:    {', '.join(marker_genes.keys())}\n\n")
        f.write("--- PubMed Validation ---\n")
        f.write(pubmed_val + "\n")
    print(f"   Summary → {txt_path}")


# ── MAIN ─────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    use_synapse = bool(args.synapse)
    source      = args.synapse if use_synapse else args.input or "pbmc3k"

    dataset_id = args.synapse if use_synapse else os.path.splitext(os.path.basename(source))[0]
    run_id     = f"{dataset_id}_{args.species}"

    print("=" * 55)
    print("  scRNA-seq Automated Analysis Journal")
    print(f"  Run:     {run_id}")
    print(f"  Source:  {source}")
    print(f"  Tissue:  {args.tissue} | Species: {args.species}")
    print("=" * 55)

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

    adata          = preprocess(adata, args.resolution)
    adata, markers = annotate(adata, args.tissue, args.species)
    pubmed_val     = validate_with_pubmed(markers, args.tissue, args.species)
    save_outputs(adata, markers, pubmed_val, run_id, args.tissue, args.species, source)

    print("\n" + "=" * 55)
    print("  Analysis complete!")
    print(f"  outputs/{run_id}_annotated.h5ad")
    print(f"  outputs/{run_id}_report.json")
    print(f"  figures/umap_{run_id}.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
