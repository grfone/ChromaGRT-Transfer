"""Build the HILIC dataset for ChromaGRT from the same RepoRT revision as the RP data.

The RP pipeline (``prepare_reportrp.py``) downloads the raw RepoRT tables and
curates the reversed-phase subset of them. The downloader already writes the
HILIC subset next to the RP one, but no step consumed it, so the HILIC
compounds were discarded. This script curates that subset the same way and
then expresses it in the RP condition vocabulary, so that a ChromaGRT
checkpoint trained on RP can be loaded into a HILIC run unchanged.

The projection matters because ChromaGRT's condition vector is the table's
condition columns in order: the RP subset carries six USP column codes and
the HILIC subset carries four different ones, so the two datasets would
otherwise give condition vectors of different width and the checkpoint's
condition encoder could not be loaded. Projecting the HILIC dataset onto the
RP vocabulary gives both the same 114-dimensional condition vector.

Only the HILIC columns that the RP reference does not know are affected: the
six ``column.usp.code_*`` columns, which are zero for HILIC. Every other
condition block (eluent composition, static column geometry, Tanaka and the
gradient) is carried over unchanged.

The pipeline takes hours - the molecular-descriptor step alone queries
PubChem once per compound - so each stage reuses its output when that output
is already there instead of rebuilding it. A run that is interrupted can be
started again and will pick up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.RepoRT_data_processing.prepare_reportrp import (
    ARTICLE_REPO_IDS,
    ARTICLE_REPORTRT_REVISION,
)
from src.RepoRT_data_processing.RepoRT_get_raw_data import (
    merge_complete_file,
    repo_rt_processed_data_url,
)
from src.RepoRT_data_processing.RepoRT_preprocessing import preprocess_rp_dataset
from src.RepoRT_data_processing.RepoRT_processing import get_processed_df_from_raw
from src.benchmarks.shared_folds.project_condition_columns import (
    condition_columns,
    project_condition_columns,
)
from src.get_molecular_descriptors.get_molecular_descriptors import complete_moldesc_query
from src.training.RepoRT.tanaka_parameters import enrich_with_graphormer_rt_tanaka


HILIC_SUBSET = "RepoRT_HILIC"
N_FOLDS = 10


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hilic_dirs(data_root: Path) -> dict[str, Path]:
    hilic_root = data_root / HILIC_SUBSET
    return {
        "raw": hilic_root / "raw_data",
        "preprocessed": hilic_root / "preprocessed_data",
        "processed": hilic_root / "processed_data",
        "artifacts": hilic_root / "processed_data" / "build_artifacts",
    }


def _run(module: str, *args: str) -> None:
    """Run one of the repository's existing command line steps.

    The step is run from the repository root, because it is invoked as
    ``python -m src....`` and the ``src`` package is imported from there.
    """
    command = [sys.executable, "-m", module, *args]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=_project_root())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_project_root() / "data")
    parser.add_argument("--skip-download", action="store_true", help="Use an already downloaded raw RepoRT tree.")
    parser.add_argument("--skip-molecular-descriptors", action="store_true")
    parser.add_argument(
        "--condition-reference",
        type=Path,
        default=(
            _project_root()
            / "data"
            / "RepoRT_RP"
            / "processed_data"
            / "no_SMRT"
            / "complete_processed_data_with_tanaka.tsv"
        ),
        help=(
            "Curated RP table whose condition vocabulary the HILIC dataset is "
            "expressed in. Only its column names are used."
        ),
    )
    parser.add_argument(
        "--column-properties",
        type=Path,
        default=(
            _project_root()
            / "data"
            / "benchmarks"
            / "graphormer_rt"
            / "column_properties_from_upstream_sample.tsv"
        ),
        help="Tanaka/HSMB column parameter table, shared with the RP pipeline.",
    )
    parser.add_argument(
        "--fold-strategy",
        choices=["cc", "scaffold", "random_within_cc"],
        default="cc",
        help="Fold strategy for the HILIC cross-validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _curate_hilic_dataset(args, data_root: Path, dirs: dict[str, Path], projected_path: Path) -> None:
    """Download, preprocess and curate the HILIC subset, then project it."""
    # 1. Raw RepoRT tables. The downloader writes the RP and the HILIC subset
    #    from the same tables, so this is the same data the RP run used.
    if args.skip_download:
        for name in ("raw_rt_data.tsv", "raw_cc_data.tsv", "raw_grad_data.tsv"):
            if not (dirs["raw"] / name).is_file():
                raise FileNotFoundError(f"--skip-download requires {dirs['raw'] / name}")
    else:
        merge_complete_file(
            path2res=data_root / "RepoRT" / "raw_data",
            seed_url=repo_rt_processed_data_url(ARTICLE_REPORTRT_REVISION),
            repos_array=ARTICLE_REPO_IDS,
            split_output_root=data_root,
        )

    # 2. Preprocessing. The only RP-specific part of preprocess_rp_dataset is
    #    the SMRT NPLS filter, and SMRT is an RP dataset, so it is switched off.
    preprocess_rp_dataset(
        raw_root=data_root / HILIC_SUBSET,
        output_dir=dirs["preprocessed"],
        filter_smrt_by_npls=False,
    )

    # 3. Curation: the article's rules, minus the SMRT removal that only
    #    applies to the reversed-phase subset.
    get_processed_df_from_raw(
        source_path=dirs["artifacts"],
        drop_smrt=False,
        down_grad_filter=False,
        smrt_dir_id=None,
        preprocessed_dir=dirs["preprocessed"],
        output_dir=dirs["processed"],
        raw_data_dir=dirs["raw"],
    )

    # 4. Tanaka column parameters, the same enrichment the RP data gets.
    processed_path = dirs["processed"] / "complete_processed_data_with_tanaka.tsv"
    enrich_with_graphormer_rt_tanaka(
        processed_path=dirs["processed"] / "complete_processed_data.tsv",
        output_path=processed_path,
        raw_cc_path=dirs["raw"] / "raw_cc_data.tsv",
        cc_map_path=dirs["artifacts"] / "cc_id_vs_dir_id.tsv",
        column_properties_path=args.column_properties,
        report_dir=dirs["processed"] / "tanaka_reports",
    )

    # 5. Express the HILIC dataset in the RP condition vocabulary, so a
    #    checkpoint trained on RP loads into a HILIC run without touching the
    #    condition encoder.
    reference_columns = list(pd.read_csv(args.condition_reference, sep="\t", nrows=0).columns)
    hilic_df = pd.read_csv(processed_path, sep="\t")
    projected_df = project_condition_columns(hilic_df, pd.DataFrame(columns=reference_columns))
    projected_df.to_csv(projected_path, sep="\t", index=False)
    print(
        f"condition columns: HILIC {len(condition_columns(hilic_df))} -> "
        f"RP vocabulary {len(condition_columns(projected_df))}"
    )

    # 6. Molecular descriptors for the HILIC compounds.
    descriptor_path = data_root / "complete_moldesc_hilic.tsv"
    if args.skip_molecular_descriptors:
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                "--skip-molecular-descriptors requires an existing descriptor table: "
                + str(descriptor_path)
            )
    else:
        complete_moldesc_query(
            initial_file=data_root / "moldescs_hilic.tsv",
            path2res=descriptor_path,
            raw_report_file=dirs["raw"] / "raw_rt_data.tsv",
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    dirs = _hilic_dirs(data_root)
    projected_path = dirs["processed"] / "complete_processed_data_with_tanaka_rp_vocabulary.tsv"
    fold_dir = data_root / "benchmarks" / "article-v1" / "hilic" / "folds"
    asset_dir = data_root / "benchmarks" / "article-v1" / "hilic"

    # Curating the dataset is the expensive part, so it is skipped when its
    # output is already there.
    if projected_path.exists():
        print(f"Reusing the curated HILIC dataset at {projected_path}")
    else:
        _curate_hilic_dataset(args, data_root, dirs, projected_path)

    # 7. Cross-validation folds over the projected HILIC dataset.
    if (fold_dir / "fold_0.tsv").exists():
        print(f"Reusing the HILIC folds at {fold_dir}")
    else:
        _run(
            "src.benchmarks.shared_folds.generate_chromagrt_folds",
            "--input",
            str(projected_path),
            "--output-dir",
            str(fold_dir),
            "--n-splits",
            str(N_FOLDS),
            "--seed",
            str(args.seed),
            "--strategy",
            args.fold_strategy,
        )

    # 8. ChromaGRT assets: one train/validation/test split per fold.
    if (asset_dir / "chromagrt_splits" / "fold_0").exists():
        print(f"Reusing the HILIC assets at {asset_dir}")
    else:
        _run(
            "src.benchmarks.shared_folds.build_kfold_assets_from_chromagrt_folds",
            "--fold-dir",
            str(fold_dir),
            "--output-dir",
            str(asset_dir),
            "--preprocessed-cc",
            str(dirs["preprocessed"] / "preprocessed_cc_data.tsv"),
            "--preprocessed-grad",
            str(dirs["preprocessed"] / "preprocessed_gradient_data.tsv"),
            "--column-properties",
            str(args.column_properties),
        )

    manifest_path = data_root / HILIC_SUBSET / "reproduction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(
            {
                "repo_rt_revision": ARTICLE_REPORTRT_REVISION,
                "repo_rt_processed_data_url": repo_rt_processed_data_url(ARTICLE_REPORTRT_REVISION),
                "repo_ids": ARTICLE_REPO_IDS,
                "subset": HILIC_SUBSET,
                "fold_strategy": args.fold_strategy,
                "n_folds": N_FOLDS,
                "seed": args.seed,
                "condition_reference": str(args.condition_reference),
                "projected_dataset": str(projected_path),
                "molecular_descriptors": str(data_root / "complete_moldesc_hilic.tsv"),
                "assets": str(asset_dir),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"Pipeline provenance: {manifest_path}")


if __name__ == "__main__":
    main()
