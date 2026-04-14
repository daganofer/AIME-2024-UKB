"""
Extract disease onset and GP scripts data using Spark on DNAnexus RAP.

Run with:
    spark-submit extract_data_spark.py

Or in a Spark-enabled JupyterLab terminal:
    python extract_data_spark.py
"""
import os
import sys
from pathlib import Path

import dxdata
import pyspark
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ID = "record-J62ZZk0Jf4jV9p23B5P3YBBG"
REPO_DIR = "/opt/notebooks/AIME-2024-UKB"
CACHE_DIR = os.path.join(REPO_DIR, "data", "raw")

# Saved field lists from previous session
PROJECT_CACHE_FILES = {
    "fo_fields_with_eid.txt": "cache/fo_fields_with_eid.txt",
    "required_fields.txt": "cache/required_fields.txt",
    "available_fields.txt": "cache/available_fields.txt",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_project_cache():
    """Download field list files saved to project cache/ in previous session."""
    print("\n=== Downloading cached files from project ===")
    os.makedirs("/tmp", exist_ok=True)
    for local_name, remote_path in PROJECT_CACHE_FILES.items():
        local_path = f"/tmp/{local_name}"
        if os.path.exists(local_path):
            print(f"  {local_name}: already exists locally")
            continue
        try:
            os.system(f'dx download "{remote_path}" -o "{local_path}" -f')
            if os.path.exists(local_path):
                print(f"  {local_name}: downloaded")
            else:
                print(f"  {local_name}: download failed (file may not exist in project)")
        except Exception as e:
            print(f"  {local_name}: error - {e}")


def init_spark():
    """Initialize Spark context and session."""
    print("\n=== Initializing Spark ===")
    sc = pyspark.SparkContext.getOrCreate()
    spark = pyspark.sql.SparkSession(sc)
    print(f"  Spark version: {spark.version}")
    return spark


def get_entity(dataset, name):
    """Look up an entity by name from the dataset's entity list."""
    for e in dataset.entities:
        if e.name == name:
            return e
    raise KeyError(f"Entity '{name}' not found. Available: {[e.name for e in dataset.entities]}")


def load_dataset():
    """Load the UKB dataset via dxdata."""
    print(f"\n=== Loading dataset {DATASET_ID} ===")
    dataset = dxdata.load_dataset(id=DATASET_ID)
    print(f"  Dataset loaded")
    print(f"  Entities: {[e.name for e in dataset.entities]}")
    return dataset


def extract_gp_scripts(dataset, spark):
    """Extract GP scripts entity table with all available columns."""
    print("\n=== Extracting GP scripts ===")

    gp = get_entity(dataset, "gp_scripts")
    fields = [f.name for f in gp.fields]
    print(f"  Available columns: {fields}")
    has_quantity = "quantity" in fields
    print(f"  Has 'quantity' column: {has_quantity}")

    # Load via Spark — retrieve all fields
    all_gp_fields = list(gp.fields)
    gp_df = gp.retrieve_fields(fields=all_gp_fields, engine=spark, coding_values="replace")
    row_count = gp_df.count()
    print(f"  Total rows: {row_count}")

    # Convert to pandas and save
    output_path = os.path.join(CACHE_DIR, "drug_export", "data", "gp_scripts_spark.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Also save as TSV for PRESNER compatibility
    tsv_path = os.path.join(CACHE_DIR, "drug_export", "data", "gp_scripts.txt")

    print(f"  Converting to pandas...")
    gp_pandas = gp_df.toPandas()
    print(f"  Rows in pandas: {len(gp_pandas)}")
    print(f"  Columns: {list(gp_pandas.columns)}")
    print(f"  Sample:\n{gp_pandas.head(3)}")

    # Save parquet
    gp_pandas.to_parquet(output_path, index=False)
    print(f"  Saved parquet: {output_path}")

    # Save TSV (for PRESNER)
    # Remove placeholder symlink if it exists
    if os.path.islink(tsv_path):
        os.unlink(tsv_path)
    gp_pandas.to_csv(tsv_path, sep="\t", index=False)
    print(f"  Saved TSV: {tsv_path}")

    return gp_pandas, has_quantity


def extract_disease_onset(dataset, spark):
    """Extract First Occurrence fields (p131XXX) as disease_onset.csv."""
    print("\n=== Extracting disease onset (First Occurrence fields) ===")

    # Find all p131XXX fields
    participant = get_entity(dataset, "participant")
    all_fields = [f.name for f in participant.fields]
    fo_fields = sorted([f for f in all_fields if f.startswith("p131")])
    print(f"  Found {len(fo_fields)} First Occurrence fields")

    if len(fo_fields) == 0:
        print("  ERROR: No First Occurrence fields found!")
        return None

    # Extract in batches (Spark can handle this but let's be safe)
    BATCH_SIZE = 200
    batches = [fo_fields[i:i + BATCH_SIZE] for i in range(0, len(fo_fields), BATCH_SIZE)]
    print(f"  Extracting in {len(batches)} batches of ~{BATCH_SIZE} fields")

    eid_field = participant.find_field(name="eid")
    result_df = None
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)}: {len(batch)} fields ({batch[0]}..{batch[-1]})")
        field_objects = [eid_field] + [participant.find_field(name=f) for f in batch]
        batch_df = participant.retrieve_fields(
            fields=field_objects,
            engine=spark,
            coding_values="raw",
        )
        if result_df is None:
            result_df = batch_df
        else:
            result_df = result_df.join(batch_df.drop("eid"), on="eid", how="outer")

    print(f"  Converting to pandas...")
    onset_pandas = result_df.toPandas()
    print(f"  Shape: {onset_pandas.shape}")
    print(f"  Sample columns: {list(onset_pandas.columns[:5])}")

    # Save
    output_path = os.path.join(CACHE_DIR, "disease_onset.csv")
    # Remove placeholder symlink if it exists
    if os.path.islink(output_path):
        os.unlink(output_path)
    onset_pandas.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return onset_pandas


def extract_participant_data(dataset, spark):
    """Extract main participant fields using the repo's field.txt list."""
    print("\n=== Extracting participant data (361 fields) ===")

    field_list_path = os.path.join(REPO_DIR, "data", "raw", "ukb_export", "field.txt")
    if not os.path.exists(field_list_path):
        print(f"  ERROR: {field_list_path} not found!")
        return None

    with open(field_list_path) as f:
        field_ids = [line.strip() for line in f if line.strip()]
    print(f"  Fields to extract: {len(field_ids)}")

    participant = get_entity(dataset, "participant")
    all_field_names = [f.name for f in participant.fields]

    # Map field IDs to dxdata field names (p<id>_i<instance>)
    # We need all instances of each field
    fields_to_get = ["eid"]
    for fid in field_ids:
        matching = [fn for fn in all_field_names if fn.startswith(f"p{fid}_") or fn == f"p{fid}"]
        if not matching:
            # Try 22189 for 189
            if fid == "189":
                matching = [fn for fn in all_field_names if fn.startswith("p22189_") or fn == "p22189"]
                if matching:
                    print(f"  Field 189 -> 22189: found {len(matching)} instances")
            if not matching:
                print(f"  WARNING: Field {fid} not found in dataset")
                continue
        fields_to_get.extend(matching)

    print(f"  Total field instances to retrieve: {len(fields_to_get)}")

    # Extract in batches
    BATCH_SIZE = 100
    # fields_to_get[0] is "eid", rest are field names
    field_name_batches = [fields_to_get[i:i + BATCH_SIZE] for i in range(1, len(fields_to_get), BATCH_SIZE)]
    print(f"  Extracting in {len(field_name_batches)} batches")

    eid_field = participant.find_field(name="eid")
    result_df = None
    for i, batch in enumerate(field_name_batches):
        print(f"  Batch {i+1}/{len(field_name_batches)}: {len(batch)} fields")
        try:
            field_objects = [eid_field] + [participant.find_field(name=fn) for fn in batch]
            batch_df = participant.retrieve_fields(
                fields=field_objects,
                engine=spark,
                coding_values="raw",
            )
            if result_df is None:
                result_df = batch_df
            else:
                result_df = result_df.join(batch_df.drop("eid"), on="eid", how="outer")
        except Exception as e:
            print(f"  ERROR in batch {i+1}: {e}")
            continue

    print(f"  Converting to pandas...")
    participant_pandas = result_df.toPandas()
    print(f"  Shape: {participant_pandas.shape}")

    # Save - the notebook expects data/raw/ukb_export/data.csv
    output_path = os.path.join(CACHE_DIR, "ukb_export", "data.csv")
    participant_pandas.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    return participant_pandas


def main():
    print("=" * 60)
    print("MQTT Paper Reproduction — Data Extraction via Spark")
    print("=" * 60)

    # Step 0: Download cached files from project
    download_project_cache()

    # Step 1: Init
    spark = init_spark()
    dataset = load_dataset()

    # Step 2: GP scripts (check for quantity column)
    gp_df, has_quantity = extract_gp_scripts(dataset, spark)

    # Step 3: Disease onset
    onset_df = extract_disease_onset(dataset, spark)

    # Step 4: Main participant data
    participant_df = extract_participant_data(dataset, spark)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  GP scripts:      {len(gp_df)} rows, quantity={'YES' if has_quantity else 'NO'}")
    if onset_df is not None:
        print(f"  Disease onset:   {onset_df.shape}")
    if participant_df is not None:
        print(f"  Participant:     {participant_df.shape}")
    print(f"\nFiles saved to: {CACHE_DIR}")

    if not has_quantity:
        print("\n*** WARNING: GP scripts missing 'quantity' column. ***")
        print("    PRESNER pipeline and extract_ukb_data.ipynb will need adaptation.")
        print("    See RUNBOOK.md for details.")


if __name__ == "__main__":
    main()
