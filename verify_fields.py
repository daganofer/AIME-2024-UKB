"""
Verify that all required UKB fields are available in your dataset/application.

Run on RAP (DNAnexus JupyterLab) with:
    python verify_fields.py

Or in a notebook cell:
    %run verify_fields.py
"""
import subprocess
import sys
from pathlib import Path

# ── All 361 fields required by the paper ─────────────────────────────────────

BLOOD_BIOCHEM = [
    30620, 30600, 30610, 30630, 30640, 30650, 30710, 30680, 30690, 30700,
    30720, 30660, 30730, 30740, 30750, 30760, 30770, 30780, 30790, 30800,
    30810, 30820, 30830, 30850, 30840, 30860, 30870, 30880, 30670, 30890,
]

BLOOD_COUNT = [
    30160, 30220, 30150, 30210, 30030, 30020, 30300, 30290, 30280, 30120,
    30180, 30050, 30060, 30040, 30100, 30260, 30270, 30130, 30190, 30140,
    30200, 30170, 30230, 30080, 30090, 30110, 30010, 30070, 30250, 30240,
    30000,
]

NMR = [
    23474, 23475, 23476, 23477, 23460, 23479, 23440, 23439, 23441, 23433,
    23432, 23431, 23484, 23526, 23561, 23533, 23498, 23568, 23540, 23505,
    23575, 23547, 23512, 23554, 23491, 23519, 23580, 23610, 23635, 23615,
    23590, 23640, 23620, 23595, 23645, 23625, 23600, 23630, 23585, 23605,
    23485, 23418, 23527, 23417, 23562, 23534, 23499, 23569, 23541, 23506,
    23576, 23548, 23513, 23416, 23555, 23492, 23520, 23581, 23611, 23636,
    23616, 23591, 23641, 23621, 23596, 23646, 23626, 23601, 23631, 23586,
    23606, 23473, 23404, 23481, 23430, 23523, 23429, 23558, 23530, 23495,
    23565, 23537, 23502, 23572, 23544, 23509, 23428, 23551, 23488, 23516,
    23478, 23443, 23450, 23457, 23486, 23422, 23528, 23421, 23563, 23535,
    23500, 23570, 23542, 23507, 23577, 23549, 23514, 23420, 23556, 23493,
    23521, 23582, 23612, 23637, 23617, 23592, 23642, 23622, 23597, 23647,
    23627, 23602, 23632, 23587, 23607, 23470, 20280, 23461, 23462, 23480,
    23406, 23463, 23465, 23405, 23471, 23466, 23449, 23456, 23447, 23454,
    23444, 23451, 23445, 23459, 23452, 23468, 23437, 23434, 23483, 23414,
    23525, 23413, 23560, 23532, 23497, 23567, 23539, 23504, 23574, 23546,
    23511, 23412, 23553, 23490, 23518, 23579, 23609, 23634, 23614, 23589,
    23639, 23619, 23594, 23644, 23624, 23599, 23629, 23584, 23604, 23446,
    23458, 23453, 23472, 23402, 23448, 23455, 20281, 23438, 23400, 23401,
    23436, 23464, 23427, 23415, 23442, 23419, 23482, 23426, 23524, 23425,
    23559, 23531, 23496, 23423, 23566, 23538, 23503, 23573, 23545, 23510,
    23424, 23552, 23489, 23517, 23411, 23407, 23487, 23410, 23529, 23409,
    23564, 23536, 23501, 23571, 23543, 23508, 23578, 23550, 23515, 23408,
    23557, 23494, 23522, 23435, 23583, 23613, 23638, 23618, 23593, 23643,
    23623, 23598, 23648, 23628, 23603, 23633, 23588, 23608, 23469, 23403,
    23467,
]

URINE = [30510, 30500, 30520, 30530]

TIME_INVARIANT = [31, 34, 52, 20022, 21000, 6138]

TIME_VARIANT = [
    189, 709, 816, 826, 6142, 6164, 1160, 1200, 1210, 1220, 1239, 1249,
    1498, 1558, 1050, 1060, 2139, 2296, 2306, 399, 4079, 4080, 102, 46,
    47, 48, 49, 50, 21001, 21002, 23099, 23100, 23101, 23102, 23104,
    2724, 2734, 3829, 21003,
]

# Additional data sources (not in field.txt but required)
EXTRA_FIELDS = {
    1062: "GP prescriptions (gp_scripts.txt)",
    1712: "First Occurrence fields (disease_onset.csv)",
}

ALL_FIELDS = {
    "blood_biochem": BLOOD_BIOCHEM,
    "blood_count": BLOOD_COUNT,
    "nmr": NMR,
    "urine": URINE,
    "time_invariant": TIME_INVARIANT,
    "time_variant": TIME_VARIANT,
}


def check_via_data_csv(csv_path: str):
    """Check which fields are present in an already-extracted data.csv."""
    import pandas as pd

    print(f"\nChecking extracted CSV: {csv_path}")
    # Only read header row
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns)

    # Extract field IDs from column names (format: fieldid-instance.array)
    present_fields = set()
    for col in cols:
        if col == "eid":
            continue
        try:
            field_id = int(col.split("-")[0])
            present_fields.add(field_id)
        except ValueError:
            pass

    print(f"  Total fields in CSV: {len(present_fields)}")
    return present_fields


def check_via_dxdata():
    """Check field availability via DNAnexus dxdata (works on RAP)."""
    try:
        import dxdata
    except ImportError:
        print("  dxdata not available (not on RAP?), skipping")
        return None

    print("\nChecking via dxdata...")
    try:
        dataset = dxdata.load_dataset(name="app*")
        available = set()
        for field in dataset.fields:
            try:
                available.add(int(field.name.split("-")[0]))
            except (ValueError, AttributeError):
                pass
        print(f"  Total fields available in dataset: {len(available)}")
        return available
    except Exception as e:
        print(f"  dxdata check failed: {e}")
        return None


def check_gp_scripts():
    """Check GP prescription data availability."""
    paths = [
        "data/raw/drug_export/data/gp_scripts.txt",
        "/opt/notebooks/data/raw/drug_export/data/gp_scripts.txt",
    ]
    for p in paths:
        if Path(p).exists():
            import pandas as pd
            df = pd.read_csv(p, sep="\t", nrows=5)
            print(f"\n  GP scripts found at {p}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample rows: {len(df)}")
            return True
    print("\n  GP scripts NOT FOUND. Download from UKB Data Portal (Field 1062)")
    return False


def check_disease_onset():
    """Check disease onset data availability."""
    paths = [
        "data/raw/disease_onset.csv",
        "/opt/notebooks/data/raw/disease_onset.csv",
    ]
    for p in paths:
        if Path(p).exists():
            import pandas as pd
            df = pd.read_csv(p, nrows=5)
            print(f"\n  Disease onset found at {p}")
            print(f"  Columns: {len(df.columns)} fields")
            return True
    print("\n  disease_onset.csv NOT FOUND. Extract First Occurrence data (Field 1712)")
    return False


def main():
    print("=" * 60)
    print("MQTT Paper Reproduction — Field Availability Check")
    print("=" * 60)

    # Try to find available fields
    available = None

    # Method 1: Check existing CSV
    csv_candidates = [
        "data/raw/ukb_export/data.csv",
        "/opt/notebooks/data/raw/ukb_export/data.csv",
    ]
    for csv_path in csv_candidates:
        if Path(csv_path).exists():
            available = check_via_data_csv(csv_path)
            break

    # Method 2: dxdata
    if available is None:
        available = check_via_dxdata()

    # Report
    if available is not None:
        all_required = []
        for group, fields in ALL_FIELDS.items():
            all_required.extend(fields)

        all_required_set = set(all_required)
        present = all_required_set & available
        missing = all_required_set - available

        print(f"\n{'=' * 60}")
        print(f"Required fields: {len(all_required_set)}")
        print(f"Present:         {len(present)}")
        print(f"Missing:         {len(missing)}")

        if missing:
            print(f"\nMISSING FIELDS:")
            for group, fields in ALL_FIELDS.items():
                group_missing = set(fields) - available
                if group_missing:
                    print(f"  {group}: {sorted(group_missing)}")
        else:
            print("\n  ALL main fields present!")

        # Check per-group completeness
        print(f"\nPer-group summary:")
        for group, fields in ALL_FIELDS.items():
            n_present = len(set(fields) & available)
            n_total = len(fields)
            status = "OK" if n_present == n_total else f"MISSING {n_total - n_present}"
            print(f"  {group:20s}: {n_present}/{n_total}  {status}")
    else:
        print("\nCould not auto-detect available fields.")
        print("Upload this script to RAP and run there, or check manually.")

    # Check supplementary data
    print(f"\n{'=' * 60}")
    print("Supplementary data sources:")
    check_gp_scripts()
    check_disease_onset()

    # Check helper files
    print(f"\n{'=' * 60}")
    print("Helper files:")
    helpers = [
        "data/raw/helper_files/Codings.tsv",
        "data/raw/helper_files/Data_Dictionary_Showcase.tsv",
        "data/raw/helper_files/ehierstring.txt",
    ]
    for h in helpers:
        status = "FOUND" if Path(h).exists() else "MISSING"
        print(f"  {h}: {status}")

    print(f"\n{'=' * 60}")
    print("Done. Fix any MISSING items before proceeding.")


if __name__ == "__main__":
    main()
