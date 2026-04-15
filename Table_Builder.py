import pandas as pd

# Build analysis table by merging baseline table with ADNI source tables.
# This version does NOT create TTAMY or AMY_EVENT.
# It merges the requested tables and creates a VISIT_DATE column using only
# VISITDATE fields from non-amyloid source tables.

BASE_TABLE_FILE = "My_Table_13Apr2026.csv"
AMYLOID_TABLE_FILE = "UCBERKELEY_AMY_6MM_13Apr2026.csv"
OUTPUT_FILE = "Merged_My_Table_UCBERKELEY_AMY_13Apr2026.csv"

ADDITIONAL_MERGES = [
    {
        "file_name": "ECOGSP_13Apr2026.csv",
        "columns_to_keep": [
            "MEMORY1", "MEMORY2", "MEMORY3", "MEMORY4", "MEMORY5", "MEMORY6", "MEMORY7", "MEMORY8",
            "LANG1", "LANG2", "LANG3", "LANG4", "LANG5", "LANG6", "LANG7", "LANG8", "LANG9",
            "VISSPAT1", "VISSPAT2", "VISSPAT3", "VISSPAT4", "VISSPAT5", "VISSPAT6", "VISSPAT7", "VISSPAT8",
            "PLAN1", "PLAN2", "PLAN3", "PLAN4", "PLAN5",
            "ORGAN1", "ORGAN2", "ORGAN3", "ORGAN4", "ORGAN5", "ORGAN6",
            "DIVATT1", "DIVATT2", "DIVATT3", "DIVATT4", "SOURCE",
            "EcogSPMem", "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "EcogSPTotal",
        ],
    },
    {
        "file_name": "FAMHXPAR_13Apr2026.csv",
        "columns_to_keep": [
            "MOTHALIVE", "MOTHAGE", "MOTHDEM", "MOTHAD", "MOTHSXAGE",
            "FATHALIVE", "FATHAGE", "FATHDEM", "FATHAD", "FATHSXAGE",
        ],
    },
    {
        "file_name": "FAMHXSIB_13Apr2026.csv",
        "columns_to_keep": [
            "SIBYOB", "SIBRELAT", "SIBGENDER", "SIBALIVE", "SIBAGE", "SIBDEMENT", "SIBAD", "SIBSXAGE",
        ],
    },
    {
        "file_name": "GDSCALE_13Apr2026.csv",
        "columns_to_keep": [
            "GDUNABL", "GDSATIS", "GDDROP", "GDEMPTY", "GDBORED", "GDSPIRIT", "GDAFRAID", "GDHAPPY",
            "GDHELP", "GDHOME", "GDMEMORY", "GDALIVE", "GDWORTH", "GDENERGY", "GDHOPE", "GDBETTER", "GDTOTAL",
        ],
    },
    {
        "file_name": "NEUROBAT_13Apr2026.csv",
        "columns_to_keep": [
            "CLOCKCIRC", "CLOCKSYM", "CLOCKNUM", "CLOCKHAND", "CLOCKTIME", "CLOCKSCOR",
            "COPYCIRC", "COPYSYM", "COPYNUM", "COPYHAND", "COPYTIME", "COPYSCOR",
            "LMSTORY", "LIMMTOTAL", "LIMMEND",
            "AVTOT1", "AVERR1", "AVTOT2", "AVERR2", "AVTOT3", "AVERR3", "AVTOT4", "AVERR4",
            "AVTOT5", "AVERR5", "AVTOT6", "AVERR6", "AVTOTB", "AVERRB", "AVENDED",
            "DSPANFOR", "DSPANFLTH", "DSPANBAC", "DSPANBLTH",
            "CATANIMSC", "CATANPERS", "CATANINTR", "CATVEGESC", "CATVGPERS", "CATVGINTR",
            "TRAASCOR", "TRAAERRCOM", "TRAAERROM", "TRABSCOR", "TRABERRCOM", "TRABERROM",
            "DIGITSCOR", "LDELBEGIN", "LDELTOTAL", "LDELCUE",
            "BNTND", "BNTSPONT", "BNTSTIM", "BNTCSTIM", "BNTPHON", "BNTCPHON", "BNTTOTAL",
            "AVDELBEGAN", "AVDEL30MIN", "AVDELERR1", "AVDELTOT", "AVDELERR2",
            "ANARTERR", "ANARTND", "ANART", "MINTSEMCUE", "MINTTOTAL", "MINTUNCUED",
        ],
    },
    {
        "file_name": "PTDEMOG_13Apr2026.csv",
        "columns_to_keep": [
            "PTGENDER", "PTDOB", "PTDOBYY", "PTHAND", "PTMARRY", "PTEDUCAT", "PTWORKHS", "PTWORK", "PTNOTRT",
            "PTRTYR", "PTHOME", "PTTLANG", "PTPLANG", "PTADBEG", "PTCOGBEG", "PTADDX", "PTETHCAT", "PTRACCAT",
            "PTIDENT", "PTORIENT", "PTORIENTOT", "PTENGSPK", "PTNLANG", "PTENGSPKAGE", "PTCLANG", "PTLANGSP",
            "PTLANGWR", "PTSPTIM", "PTSPOTTIM", "PTLANGPR1", "PTLANGSP1", "PTLANGRD1", "PTLANGWR1", "PTLANGUN1",
            "PTLANGPR2", "PTLANGSP2", "PTLANGRD2", "PTLANGWR2", "PTLANGUN2", "PTLANGPR3", "PTLANGSP3", "PTLANGRD3",
            "PTLANGWR3", "PTLANGUN3", "PTLANGPR4", "PTLANGSP4", "PTLANGRD4", "PTLANGWR4", "PTLANGUN4", "PTLANGPR5",
            "PTLANGSP5", "PTLANGRD5", "PTLANGWR5", "PTLANGUN5", "PTLANGPR6", "PTLANGSP6", "PTLANGRD6", "PTLANGWR6",
            "PTLANGUN6", "PTLANGTTL", "PTETHCATH", "PTASIAN", "PTOPI", "PTBORN", "PTBIRPL", "PTIMMAGE", "PTIMMWHY",
            "PTBIRPR", "PTBIRGR",
        ],
    },
]

AMYLOID_COLUMNS_TO_KEEP = [
    "PTID",
    "VISCODE2",
    "AMYLOID_STATUS",
    "AMYLOID_STATUS_COMPOSITE_REF",
    "SCANDATE",
    "PROCESSDATE",
]


def normalize_visit_code(value: object) -> str:
    if pd.isna(value):
        return ""

    normalized = str(value).strip().lower()
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "")
    normalized = normalized.replace("_", "")

    visit_map = {
        "screening": "sc",
        "screen": "sc",
        "sc": "sc",
        "bl": "bl",
        "baseline": "bl",
        "init": "bl",
        "m03": "m03",
        "month3": "m03",
        "month03": "m03",
        "3m": "m03",
        "m06": "m06",
        "month6": "m06",
        "month06": "m06",
        "6m": "m06",
        "m12": "m12",
        "month12": "m12",
        "12m": "m12",
        "m18": "m18",
        "month18": "m18",
        "18m": "m18",
        "m24": "m24",
        "month24": "m24",
        "24m": "m24",
        "m30": "m30",
        "month30": "m30",
        "30m": "m30",
        "m36": "m36",
        "month36": "m36",
        "36m": "m36",
        "m42": "m42",
        "month42": "m42",
        "42m": "m42",
        "m48": "m48",
        "month48": "m48",
        "48m": "m48",
        "m54": "m54",
        "month54": "m54",
        "54m": "m54",
        "m60": "m60",
        "month60": "m60",
        "60m": "m60",
        "m72": "m72",
        "month72": "m72",
        "72m": "m72",
        "m84": "m84",
        "month84": "m84",
        "84m": "m84",
        "m96": "m96",
        "month96": "m96",
        "96m": "m96",
    }

    if normalized in visit_map:
        return visit_map[normalized]

    if normalized.startswith("m") and normalized[1:].isdigit():
        return f"m{int(normalized[1:]):02d}"

    if normalized.isdigit():
        return f"m{int(normalized):02d}"

    return normalized


def print_merge_diagnostics(
    left_table: pd.DataFrame,
    right_table: pd.DataFrame,
    left_id_col: str,
    left_visit_col: str,
    right_id_col: str,
    right_visit_col: str,
    label: str,
) -> None:
    left_keys = left_table[[left_id_col, left_visit_col]].drop_duplicates().copy()
    right_keys = right_table[[right_id_col, right_visit_col]].drop_duplicates().copy()

    left_keys.columns = ["id", "visit"]
    right_keys.columns = ["id", "visit"]

    matched_keys = left_keys.merge(right_keys, how="inner", on=["id", "visit"])

    print(f"\n{label} merge diagnostics:")
    print(f"  Left unique subject-visit keys:  {len(left_keys)}")
    print(f"  Right unique subject-visit keys: {len(right_keys)}")
    print(f"  Matched subject-visit keys:      {len(matched_keys)}")


def print_visitdate_diagnostics(merged_table: pd.DataFrame) -> None:
    visitdate_columns = [column for column in merged_table.columns if column.startswith("VISITDATE_")]

    print("\nVISITDATE contribution diagnostics:")
    if not visitdate_columns:
        print("  No VISITDATE_* columns were merged into the table.")
        return

    for column in sorted(visitdate_columns):
        non_missing = int(pd.to_datetime(merged_table[column], errors="coerce").notna().sum())
        print(f"  {column}: {non_missing} non-missing rows")


def build_source_date_columns(source_table: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, list[str]]:
    output = source_table.copy()

    candidate_date_columns = [
        "VISITDATE",
        "VISDATE",
        "EXAMDATE",
        "USERDATE",
        "COLDATE",
        "TRANDATE",
        "RUNDATE",
        "DUSERDATE",
    ]

    created_columns: list[str] = []
    for original_column in candidate_date_columns:
        if original_column in output.columns:
            new_column = f"VISITDATE_{source_name}_{original_column}"
            output[new_column] = output[original_column]
            created_columns.append(new_column)

    return output, created_columns


def attach_visit_date_column(merged_table: pd.DataFrame) -> pd.DataFrame:
    output = merged_table.copy()

    # Use every date-like column contributed by non-amyloid source tables.
    date_columns_in_priority_order = sorted(
        [column for column in output.columns if column.startswith("VISITDATE_")]
    )

    for column in date_columns_in_priority_order:
        output[column] = pd.to_datetime(output[column], errors="coerce")

    if "VISIT_DATE" in output.columns:
        output = output.drop(columns=["VISIT_DATE"])

    if date_columns_in_priority_order:
        output["VISIT_DATE"] = output[date_columns_in_priority_order].bfill(axis=1).iloc[:, 0]
    else:
        output["VISIT_DATE"] = pd.NaT

    insert_after = None
    if "visit" in output.columns:
        insert_after = output.columns.get_loc("visit") + 1
    elif "VISCODE2" in output.columns:
        insert_after = output.columns.get_loc("VISCODE2") + 1

    if insert_after is not None:
        visit_date_series = output.pop("VISIT_DATE")
        output.insert(insert_after, "VISIT_DATE", visit_date_series)

    return output


def merge_source_table(
    merged_table: pd.DataFrame,
    file_name: str,
    columns_to_keep: list[str],
) -> pd.DataFrame:
    source_table = pd.read_csv(file_name)
    merged_table = merged_table.copy()
    source_table = source_table.copy()

    merged_table["visit_merge"] = merged_table["visit"].apply(normalize_visit_code)
    source_table["VISCODE2_MERGE"] = source_table["VISCODE2"].apply(normalize_visit_code)

    source_name = file_name.replace(".csv", "")
    source_table, extra_cols = build_source_date_columns(source_table, source_name)

    required_columns = list(
        dict.fromkeys(["PTID", "VISCODE2", "VISCODE2_MERGE", *extra_cols, *columns_to_keep])
    )
    source_subset = source_table[required_columns].copy()

    duplicate_mask = source_subset.duplicated(subset=["PTID", "VISCODE2"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        print(
            f"  Warning: {file_name} has {duplicate_count} duplicate PTID/VISCODE2 rows. "
            "Keeping the row with the earliest available visit date when possible."
        )
        sort_columns = [col for col in extra_cols if col in source_subset.columns]
        if sort_columns:
            for sort_column in sort_columns:
                source_subset[sort_column] = pd.to_datetime(source_subset[sort_column], errors="coerce")
            source_subset = source_subset.sort_values(["PTID", "VISCODE2", *sort_columns], na_position="last")
        else:
            source_subset = source_subset.sort_values(["PTID", "VISCODE2"])
        source_subset = source_subset.drop_duplicates(subset=["PTID", "VISCODE2"], keep="first")

    print_merge_diagnostics(
        left_table=merged_table,
        right_table=source_subset,
        left_id_col="subject_id",
        left_visit_col="visit_merge",
        right_id_col="PTID",
        right_visit_col="VISCODE2_MERGE",
        label=file_name,
    )

    merged_output = merged_table.merge(
        source_subset,
        how="left",
        left_on=["subject_id", "visit_merge"],
        right_on=["PTID", "VISCODE2_MERGE"],
    )

    drop_cols = ["PTID", "VISCODE2", "VISCODE2_MERGE", "visit_merge"]
    merged_output = merged_output.drop(columns=[c for c in drop_cols if c in merged_output.columns])
    return merged_output


def main() -> None:
    my_table = pd.read_csv(BASE_TABLE_FILE)
    amyloid_table = pd.read_csv(AMYLOID_TABLE_FILE)

    # Step 1: merge per-visit amyloid columns only.
    amyloid_subset = amyloid_table[AMYLOID_COLUMNS_TO_KEEP].copy().drop_duplicates()

    my_table_work = my_table.copy()
    amyloid_subset_work = amyloid_subset.copy()

    my_table_work["visit_merge"] = my_table_work["visit"].apply(normalize_visit_code)
    amyloid_subset_work["VISCODE2_MERGE"] = amyloid_subset_work["VISCODE2"].apply(normalize_visit_code)

    print_merge_diagnostics(
        left_table=my_table_work,
        right_table=amyloid_subset_work,
        left_id_col="subject_id",
        left_visit_col="visit_merge",
        right_id_col="PTID",
        right_visit_col="VISCODE2_MERGE",
        label="Amyloid (visit-level)",
    )

    merged_table = my_table_work.merge(
        amyloid_subset_work,
        how="left",
        left_on=["subject_id", "visit_merge"],
        right_on=["PTID", "VISCODE2_MERGE"],
    )
    drop_cols = ["PTID", "VISCODE2", "VISCODE2_MERGE", "visit_merge"]
    merged_table = merged_table.drop(columns=[c for c in drop_cols if c in merged_table.columns])

    merged_table = attach_visit_date_column(merged_table)
    print_visitdate_diagnostics(merged_table)
    print("Rows with non-missing VISIT_DATE after amyloid merge:                 ", int(merged_table["VISIT_DATE"].notna().sum()))

    # Step 2: merge all requested non-amyloid tables.
    for merge_config in ADDITIONAL_MERGES:
        merged_table = merge_source_table(
            merged_table=merged_table,
            file_name=merge_config["file_name"],
            columns_to_keep=merge_config["columns_to_keep"],
        )

    print_visitdate_diagnostics(merged_table)
    merged_table = attach_visit_date_column(merged_table)
    merged_table = merged_table.dropna(subset=["VISIT_DATE"]).copy()


    # Step 3: final diagnostics and output.
    print("\nFinal VISIT_DATE non-missing rows:", int(merged_table["VISIT_DATE"].notna().sum()))
    print_visitdate_diagnostics(merged_table)
    print("Unique subjects in final table:   ", int(merged_table["subject_id"].nunique()))

    preview_columns = [
        column
        for column in [
            "subject_id", "visit", "VISIT_DATE", "entry_date",
            "AMYLOID_STATUS", "AMYLOID_STATUS_COMPOSITE_REF", "SCANDATE", "PROCESSDATE"
        ]
        if column in merged_table.columns
    ]
    print("\nFinal visit-date preview:")
    print(merged_table[preview_columns].head(25).to_string(index=False))

    merged_table.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal merged table saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
