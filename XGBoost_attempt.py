
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


DATA_FILE_CANDIDATES = [
    "Final.csv",
    "Final",
    "Merged_My_Table_UCBERKELEY_AMY_Final_13Apr2026.csv",
    "Merged_My_Table_UCBERKELEY_AMY_13Apr2026.csv",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
HIGH_MISSINGNESS_THRESHOLD = 0.85
MODEL_INPUT_TABLE_FILE = "survival_xgboost_model_input_table.csv"
ENCODED_FEATURE_TABLE_FILE = "survival_xgboost_encoded_features.csv"
TEST_PREDICTIONS_FILE = "survival_xgboost_test_predictions.csv"

# Do not use identifiers, direct outcome columns, or post-baseline leakage columns.
LEAKAGE_OR_NON_PREDICTOR_COLUMNS = {
    "subject_id",
    "visit",
    "PTID",
    "VISCODE2",
    "AMYLOID_STATUS",
    "AMYLOID_STATUS_COMPOSITE_REF",
    "SCANDATE",
    "PROCESSDATE",
    "FIRST_POS_AMY_DATE",
    "LAST_AMY_DATE",
    "FIRST_AMY_POS_DATE",
    "POS_FROM_FIRST_IMAGING",
    "LAST_VISIT_DATE",
    "TTAMY",
    "AMY_EVENT",
    "AMY_ROW_EXCLUSION_REASON",
    "OUTCOME",
    "entry_date",
    "VISIT_DATE",
}

OPTIONAL_EXCLUDE_COLUMNS = {
    "PTDOB",
}



def resolve_data_file() -> Path:
    for candidate in DATA_FILE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            print(f"Using data file: {path}")
            return path
    raise FileNotFoundError(
        "Could not find any expected input file. Looked for: "
        + ", ".join(DATA_FILE_CANDIDATES)
    )



def load_data(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded data: {data.shape[0]} rows x {data.shape[1]} columns")
    return data



def choose_baseline_rows(data: pd.DataFrame) -> pd.DataFrame:
    working = data.copy()

    if "VISIT_DATE" in working.columns:
        working["VISIT_DATE"] = pd.to_datetime(working["VISIT_DATE"], errors="coerce")
    if "entry_date" in working.columns:
        working["entry_date"] = pd.to_datetime(working["entry_date"], errors="coerce")

    for column in ["TTAMY", "AMY_EVENT", "POS_FROM_FIRST_IMAGING"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    for column in ["FIRST_AMY_POS_DATE", "LAST_VISIT_DATE"]:
        if column in working.columns:
            working[column] = pd.to_datetime(working[column], errors="coerce")

    if "subject_id" not in working.columns:
        raise ValueError("The input table must contain a 'subject_id' column.")

    sort_columns = [column for column in ["subject_id", "VISIT_DATE", "entry_date", "visit"] if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, na_position="last")

    baseline = working.drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)

    print(f"Baseline table: {baseline.shape[0]} subjects x {baseline.shape[1]} columns")
    if "TTAMY" in baseline.columns:
        print(f"Subjects with non-missing TTAMY: {int(baseline['TTAMY'].notna().sum())}")
    if "AMY_EVENT" in baseline.columns:
        print(f"Subjects with non-missing AMY_EVENT: {int(baseline['AMY_EVENT'].notna().sum())}")
    return baseline



def keep_valid_survival_rows(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["TTAMY", "AMY_EVENT", "POS_FROM_FIRST_IMAGING"]
    missing_required = [column for column in required_columns if column not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required survival columns: {missing_required}")

    filtered = data.copy()
    filtered["TTAMY"] = pd.to_numeric(filtered["TTAMY"], errors="coerce")
    filtered["AMY_EVENT"] = pd.to_numeric(filtered["AMY_EVENT"], errors="coerce")
    filtered["POS_FROM_FIRST_IMAGING"] = pd.to_numeric(filtered["POS_FROM_FIRST_IMAGING"], errors="coerce")

    print("TTAMY summary before filtering:")
    print(filtered["TTAMY"].describe(include="all"))
    print("AMY_EVENT counts before filtering:")
    print(filtered["AMY_EVENT"].value_counts(dropna=False).sort_index())
    print("POS_FROM_FIRST_IMAGING counts before filtering:")
    print(filtered["POS_FROM_FIRST_IMAGING"].value_counts(dropna=False).sort_index())

    # Exclude subjects already positive at first imaging: they do not have an observed time-to-positivity interval.
    filtered = filtered[filtered["POS_FROM_FIRST_IMAGING"] != 1].copy()

    # Keep only subjects with usable survival labels.
    filtered = filtered.dropna(subset=["TTAMY", "AMY_EVENT"])
    filtered = filtered[np.isfinite(filtered["TTAMY"])].copy()
    filtered = filtered[filtered["TTAMY"] >= 0].copy()
    filtered["AMY_EVENT"] = filtered["AMY_EVENT"].astype(int)

    print(f"Rows with valid survival outcome: {filtered.shape[0]}")
    print("Event counts after filtering:")
    print(filtered["AMY_EVENT"].value_counts(dropna=False).sort_index())
    return filtered



def drop_high_missingness_columns(data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    missing_fraction = data.isna().mean()
    to_drop = sorted(
        column
        for column, fraction in missing_fraction.items()
        if fraction > threshold and column not in {"TTAMY", "AMY_EVENT"}
    )
    reduced = data.drop(columns=to_drop, errors="ignore")
    return reduced, to_drop



def build_feature_matrix(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    modeling_data = data.copy()

    modeling_data, high_missing_columns = drop_high_missingness_columns(
        modeling_data,
        threshold=HIGH_MISSINGNESS_THRESHOLD,
    )

    excluded_columns = sorted(
        column
        for column in modeling_data.columns
        if column in LEAKAGE_OR_NON_PREDICTOR_COLUMNS or column in OPTIONAL_EXCLUDE_COLUMNS
    )

    feature_columns = [
        column
        for column in modeling_data.columns
        if column not in LEAKAGE_OR_NON_PREDICTOR_COLUMNS
        and column not in OPTIONAL_EXCLUDE_COLUMNS
    ]

    X = modeling_data[feature_columns].copy()
    y_time = modeling_data["TTAMY"].copy()
    y_event = modeling_data["AMY_EVENT"].copy()

    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    for column in numeric_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].fillna(X[column].median())

    for column in categorical_columns:
        X[column] = X[column].astype("string").fillna("Missing")

    X = pd.get_dummies(X, columns=categorical_columns, dummy_na=False)

    print(f"Feature matrix after encoding: {X.shape[0]} rows x {X.shape[1]} features")
    return X, y_time, y_event, excluded_columns, high_missing_columns



def save_model_input_tables(
    survival_data: pd.DataFrame,
    X: pd.DataFrame,
    y_time: pd.Series,
    y_event: pd.Series,
) -> None:
    survival_data.to_csv(MODEL_INPUT_TABLE_FILE, index=False)

    encoded_feature_table = X.copy()
    encoded_feature_table["TTAMY"] = y_time.values
    encoded_feature_table["AMY_EVENT"] = y_event.values
    encoded_feature_table.to_csv(ENCODED_FEATURE_TABLE_FILE, index=False)

    print(f"\nSaved pre-encoding model input table to {MODEL_INPUT_TABLE_FILE}")
    print(f"Saved encoded feature table to {ENCODED_FEATURE_TABLE_FILE}")

    preview_columns = [
        column
        for column in [
            "subject_id",
            "entry_age",
            "GENOTYPE",
            "MMSCORE",
            "MOCA",
            "LDELTOTAL",
            "FIRST_AMY_POS_DATE",
            "POS_FROM_FIRST_IMAGING",
            "LAST_VISIT_DATE",
            "TTAMY",
            "AMY_EVENT",
        ]
        if column in survival_data.columns
    ]

    print("\nPreview of model input table:")
    if preview_columns:
        print(survival_data[preview_columns].head(10).to_string(index=False))
    else:
        print(survival_data.head(10).to_string(index=False))

    print("\nPreview of encoded feature table:")
    print(encoded_feature_table.head(10).to_string(index=False))



def make_aft_bounds(times: pd.Series, events: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lower_bound = times.to_numpy(dtype=float)
    upper_bound = times.to_numpy(dtype=float)
    upper_bound[events.to_numpy(dtype=int) == 0] = np.inf
    return lower_bound, upper_bound



def build_dmatrix(X: pd.DataFrame, y_time: pd.Series, y_event: pd.Series) -> xgb.DMatrix:
    dmatrix = xgb.DMatrix(X)
    lower_bound, upper_bound = make_aft_bounds(y_time, y_event)
    dmatrix.set_float_info("label_lower_bound", lower_bound)
    dmatrix.set_float_info("label_upper_bound", upper_bound)
    return dmatrix



def concordance_index_from_risk(times: Iterable[float], events: Iterable[int], risk_scores: Iterable[float]) -> float:
    times_array = np.asarray(list(times), dtype=float)
    events_array = np.asarray(list(events), dtype=int)
    risk_array = np.asarray(list(risk_scores), dtype=float)

    permissible = 0.0
    concordant = 0.0
    ties = 0.0

    n = len(times_array)
    for i in range(n):
        for j in range(i + 1, n):
            time_i = times_array[i]
            time_j = times_array[j]
            event_i = events_array[i]
            event_j = events_array[j]
            risk_i = risk_array[i]
            risk_j = risk_array[j]

            if time_i == time_j and event_i == 0 and event_j == 0:
                continue

            if time_i < time_j and event_i == 1:
                permissible += 1
                if risk_i > risk_j:
                    concordant += 1
                elif risk_i == risk_j:
                    ties += 1
            elif time_j < time_i and event_j == 1:
                permissible += 1
                if risk_j > risk_i:
                    concordant += 1
                elif risk_i == risk_j:
                    ties += 1

    if permissible == 0:
        return math.nan

    return (concordant + 0.5 * ties) / permissible



def train_survival_xgboost(
    X_train: pd.DataFrame,
    y_time_train: pd.Series,
    y_event_train: pd.Series,
    X_valid: pd.DataFrame,
    y_time_valid: pd.Series,
    y_event_valid: pd.Series,
) -> xgb.Booster:
    dtrain = build_dmatrix(X_train, y_time_train, y_event_train)
    dvalid = build_dmatrix(X_valid, y_time_valid, y_event_valid)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": RANDOM_STATE,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )
    return booster



def print_top_features(model: xgb.Booster, top_n: int = 25) -> None:
    score = model.get_score(importance_type="gain")
    if not score:
        print("\nNo feature importance scores were returned.")
        return

    importance_table = (
        pd.DataFrame({"feature": list(score.keys()), "importance": list(score.values())})
        .sort_values("importance", ascending=False)
    )
    print("\nTop feature importances:")
    print(importance_table.head(top_n).to_string(index=False))



def main() -> None:
    file_path = resolve_data_file()
    raw_data = load_data(file_path)
    baseline_data = choose_baseline_rows(raw_data)
    survival_data = keep_valid_survival_rows(baseline_data)

    if survival_data.empty:
        raise ValueError(
            "No subjects have valid TTAMY/AMY_EVENT values after baseline selection. "
            "Check that the Final table contains TTAMY, AMY_EVENT, and POS_FROM_FIRST_IMAGING."
        )

    X, y_time, y_event, excluded_columns, high_missing_columns = build_feature_matrix(survival_data)
    save_model_input_tables(survival_data, X, y_time, y_event)

    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
        X,
        y_time,
        y_event,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_event,
    )

    X_train_inner, X_valid, y_time_train_inner, y_time_valid, y_event_train_inner, y_event_valid = train_test_split(
        X_train,
        y_time_train,
        y_event_train,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        random_state=RANDOM_STATE,
        stratify=y_event_train,
    )

    print(f"Training subjects: {len(X_train_inner)}")
    print(f"Validation subjects: {len(X_valid)}")
    print(f"Test subjects: {len(X_test)}")

    model = train_survival_xgboost(
        X_train_inner,
        y_time_train_inner,
        y_event_train_inner,
        X_valid,
        y_time_valid,
        y_event_valid,
    )

    pred_train_time = model.predict(xgb.DMatrix(X_train_inner))
    pred_valid_time = model.predict(xgb.DMatrix(X_valid))
    pred_test_time = model.predict(xgb.DMatrix(X_test))

    train_risk = -pred_train_time
    valid_risk = -pred_valid_time
    test_risk = -pred_test_time

    train_c_index = concordance_index_from_risk(y_time_train_inner, y_event_train_inner, train_risk)
    valid_c_index = concordance_index_from_risk(y_time_valid, y_event_valid, valid_risk)
    test_c_index = concordance_index_from_risk(y_time_test, y_event_test, test_risk)

    results = pd.DataFrame(
        {
            "subject_id": survival_data.loc[X_test.index, "subject_id"].values,
            "observed_time_years": y_time_test.values,
            "event": y_event_test.values,
            "predicted_time_years": pred_test_time,
            "predicted_risk": test_risk,
        }
    ).sort_values("predicted_risk", ascending=False)
    results.to_csv(TEST_PREDICTIONS_FILE, index=False)

    print("\nDropped for leakage / non-predictor reasons:")
    print(excluded_columns)

    print("\nDropped for high missingness:")
    print(high_missing_columns)

    print(f"\nTrain c-index: {train_c_index:.4f}")
    print(f"Validation c-index: {valid_c_index:.4f}")
    print(f"Test c-index:  {test_c_index:.4f}")
    print(f"\nSaved test predictions to {TEST_PREDICTIONS_FILE}")
    print_top_features(model, top_n=25)
    print(f"\nModel input file: {MODEL_INPUT_TABLE_FILE}")
    print(f"Encoded feature file: {ENCODED_FEATURE_TABLE_FILE}")


if __name__ == "__main__":
    main()
