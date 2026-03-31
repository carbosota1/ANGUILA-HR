import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from performance import ensure_dir, load_csv_safe, grade_pick_row

TZ_RD = ZoneInfo("America/Santo_Domingo")

PICK_LOG_COLUMNS = [
    "generated_rd",
    "target_fecha",
    "target_hora",
    "signal_key",
    "edge_label",
    "alert_level",
    "best_score",
    "best_lift",
    "best_mi",
    "best_chi2",
    "rows_used",
    "support",
    "top3",
    "top12",
    "top_pairs",
    "notified",
    "fingerprint",
]

PERFORMANCE_LOG_COLUMNS = [
    "target_fecha",
    "target_hora",
    "signal_key",
    "generated_rd",
    "graded_at_rd",
    "actual_primero",
    "actual_segundo",
    "actual_tercero",
    "actual_nums",
    "actual_primero_pos",
    "actual_segundo_pos",
    "actual_tercero_pos",
    "top3",
    "top12",
    "top_pairs",
    "top3_hits",
    "top12_hits",
    "top_pair_hits",
    "top3_hit_nums",
    "top12_hit_nums",
    "top_pair_hit_pairs",
    "top3_hit_positions",
    "top12_hit_positions",
    "hit_top3",
    "hit_top12",
    "hit_top_pair",
    "edge_label",
    "best_score",
    "best_lift",
    "best_mi",
    "best_chi2",
    "rows_used",
    "support",
    "notified",
    "fingerprint",
]


def rd_now_str() -> str:
    return datetime.now(TZ_RD).strftime("%Y-%m-%d %H:%M:%S")


def load_history_ok(history_path: str) -> pd.DataFrame:
    df = pd.read_csv(history_path, dtype=str).fillna("")
    df = df[df["status"] == "OK"].copy()

    for c in ["primero", "segundo", "tercero"]:
        df[c] = df[c].astype(str).str.zfill(2)

    return df


def load_pick_log(path: str) -> pd.DataFrame:
    return load_csv_safe(path, PICK_LOG_COLUMNS)


def load_performance_log(path: str) -> pd.DataFrame:
    return load_csv_safe(path, PERFORMANCE_LOG_COLUMNS)


def save_performance_log(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def result_lookup(history_df: pd.DataFrame, fecha: str, hora: str) -> pd.DataFrame:
    return history_df[(history_df["fecha"] == fecha) & (history_df["hora"] == hora)].copy()


def grade_pending_picks(history_path: str, pick_log_path: str, performance_log_path: str) -> pd.DataFrame:
    history_df = load_history_ok(history_path)
    pick_df = load_pick_log(pick_log_path)
    perf_df = load_performance_log(performance_log_path)

    already = set(perf_df["signal_key"].astype(str).tolist())
    new_rows = []

    for _, pick_row in pick_df.iterrows():
        signal_key = str(pick_row["signal_key"])
        if not signal_key or signal_key in already:
            continue

        fecha = str(pick_row["target_fecha"])
        hora = str(pick_row["target_hora"])

        result_df = result_lookup(history_df, fecha, hora)
        if result_df.empty:
            continue

        result_row = result_df.iloc[0]
        graded = grade_pick_row(
            pick_row=pick_row,
            result_row=result_row,
            graded_at_rd=rd_now_str(),
        )
        new_rows.append(graded)

    if new_rows:
        perf_df = pd.concat([perf_df, pd.DataFrame(new_rows)], ignore_index=True)
        perf_df = perf_df.drop_duplicates(subset=["signal_key"], keep="last")
        perf_df = perf_df.sort_values(by=["target_fecha", "target_hora", "graded_at_rd"], ascending=[True, True, True])
        save_performance_log(perf_df, performance_log_path)

    return pd.DataFrame(new_rows)