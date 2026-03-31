import os
import pandas as pd


def summarize_performance(perf_path: str) -> dict:
    if not os.path.exists(perf_path):
        return {
            "total": 0,
            "hit_top3_rate": 0.0,
            "hit_top12_rate": 0.0,
            "hit_top_pair_rate": 0.0,
        }

    df = pd.read_csv(perf_path, dtype=str).fillna("")
    if df.empty:
        return {
            "total": 0,
            "hit_top3_rate": 0.0,
            "hit_top12_rate": 0.0,
            "hit_top_pair_rate": 0.0,
        }

    for c in ["hit_top3", "hit_top12", "hit_top_pair"]:
        df[c] = df[c].astype(int)

    total = len(df)

    by_edge = (
        df.groupby("edge_label")[["hit_top3", "hit_top12", "hit_top_pair"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    by_hour = (
        df.groupby("target_hora")[["hit_top3", "hit_top12", "hit_top_pair"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "total": total,
        "hit_top3_rate": round(float(df["hit_top3"].mean()), 4),
        "hit_top12_rate": round(float(df["hit_top12"].mean()), 4),
        "hit_top_pair_rate": round(float(df["hit_top_pair"].mean()), 4),
        "by_edge": by_edge,
        "by_hour": by_hour,
    }