import os
from itertools import combinations

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_csv_safe(path: str, columns: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path, dtype=str).fillna("")
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    return df[columns]


def normalize_num(x: str) -> str:
    s = str(x).strip()
    return s.zfill(2) if s.isdigit() else s


def parse_pipe_list(text: str) -> list[str]:
    text = str(text).strip()
    if not text:
        return []
    return [normalize_num(x) for x in text.split("|") if str(x).strip()]


def make_pipe_list(values: list[str]) -> str:
    return "|".join([normalize_num(x) for x in values])


def build_position_map(result_row: pd.Series) -> dict:
    return {
        normalize_num(result_row["primero"]): ["1ra"],
        normalize_num(result_row["segundo"]): ["2da"],
        normalize_num(result_row["tercero"]): ["3ra"],
    }


def positions_for_predicted(nums: list[str], pos_map: dict) -> tuple[list[str], list[str]]:
    hit_nums = []
    hit_positions = []

    for n in nums:
        n2 = normalize_num(n)
        if n2 in pos_map:
            hit_nums.append(n2)
            hit_positions.append(f"{n2}:{'/'.join(pos_map[n2])}")

    return hit_nums, hit_positions


def pair_hits(actual_nums: list[str], predicted_pairs: list[str]) -> tuple[int, str]:
    actual_set = set(actual_nums)
    actual_pairs = {f"{a}-{b}" for a, b in combinations(sorted(actual_set), 2)}
    hits = [p for p in predicted_pairs if p in actual_pairs]
    return len(hits), "|".join(hits)


def grade_pick_row(pick_row: pd.Series, result_row: pd.Series, graded_at_rd: str) -> dict:
    actual_primero = normalize_num(result_row["primero"])
    actual_segundo = normalize_num(result_row["segundo"])
    actual_tercero = normalize_num(result_row["tercero"])
    actual_nums = [actual_primero, actual_segundo, actual_tercero]

    pos_map = build_position_map(result_row)

    top3 = parse_pipe_list(pick_row["top3"])
    top12 = parse_pipe_list(pick_row["top12"])
    top_pairs = [x.strip() for x in str(pick_row["top_pairs"]).split("|") if x.strip()]

    top3_hits_list, top3_hit_positions = positions_for_predicted(top3, pos_map)
    top12_hits_list, top12_hit_positions = positions_for_predicted(top12, pos_map)
    pair_hit_count, pair_hit_pairs = pair_hits(actual_nums, top_pairs)

    return {
        "target_fecha": str(pick_row["target_fecha"]),
        "target_hora": str(pick_row["target_hora"]),
        "signal_key": str(pick_row["signal_key"]),
        "generated_rd": str(pick_row["generated_rd"]),
        "graded_at_rd": graded_at_rd,

        "actual_primero": actual_primero,
        "actual_segundo": actual_segundo,
        "actual_tercero": actual_tercero,
        "actual_nums": make_pipe_list(actual_nums),

        "actual_primero_pos": "1ra",
        "actual_segundo_pos": "2da",
        "actual_tercero_pos": "3ra",

        "top3": str(pick_row["top3"]),
        "top12": str(pick_row["top12"]),
        "top_pairs": str(pick_row["top_pairs"]),

        "top3_hits": str(len(top3_hits_list)),
        "top12_hits": str(len(top12_hits_list)),
        "top_pair_hits": str(pair_hit_count),

        "top3_hit_nums": make_pipe_list(top3_hits_list),
        "top12_hit_nums": make_pipe_list(top12_hits_list),
        "top_pair_hit_pairs": pair_hit_pairs,

        "top3_hit_positions": "|".join(top3_hit_positions),
        "top12_hit_positions": "|".join(top12_hit_positions),

        "hit_top3": "1" if len(top3_hits_list) > 0 else "0",
        "hit_top12": "1" if len(top12_hits_list) > 0 else "0",
        "hit_top_pair": "1" if pair_hit_count > 0 else "0",

        "edge_label": str(pick_row["edge_label"]),
        "best_score": str(pick_row["best_score"]),
        "best_lift": str(pick_row["best_lift"]),
        "best_mi": str(pick_row["best_mi"]),
        "best_chi2": str(pick_row["best_chi2"]),
        "rows_used": str(pick_row["rows_used"]),
        "support": str(pick_row["support"]),
        "notified": str(pick_row["notified"]),
        "fingerprint": str(pick_row["fingerprint"]),
    }