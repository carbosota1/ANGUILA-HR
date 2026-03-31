import math
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd


SLOT_TO_HOUR24 = {
    "8AM": 8,
    "9AM": 9,
    "10AM": 10,
    "11AM": 11,
    "12PM": 12,
    "1PM": 13,
    "2PM": 14,
    "3PM": 15,
    "4PM": 16,
    "5PM": 17,
    "6PM": 18,
    "7PM": 19,
    "8PM": 20,
    "9PM": 21,
    "10PM": 22,
}

HOUR24_TO_SLOT = {v: k for k, v in SLOT_TO_HOUR24.items()}


def slot_to_hour24(slot: str) -> int:
    return SLOT_TO_HOUR24[slot.strip().upper()]


def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["status"] == "OK"].copy()

    for col in ["primero", "segundo", "tercero"]:
        df[col] = df[col].astype(str).str.zfill(2)

    df["hour24"] = df["hora"].map(lambda x: slot_to_hour24(str(x)))
    df["datetime"] = pd.to_datetime(
        df["fecha"] + " " + df["hour24"].astype(str) + ":00:00",
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    df["nums"] = df.apply(lambda r: [r["primero"], r["segundo"], r["tercero"]], axis=1)
    return df


def get_recent_windows(train_df: pd.DataFrame, target_dt: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    return {
        "w15": train_df[train_df["datetime"] >= (target_dt - pd.Timedelta(days=15))].copy(),
        "w30": train_df[train_df["datetime"] >= (target_dt - pd.Timedelta(days=30))].copy(),
        "w90": train_df[train_df["datetime"] >= (target_dt - pd.Timedelta(days=90))].copy(),
        "w365": train_df[train_df["datetime"] >= (target_dt - pd.Timedelta(days=365))].copy(),
    }


def draw_has_num(draw_nums: List[str], num: str) -> int:
    return 1 if num in draw_nums else 0


def contingency_from_lag(df: pd.DataFrame, lag: int, observed_num: str, candidate_num: str) -> Tuple[int, int, int, int]:
    """
    a = X=1,Y=1
    b = X=1,Y=0
    c = X=0,Y=1
    d = X=0,Y=0
    X = observed_num apareció en t-lag
    Y = candidate_num apareció en t
    """
    if len(df) <= lag:
        return 0, 0, 0, 0

    a = b = c = d = 0

    rows = df[["nums"]].reset_index(drop=True)
    for i in range(lag, len(rows)):
        x = draw_has_num(rows.iloc[i - lag]["nums"], observed_num)
        y = draw_has_num(rows.iloc[i]["nums"], candidate_num)

        if x == 1 and y == 1:
            a += 1
        elif x == 1 and y == 0:
            b += 1
        elif x == 0 and y == 1:
            c += 1
        else:
            d += 1

    return a, b, c, d


def chi2_from_contingency(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0

    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n

    vals = [(a, e_a), (b, e_b), (c, e_c), (d, e_d)]
    chi2 = 0.0
    for obs, exp in vals:
        if exp > 0:
            chi2 += ((obs - exp) ** 2) / exp
    return float(chi2)


def mutual_info_from_contingency(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0

    p11 = a / n
    p10 = b / n
    p01 = c / n
    p00 = d / n

    px1 = (a + b) / n
    px0 = (c + d) / n
    py1 = (a + c) / n
    py0 = (b + d) / n

    mi = 0.0
    terms = [
        (p11, px1, py1),
        (p10, px1, py0),
        (p01, px0, py1),
        (p00, px0, py0),
    ]
    for pxy, px, py in terms:
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log(pxy / (px * py))
    return float(mi)


def draw_probability(df: pd.DataFrame, num: str) -> float:
    if df.empty:
        return 0.0
    hits = df["nums"].apply(lambda x: 1 if num in x else 0).sum()
    return float(hits) / float(len(df))


def pair_probability(df: pd.DataFrame, a: str, b: str) -> float:
    if df.empty:
        return 0.0
    hits = df["nums"].apply(lambda x: 1 if (a in x and b in x) else 0).sum()
    return float(hits) / float(len(df))


def build_context(train_df: pd.DataFrame) -> Dict[str, List[str]]:
    ctx = {"lag1": [], "lag2": [], "lag3": []}
    if len(train_df) >= 1:
        ctx["lag1"] = list(train_df.iloc[-1]["nums"])
    if len(train_df) >= 2:
        ctx["lag2"] = list(train_df.iloc[-2]["nums"])
    if len(train_df) >= 3:
        ctx["lag3"] = list(train_df.iloc[-3]["nums"])
    return ctx


def score_candidate(
    candidate_num: str,
    windows: Dict[str, pd.DataFrame],
    context: Dict[str, List[str]],
) -> Dict:
    w15 = windows["w15"]
    w30 = windows["w30"]
    w90 = windows["w90"]
    w365 = windows["w365"]

    p15 = draw_probability(w15, candidate_num)
    p30 = draw_probability(w30, candidate_num)
    p90 = draw_probability(w90, candidate_num)
    p365 = draw_probability(w365, candidate_num)

    cond_signal = 0.0
    best_lift = 0.0
    best_mi = 0.0
    best_chi2 = 0.0
    total_support = 0

    lag_weights = {"lag1": 1.00, "lag2": 0.70, "lag3": 0.45}
    lag_to_int = {"lag1": 1, "lag2": 2, "lag3": 3}

    for lag_name, observed_nums in context.items():
        lag = lag_to_int[lag_name]
        for obs_num in observed_nums:
            a, b, c, d = contingency_from_lag(w365, lag, obs_num, candidate_num)
            support = a + b
            if support < 3:
                continue

            total_support += support

            baseline = max(p365, 1e-9)
            cond_prob = a / (a + b) if (a + b) > 0 else 0.0
            lift = cond_prob / baseline if baseline > 0 else 0.0
            chi2 = chi2_from_contingency(a, b, c, d)
            mi = mutual_info_from_contingency(a, b, c, d)

            best_lift = max(best_lift, lift)
            best_mi = max(best_mi, mi)
            best_chi2 = max(best_chi2, chi2)

            lag_signal = (
                0.40 * min(max(lift - 1.0, 0.0), 5.0)
                + 0.35 * min(mi * 25.0, 3.0)
                + 0.25 * min(math.log1p(chi2), 3.0)
            )
            cond_signal += lag_weights[lag_name] * lag_signal

    recency_boost = max(p15 - p90, 0.0)
    stability_boost = max(p30 - p365, 0.0)

    final_score = (
        0.35 * p90
        + 0.15 * p30
        + 0.10 * p15
        + 0.25 * cond_signal
        + 0.10 * recency_boost
        + 0.05 * stability_boost
    )

    return {
        "num": candidate_num,
        "score": float(final_score),
        "p15": float(p15),
        "p30": float(p30),
        "p90": float(p90),
        "p365": float(p365),
        "cond_signal": float(cond_signal),
        "best_lift": float(best_lift),
        "best_mi": float(best_mi),
        "best_chi2": float(best_chi2),
        "support": int(total_support),
    }


def build_pair_scores(top_numbers: List[Dict], windows: Dict[str, pd.DataFrame]) -> List[Dict]:
    w30 = windows["w30"]
    w90 = windows["w90"]
    w365 = windows["w365"]

    num_map = {x["num"]: x for x in top_numbers}
    nums = [x["num"] for x in top_numbers[:8]]

    pair_rows = []
    for a, b in combinations(nums, 2):
        pa90 = draw_probability(w90, a)
        pb90 = draw_probability(w90, b)
        pair90 = pair_probability(w90, a, b)
        pair30 = pair_probability(w30, a, b)
        pair365 = pair_probability(w365, a, b)

        indep = max(pa90 * pb90, 1e-9)
        lift90 = pair90 / indep if indep > 0 else 0.0

        score = (
            0.42 * num_map[a]["score"]
            + 0.42 * num_map[b]["score"]
            + 0.10 * pair90
            + 0.04 * pair30
            + 0.02 * max(lift90 - 1.0, 0.0)
        )

        pair_rows.append({
            "pair": f"{a}-{b}",
            "a": a,
            "b": b,
            "score": float(score),
            "pair30": float(pair30),
            "pair90": float(pair90),
            "pair365": float(pair365),
            "lift90": float(lift90),
        })

    pair_rows = sorted(pair_rows, key=lambda x: x["score"], reverse=True)
    return pair_rows


def classify_edge(best_score: float, best_lift: float, best_mi: float) -> Dict[str, str]:
    if best_score >= 0.42 or (best_lift >= 2.2 and best_mi >= 0.02):
        return {
            "edge_label": "EDGE FUERTE",
            "fire": "🔥🔥🔥🔥🔥🔥",
            "alert": "HIGH",
        }
    if best_score >= 0.26 or (best_lift >= 1.7 and best_mi >= 0.01):
        return {
            "edge_label": "EDGE REAL",
            "fire": "🔥🔥🔥🔥",
            "alert": "MEDIUM",
        }
    if best_score >= 0.16:
        return {
            "edge_label": "EDGE MODERADO",
            "fire": "🔥🔥",
            "alert": "LOW",
        }
    return {
        "edge_label": "SEÑAL DÉBIL",
        "fire": "⚠️",
        "alert": "INFO",
    }


def run_model_for_target(history_df: pd.DataFrame, target_dt: pd.Timestamp) -> Dict:
    train_df = history_df[history_df["datetime"] < target_dt].copy()
    if len(train_df) < 30:
        raise ValueError("Historial insuficiente para modelar. Se necesitan al menos 30 sorteos OK.")

    windows = get_recent_windows(train_df, target_dt)
    context = build_context(train_df)

    candidate_rows = []
    for i in range(100):
        num = f"{i:02d}"
        row = score_candidate(num, windows, context)
        candidate_rows.append(row)

    candidate_rows = sorted(candidate_rows, key=lambda x: x["score"], reverse=True)

    top3 = [x["num"] for x in candidate_rows[:3]]
    top12 = [x["num"] for x in candidate_rows[:12]]
    pairs = build_pair_scores(candidate_rows[:12], windows)
    top_pairs = [x["pair"] for x in pairs[:3]]

    best = candidate_rows[0]
    edge = classify_edge(best["score"], best["best_lift"], best["best_mi"])

    last_draw = train_df.iloc[-1]
    last3 = train_df.tail(3).copy()

    return {
        "target_fecha": str(target_dt.date()),
        "target_hora": HOUR24_TO_SLOT[target_dt.hour],
        "target_datetime": str(target_dt),
        "top3": top3,
        "top12": top12,
        "top_pairs": top_pairs,
        "edge_label": edge["edge_label"],
        "fire": edge["fire"],
        "alert_level": edge["alert"],
        "best_score": round(float(best["score"]), 6),
        "best_lift": round(float(best["best_lift"]), 6),
        "best_mi": round(float(best["best_mi"]), 6),
        "best_chi2": round(float(best["best_chi2"]), 6),
        "support": int(best["support"]),
        "rows_used": int(len(train_df)),
        "context": context,
        "last_draw": {
            "fecha": str(last_draw["fecha"]),
            "hora": str(last_draw["hora"]),
            "nums": list(last_draw["nums"]),
        },
        "last_3_draws": [
            {
                "fecha": str(r["fecha"]),
                "hora": str(r["hora"]),
                "nums": list(r["nums"]),
            }
            for _, r in last3.iterrows()
        ],
        "candidates": candidate_rows[:20],
        "pairs": pairs[:10],
    }