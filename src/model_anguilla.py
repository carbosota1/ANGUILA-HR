import math
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


def to_naive_timestamp(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["status"] == "OK"].copy()

    for col in ["primero", "segundo", "tercero"]:
        df[col] = df[col].astype(str).str.zfill(2)

    df["hour24"] = df["hora"].map(lambda x: slot_to_hour24(str(x)))
    df["datetime"] = pd.to_datetime(
        df["fecha"] + " " + df["hour24"].astype(str) + ":00:00",
        errors="coerce",
    )

    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].map(to_naive_timestamp)
    df = df.sort_values("datetime").reset_index(drop=True)
    df["nums"] = df.apply(lambda r: [r["primero"], r["segundo"], r["tercero"]], axis=1)
    return df


def draw_probability(df: pd.DataFrame, num: str) -> float:
    if df.empty:
        return 0.0
    hits = df["nums"].apply(lambda x: num in x).sum()
    return float(hits) / float(len(df))


def contingency(df: pd.DataFrame, lag: int, obs: str, cand: str) -> Tuple[int, int, int, int]:
    if len(df) <= lag:
        return 0, 0, 0, 0

    a = b = c = d = 0
    rows = df["nums"].tolist()

    for i in range(lag, len(rows)):
        x = obs in rows[i - lag]
        y = cand in rows[i]

        if x and y:
            a += 1
        elif x and not y:
            b += 1
        elif not x and y:
            c += 1
        else:
            d += 1

    return a, b, c, d


def chi2(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0

    denom = ((a + b) * (c + d) * (a + c) * (b + d)) + 1e-9
    return float((n * ((a * d - b * c) ** 2)) / denom)


def mi(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0

    def p(x: int) -> float:
        return x / n

    p11, p10, p01, p00 = p(a), p(b), p(c), p(d)
    px1, px0 = p(a + b), p(c + d)
    py1, py0 = p(a + c), p(b + d)

    res = 0.0
    for pxy, px, py in [
        (p11, px1, py1),
        (p10, px1, py0),
        (p01, px0, py1),
        (p00, px0, py0),
    ]:
        if pxy > 0:
            res += pxy * math.log((pxy / ((px * py) + 1e-9)) + 1e-9)

    return float(res)


def classify_edge(
    best_score: float,
    best_lift: float,
    best_mi: float,
    best_chi2: float,
    active_edges: int,
) -> Dict[str, str]:

    # NO JUGAR: señal realmente mala
    if best_score < 0.65 or best_mi < 0.00045:
        return {
            "edge_label": "NO JUGAR",
            "fire": "❌",
            "alert": "NONE",
            "strong_detected": "0",
            "risk_flag": "",
        }

    # EDGE ELITE
    if (
        best_score >= 0.90
        and best_lift >= 2.50
        and best_chi2 >= 12.0
        and best_mi >= 0.00100
    ):
        return {
            "edge_label": "EDGE ELITE",
            "fire": "🔥🔥🔥🔥🔥🔥",
            "alert": "HIGH",
            "strong_detected": "1",
            "risk_flag": "",
        }

    # EDGE REAL
    if (
        best_score >= 0.70
        and best_lift >= 2.00
        and best_chi2 >= 6.0
        and best_mi >= 0.00055
    ):
        risk = "Riesgo: alta densidad de edges" if active_edges >= 7 else ""
        return {
            "edge_label": "EDGE REAL",
            "fire": "🔥🔥🔥🔥",
            "alert": "MEDIUM",
            "strong_detected": "0",
            "risk_flag": risk,
        }

    # EDGE MODERADO
    risk = "Riesgo: señal inestable" if active_edges >= 7 else ""
    return {
        "edge_label": "EDGE MODERADO",
        "fire": "🔥🔥",
        "alert": "LOW",
        "strong_detected": "0",
        "risk_flag": risk,
    }


def run_model_for_target(df: pd.DataFrame, target_dt) -> Dict:
    # FIX timezone
    target_dt = to_naive_timestamp(target_dt)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").map(to_naive_timestamp)
    df = df.dropna(subset=["datetime"]).copy()

    # train primero
    train = df[df["datetime"] < target_dt].copy()

    if len(train) < 3:
        raise ValueError("No hay suficiente data para generar contexto (mínimo 3 sorteos)")
    if len(train) < 30:
        raise ValueError("Historial insuficiente para modelar. Se necesitan al menos 30 sorteos OK.")

    # contexto después
    context = {
        "lag1": train.iloc[-1]["nums"],
        "lag2": train.iloc[-2]["nums"],
        "lag3": train.iloc[-3]["nums"],
    }

    results = []

    for i in range(100):
        num = f"{i:02d}"

        p = draw_probability(train, num)
        total_signal = 0.0
        best_lift = 0.0
        best_mi = 0.0
        best_chi2 = 0.0
        active_edges = 0
        support = 0

        for lag_name, obs_nums in context.items():
            lag = int(lag_name[-1])

            for obs in obs_nums:
                a, b, c, d = contingency(train, lag, obs, num)

                if a + b < 5:
                    continue

                support += (a + b)

                cond = a / (a + b)
                lift = cond / (p + 1e-9)
                chi_val = chi2(a, b, c, d)
                mi_val = mi(a, b, c, d)

                best_lift = max(best_lift, lift)
                best_mi = max(best_mi, mi_val)
                best_chi2 = max(best_chi2, chi_val)

                if lift > 1.10 or chi_val > 1.0 or mi_val > 0.00010:
                    active_edges += 1

                total_signal += (lift - 1.0) + math.log1p(chi_val) + (mi_val * 1500.0)

        score = p + (total_signal * 0.05)

        results.append({
            "num": num,
            "score": float(score),
            "lift": float(best_lift),
            "mi": float(best_mi),
            "chi2": float(best_chi2),
            "active_edges": int(active_edges),
            "support": int(support),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results[0]

    edge = classify_edge(
        best_score=best["score"],
        best_lift=best["lift"],
        best_mi=best["mi"],
        best_chi2=best["chi2"],
        active_edges=best["active_edges"],
    )

    last_draw = train.iloc[-1]
    last_3 = train.tail(3)

    return {
        "target_fecha": str(target_dt.date()),
        "target_hora": HOUR24_TO_SLOT[target_dt.hour],
        "target_datetime": str(target_dt),
        "top3": [x["num"] for x in results[:3]],
        "top12": [x["num"] for x in results[:12]],
        "top_pairs": [],
        "pairs": [],
        "edge_label": edge["edge_label"],
        "fire": edge["fire"],
        "alert_level": edge["alert"],
        "strong_detected": edge["strong_detected"],
        "risk_flag": edge["risk_flag"],
        "best_score": round(best["score"], 6),
        "best_lift": round(best["lift"], 6),
        "best_mi": round(best["mi"], 6),
        "best_chi2": round(best["chi2"], 6),
        "active_edges": int(best["active_edges"]),
        "rows_used": int(len(train)),
        "support": int(best["support"]),
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
            for _, r in last_3.iterrows()
        ],
        "candidates": results[:20],
    }