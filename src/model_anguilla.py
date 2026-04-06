import math
from typing import Dict, List, Tuple
import pandas as pd


SLOT_TO_HOUR24 = {
    "8AM": 8, "9AM": 9, "10AM": 10, "11AM": 11,
    "12PM": 12, "1PM": 13, "2PM": 14, "3PM": 15,
    "4PM": 16, "5PM": 17, "6PM": 18, "7PM": 19,
    "8PM": 20, "9PM": 21, "10PM": 22,
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
    df["datetime"] = pd.to_datetime(df["fecha"] + " " + df["hour24"].astype(str) + ":00:00")

    df = df.sort_values("datetime").reset_index(drop=True)
    df["nums"] = df.apply(lambda r: [r["primero"], r["segundo"], r["tercero"]], axis=1)
    return df


def draw_probability(df: pd.DataFrame, num: str) -> float:
    if df.empty:
        return 0.0
    hits = df["nums"].apply(lambda x: num in x).sum()
    return hits / len(df)


def contingency(df, lag, obs, cand):
    if len(df) <= lag:
        return 0, 0, 0, 0

    a = b = c = d = 0
    rows = df["nums"].tolist()

    for i in range(lag, len(rows)):
        x = obs in rows[i - lag]
        y = cand in rows[i]

        if x and y: a += 1
        elif x and not y: b += 1
        elif not x and y: c += 1
        else: d += 1

    return a, b, c, d


def chi2(a, b, c, d):
    n = a + b + c + d
    if n == 0: return 0
    try:
        return (n * (a*d - b*c)**2) / ((a+b)*(c+d)*(a+c)*(b+d) + 1e-9)
    except:
        return 0


def mi(a, b, c, d):
    n = a + b + c + d
    if n == 0: return 0

    def p(x): return x / n

    p11, p10, p01, p00 = p(a), p(b), p(c), p(d)
    px1, px0 = p(a+b), p(c+d)
    py1, py0 = p(a+c), p(b+d)

    res = 0
    for pxy, px, py in [(p11, px1, py1), (p10, px1, py0), (p01, px0, py1), (p00, px0, py0)]:
        if pxy > 0:
            res += pxy * math.log(pxy/(px*py + 1e-9) + 1e-9)

    return res


# =========================
# 🔥 CLASIFICACIÓN FINAL
# =========================
def classify_edge(score, lift, mi_val, chi2_val, active_edges):

    # ❌ NO JUGAR (solo cuando realmente es malo)
    if score < 0.65 or mi_val < 0.00045:
        return {
            "edge_label": "NO JUGAR",
            "fire": "❌",
            "alert": "NONE",
            "strong_detected": "0",
            "risk_flag": "",
        }

    # 🔥 ELITE
    if score >= 0.90 and lift >= 2.5 and chi2_val >= 12 and mi_val >= 0.001:
        return {
            "edge_label": "EDGE ELITE",
            "fire": "🔥🔥🔥🔥🔥🔥",
            "alert": "HIGH",
            "strong_detected": "1",
            "risk_flag": "",
        }

    # 🔥 REAL
    if score >= 0.70 and lift >= 2.0 and chi2_val >= 6 and mi_val >= 0.00055:
        risk = "⚠️ Riesgo: alta densidad de edges" if active_edges >= 7 else ""
        return {
            "edge_label": "EDGE REAL",
            "fire": "🔥🔥🔥🔥",
            "alert": "MEDIUM",
            "strong_detected": "0",
            "risk_flag": risk,
        }

    # 🔥 MODERADO
    risk = "⚠️ Riesgo: señal inestable" if active_edges >= 7 else ""
    return {
        "edge_label": "EDGE MODERADO",
        "fire": "🔥🔥",
        "alert": "LOW",
        "strong_detected": "0",
        "risk_flag": risk,
    }


def run_model_for_target(df, target_dt):

    target_dt = pd.Timestamp(target_dt).tz_localize(None)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    context = {
        "lag1": train.iloc[-1]["nums"],
        "lag2": train.iloc[-2]["nums"],
        "lag3": train.iloc[-3]["nums"],
    }

    results = []

    for i in range(100):
        num = f"{i:02d}"

        p = draw_probability(train, num)
        total_signal = 0
        best_lift = best_mi = best_chi2 = 0
        active_edges = 0

        for lag_name, obs_nums in context.items():
            lag = int(lag_name[-1])

            for obs in obs_nums:
                a, b, c, d = contingency(train, lag, obs, num)

                if a + b < 5:
                    continue

                cond = a / (a + b)
                lift = cond / (p + 1e-9)
                chi = chi2(a, b, c, d)
                mi_val = mi(a, b, c, d)

                best_lift = max(best_lift, lift)
                best_mi = max(best_mi, mi_val)
                best_chi2 = max(best_chi2, chi)

                if lift > 1.1 or chi > 1 or mi_val > 0.0001:
                    active_edges += 1

                total_signal += (lift - 1) + math.log1p(chi) + mi_val * 1500

        score = p + total_signal * 0.05

        results.append({
            "num": num,
            "score": score,
            "lift": best_lift,
            "mi": best_mi,
            "chi2": best_chi2,
            "active_edges": active_edges
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    best = results[0]
    edge = classify_edge(best["score"], best["lift"], best["mi"], best["chi2"], best["active_edges"])

    return {
        "target_fecha": str(target_dt.date()),
        "target_hora": HOUR24_TO_SLOT[target_dt.hour],
        "top3": [x["num"] for x in results[:3]],
        "top12": [x["num"] for x in results[:12]],
        "edge_label": edge["edge_label"],
        "fire": edge["fire"],
        "risk_flag": edge["risk_flag"],
        "strong_detected": edge["strong_detected"],
        "best_score": round(best["score"], 6),
        "best_lift": round(best["lift"], 6),
        "best_mi": round(best["mi"], 6),
        "best_chi2": round(best["chi2"], 6),
        "active_edges": best["active_edges"],
        "rows_used": len(train),
        "support": 0,
        "context": context,
    }