import os
import pandas as pd


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_pick_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=PICK_LOG_COLUMNS)

    df = pd.read_csv(path, dtype=str).fillna("")
    for c in PICK_LOG_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[PICK_LOG_COLUMNS]


def save_pick_log(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, encoding="utf-8-sig")


def payload_to_pick_row(payload: dict, notified: bool) -> dict:
    return {
        "generated_rd": str(payload.get("generated_rd", "")),
        "target_fecha": str(payload.get("target_fecha", "")),
        "target_hora": str(payload.get("target_hora", "")),
        "signal_key": str(payload.get("signal_key", "")),
        "edge_label": str(payload.get("edge_label", "")),
        "alert_level": str(payload.get("alert_level", "")),
        "best_score": str(payload.get("best_score", "")),
        "best_lift": str(payload.get("best_lift", "")),
        "best_mi": str(payload.get("best_mi", "")),
        "best_chi2": str(payload.get("best_chi2", "")),
        "rows_used": str(payload.get("rows_used", "")),
        "support": str(payload.get("support", "")),
        "top3": "|".join(payload.get("top3", [])),
        "top12": "|".join(payload.get("top12", [])),
        "top_pairs": "|".join(payload.get("top_pairs", [])),
        "notified": "1" if notified else "0",
        "fingerprint": str(payload.get("fingerprint", "")),
    }


def upsert_pick_log(path: str, payload: dict, notified: bool) -> None:
    df = load_pick_log(path)
    row = payload_to_pick_row(payload, notified)

    signal_key = row["signal_key"]
    if not signal_key:
        raise ValueError("payload sin signal_key")

    mask = df["signal_key"] == signal_key
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df.sort_values(by=["target_fecha", "target_hora", "generated_rd"], ascending=[True, True, True])
    save_pick_log(df, path)