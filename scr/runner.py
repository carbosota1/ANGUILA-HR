import json
import os
import sys
import hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from model_anguilla import load_history_csv, run_model_for_target, HOUR24_TO_SLOT
from telegram import send_telegram

try:
    from scrape_anguilla_enloteria import update_history_with_day
except Exception:
    update_history_with_day = None


TZ_RD = ZoneInfo("America/Santo_Domingo")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

HISTORY_CSV = os.path.join(DATA_DIR, "anguilla_hourly_history.csv")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
PICKS_JSON = os.path.join(OUT_DIR, "picks.json")
REPORT_TXT = os.path.join(OUT_DIR, "daily_report.txt")

SCHEDULE_HOURS = list(range(8, 23))  # 8AM .. 10PM


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rd_now() -> datetime:
    return datetime.now(TZ_RD)


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    ensure_dir(os.path.dirname(STATE_PATH))
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def save_json(path: str, data: dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_next_target_dt(now_rd: datetime) -> datetime:
    """
    Devuelve el próximo sorteo horario.
    Ej:
      7:12 -> hoy 8:00
      8:05 -> hoy 9:00
      21:58 -> hoy 22:00
      22:10 -> mañana 8:00
    """
    today = now_rd.date()

    for h in SCHEDULE_HOURS:
        target = datetime(today.year, today.month, today.day, h, 0, 0, tzinfo=TZ_RD)
        if now_rd < target:
            return target

    tomorrow = today + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, 8, 0, 0, tzinfo=TZ_RD)


def maybe_update_history(now_rd: datetime) -> None:
    if update_history_with_day is None:
        print("SCRAPER: no importado, continúo con historial existente.")
        return

    try:
        update_history_with_day(now_rd.date())
        # cerca del cambio de día/primeras horas conviene asegurar ayer también
        update_history_with_day((now_rd - timedelta(days=1)).date())
        print("SCRAPER: historial actualizado.")
    except Exception as e:
        print(f"SCRAPER ERROR: {e}")


def fingerprint_payload(payload: dict) -> str:
    raw = json.dumps(
        {
            "target_fecha": payload["target_fecha"],
            "target_hora": payload["target_hora"],
            "top3": payload["top3"],
            "top12": payload["top12"],
            "top_pairs": payload["top_pairs"],
            "best_score": payload["best_score"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_signal_key(payload: dict) -> str:
    return f'{payload["target_fecha"]}|Anguilla|{payload["target_hora"]}'


def env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def build_telegram_message(payload: dict, signal_key: str) -> str:
    fire = payload["fire"]
    edge_label = payload["edge_label"]

    top3 = ", ".join(payload["top3"])
    top12 = ", ".join(payload["top12"])
    top_pairs = " | ".join(payload["top_pairs"])

    ctx1 = ", ".join(payload["context"].get("lag1", []))
    ctx2 = ", ".join(payload["context"].get("lag2", []))
    ctx3 = ", ".join(payload["context"].get("lag3", []))

    msg = (
        f"{fire} <b>ANGUILLA CHI/MI ENGINE</b>\n"
        f"{fire} <b>{edge_label}</b>\n\n"
        f"🧩 <b>Señal:</b> {signal_key}\n"
        f"🎯 <b>Target:</b> Anguilla {payload['target_hora']}\n"
        f"📅 <b>Fecha:</b> {payload['target_fecha']}\n\n"
        f"✅ <b>Top3:</b>\n{top3}\n\n"
        f"📌 <b>Top12:</b>\n{top12}\n\n"
        f"🎲 <b>Palé Top3:</b>\n{top_pairs}\n\n"
        f"📊 <b>Debug:</b>\n"
        f"best_score={payload['best_score']} | lift={payload['best_lift']} | mi={payload['best_mi']} | chi2={payload['best_chi2']}\n"
        f"rows_used={payload['rows_used']} | support={payload['support']}\n\n"
        f"🕒 <b>Contexto observado:</b>\n"
        f"lag1: {ctx1}\n"
        f"lag2: {ctx2}\n"
        f"lag3: {ctx3}\n"
    )

    if payload["alert_level"] == "HIGH":
        msg += f"\n{fire} <b>ATENCIÓN: EDGE FUERTE DETECTADO</b> {fire}\n"

    return msg


def should_send_notification(state: dict, signal_key: str, fp: str) -> tuple[bool, str]:
    notify_enabled = env_bool("TELEGRAM_NOTIFY", "1")
    dedupe_enabled = env_bool("NOTIFY_DEDUPE", "1")
    force_notify = env_bool("FORCE_NOTIFY", "0")

    if not notify_enabled:
        return False, "TELEGRAM_NOTIFY=0"

    if force_notify:
        return True, "FORCE_NOTIFY=1"

    if not dedupe_enabled:
        return True, "NOTIFY_DEDUPE=0"

    last_signal_key = state.get("last_signal_key", "")
    last_fingerprint = state.get("last_fingerprint", "")

    if last_signal_key == signal_key:
        return False, "Ya se notificó este target y NOTIFY_DEDUPE=1"

    if last_fingerprint == fp and state.get("last_target_hora") == signal_key:
        return False, "Fingerprint duplicado y NOTIFY_DEDUPE=1"

    return True, "OK"


def main() -> None:
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)

    now_rd = rd_now()
    maybe_update_history(now_rd)

    if not os.path.exists(HISTORY_CSV):
        raise FileNotFoundError(f"No existe historial: {HISTORY_CSV}")

    history_df = load_history_csv(HISTORY_CSV)
    if history_df.empty:
        raise ValueError("Historial vacío después de filtrar status=OK")

    target_dt = get_next_target_dt(now_rd)
    payload = run_model_for_target(history_df, pd.Timestamp(target_dt))

    signal_key = build_signal_key(payload)
    fp = fingerprint_payload(payload)

    payload["signal_key"] = signal_key
    payload["fingerprint"] = fp
    payload["generated_rd"] = now_rd.strftime("%Y-%m-%d %H:%M:%S")
    payload["workflow"] = {
        "notify_enabled": env_bool("TELEGRAM_NOTIFY", "1"),
        "notify_dedupe": env_bool("NOTIFY_DEDUPE", "1"),
        "force_notify": env_bool("FORCE_NOTIFY", "0"),
    }

    save_json(PICKS_JSON, payload)

    report_text = (
        f'ANGUILLA CHI/MI ENGINE\n'
        f'Generated RD: {payload["generated_rd"]}\n'
        f'Target: {payload["target_fecha"]} {payload["target_hora"]}\n'
        f'Signal key: {signal_key}\n'
        f'Edge: {payload["edge_label"]}\n'
        f'Top3: {", ".join(payload["top3"])}\n'
        f'Top12: {", ".join(payload["top12"])}\n'
        f'Pale Top3: {" | ".join(payload["top_pairs"])}\n'
        f'best_score={payload["best_score"]} | lift={payload["best_lift"]} | mi={payload["best_mi"]} | chi2={payload["best_chi2"]}\n'
        f'rows_used={payload["rows_used"]} | support={payload["support"]}\n'
    )
    save_text(REPORT_TXT, report_text)

    state = load_state()
    send_ok, reason = should_send_notification(state, signal_key, fp)

    if send_ok:
        text = build_telegram_message(payload, signal_key)
        sent = send_telegram(text)
        if sent:
            state["last_signal_key"] = signal_key
            state["last_fingerprint"] = fp
            state["last_target_hora"] = signal_key
            state["last_sent_at_rd"] = now_rd.strftime("%Y-%m-%d %H:%M:%S")
            print(f"NOTIFY: enviado -> {signal_key}")
        else:
            print("NOTIFY: fallo el envío")
    else:
        print(f"NOTIFY SKIPPED: {reason}")

    state["last_run_at_rd"] = now_rd.strftime("%Y-%m-%d %H:%M:%S")
    state["last_target_dt"] = str(target_dt)
    state["last_edge_label"] = payload["edge_label"]
    state["last_top3"] = payload["top3"]
    save_state(state)

    print(report_text)


if __name__ == "__main__":
    main()