import json
import os
import sys
import hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from model_anguilla import load_history_csv, run_model_for_target
from telegram import send_telegram
from pick_logger import upsert_pick_log
from grader import grade_pending_picks
from performance_summary import summarize_performance

try:
    from scrape_anguilla_enloteria import update_history_with_day, backfill_days
except Exception:
    update_history_with_day = None
    backfill_days = None


TZ_RD = ZoneInfo("America/Santo_Domingo")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

HISTORY_CSV = os.path.join(DATA_DIR, "anguilla_hourly_history.csv")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
PICKS_JSON = os.path.join(OUT_DIR, "picks.json")
REPORT_TXT = os.path.join(OUT_DIR, "daily_report.txt")

PICK_LOG_CSV = os.path.join(DATA_DIR, "pick_log.csv")
PERFORMANCE_LOG_CSV = os.path.join(DATA_DIR, "performance_log.csv")

SCHEDULE_HOURS = list(range(8, 23))  # 8AM..10PM


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rd_now() -> datetime:
    return datetime.now(TZ_RD)


def env_bool(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


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
        update_history_with_day((now_rd - timedelta(days=1)).date())
        print("SCRAPER: historial actualizado.")
    except Exception as e:
        print(f"SCRAPER ERROR: {e}")


def maybe_bootstrap_history() -> None:
    if backfill_days is None:
        print("BOOTSTRAP: backfill_days no disponible.")
        return

    bootstrap_enabled = env_bool("BOOTSTRAP_BACKFILL", "1")
    bootstrap_days = env_int("BOOTSTRAP_DAYS", 365)

    if not bootstrap_enabled:
        print("BOOTSTRAP: desactivado por env.")
        return

    try:
        print(f"BOOTSTRAP: iniciando backfill {bootstrap_days} días...")
        backfill_days(bootstrap_days)
        print("BOOTSTRAP: backfill completado.")
    except Exception as e:
        print(f"BOOTSTRAP ERROR: {e}")


def count_ok_draws(history_df: pd.DataFrame) -> int:
    return int(len(history_df))


def build_signal_key(payload: dict) -> str:
    return f'{payload["target_fecha"]}|Anguilla|{payload["target_hora"]}'


def fingerprint_payload(payload: dict) -> str:
    raw = json.dumps(
        {
            "target_fecha": payload["target_fecha"],
            "target_hora": payload["target_hora"],
            "top3": payload["top3"],
            "top12": payload["top12"],
            "best_score": payload["best_score"],
            "best_lift": payload["best_lift"],
            "best_mi": payload["best_mi"],
            "best_chi2": payload["best_chi2"],
            "edge_label": payload["edge_label"],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def should_send_notification(state: dict, signal_key: str, fp: str, payload: dict) -> tuple[bool, str]:
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


def build_telegram_message(payload: dict, signal_key: str) -> str:
    fire = payload["fire"]
    edge_label = payload["edge_label"]

    top3 = ", ".join(payload["top3"])
    top12 = ", ".join(payload["top12"])
    top_pairs = payload.get("top_pairs", [])

    ctx1 = ", ".join(payload["context"].get("lag1", []))
    ctx2 = ", ".join(payload["context"].get("lag2", []))
    ctx3 = ", ".join(payload["context"].get("lag3", []))

    lines = [
        f"{fire} <b>ANGUILLA CHI/MI ENGINE</b>",
        f"{fire} <b>{edge_label}</b>",
        risk_flag = payload.get("risk_flag", "")
        if risk_flag:
            lines.append(f"⚠️ {risk_flag}")
        "",
        f"🧩 <b>Señal:</b> {signal_key}",
        f"🎯 <b>Target:</b> Anguilla {payload['target_hora']}",
        f"📅 <b>Fecha:</b> {payload['target_fecha']}",
    ]

    if payload["edge_label"] == "NO JUGAR":
        lines.extend([
            "",
            "❌❌❌ <b>NO JUGAR ESTE PICK</b> ❌❌❌",
        ])

    lines.extend([
        "",
        f"✅ <b>Top3:</b>",
        top3,
        "",
        f"📌 <b>Top12:</b>",
        top12,
    ])

    if top_pairs:
        lines.extend([
            "",
            f"🎲 <b>Palé Top3:</b>",
            " | ".join(top_pairs),
        ])

    lines.extend([
        "",
        f"📊 <b>Debug:</b>",
        f"best_score={payload['best_score']} | lift={payload['best_lift']} | mi={payload['best_mi']} | chi2={payload['best_chi2']}",
        f"rows_used={payload['rows_used']} | support={payload['support']} | active_edges={payload.get('active_edges', 0)}",
        "",
        f"🕒 <b>Contexto observado:</b>",
        f"lag1: {ctx1}",
        f"lag2: {ctx2}",
        f"lag3: {ctx3}",
    ])

    if str(payload.get("strong_detected", "0")) == "1":
        lines.extend([
            "",
            f"{fire} <b>ATENCIÓN: EDGE FUERTE DETECTADO</b> {fire}",
        ])

    return "\n".join(lines)


def maybe_send_warmup_message(ok_count: int) -> None:
    if not env_bool("SEND_WARMUP_INFO", "0"):
        return

    text = (
        "🧠 <b>ANGUILLA CHI/MI ENGINE</b>\n\n"
        "⏳ <b>WARMING UP</b>\n"
        f"Historial OK actual: <b>{ok_count}</b>\n"
        "Se necesitan al menos <b>30 sorteos OK</b> para modelar.\n"
        "El workflow no falló; seguirá acumulando data automáticamente."
    )
    send_telegram(text)


def main() -> None:
    ensure_dir(DATA_DIR)
    ensure_dir(OUT_DIR)

    now_rd = rd_now()
    state = load_state()

    maybe_update_history(now_rd)

    if not os.path.exists(HISTORY_CSV):
        save_text(REPORT_TXT, f"No existe historial: {HISTORY_CSV}\n")
        print(f"No existe historial: {HISTORY_CSV}")
        save_state(state)
        return

    history_df = load_history_csv(HISTORY_CSV)
    ok_count = count_ok_draws(history_df)

    min_ok_draws = env_int("MIN_OK_DRAWS", 30)

    if ok_count < min_ok_draws:
        print(f"HISTORIAL CORTO: {ok_count} OK < {min_ok_draws}. Intentando bootstrap...")
        maybe_bootstrap_history()

        if os.path.exists(HISTORY_CSV):
            history_df = load_history_csv(HISTORY_CSV)
            ok_count = count_ok_draws(history_df)

    if ok_count < min_ok_draws:
        report_text = (
            "ANGUILLA CHI/MI ENGINE\n"
            f"Generated RD: {now_rd.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Estado: WARMING UP\n"
            f"OK draws actuales: {ok_count}\n"
            f"Mínimo requerido: {min_ok_draws}\n"
            "Acción: esperando más historial para modelar.\n"
        )
        save_text(REPORT_TXT, report_text)

        state["last_run_at_rd"] = now_rd.strftime("%Y-%m-%d %H:%M:%S")
        state["last_status"] = "WARMING_UP"
        state["ok_draws"] = ok_count
        save_state(state)

        maybe_send_warmup_message(ok_count)
        print(report_text)
        return

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
        "bootstrap_backfill": env_bool("BOOTSTRAP_BACKFILL", "1"),
        "bootstrap_days": env_int("BOOTSTRAP_DAYS", 365),
        "min_ok_draws": min_ok_draws,
    }

    save_json(PICKS_JSON, payload)

    report_lines = [
        "ANGUILLA CHI/MI ENGINE",
        f"Generated RD: {payload['generated_rd']}",
        f"Target: {payload['target_fecha']} {payload['target_hora']}",
        f"Signal key: {signal_key}",
        f"Edge: {payload['edge_label']}",
        f"Top3: {', '.join(payload['top3'])}",
        f"Top12: {', '.join(payload['top12'])}",
        f"best_score={payload['best_score']} | lift={payload['best_lift']} | mi={payload['best_mi']} | chi2={payload['best_chi2']}",
        f"rows_used={payload['rows_used']} | support={payload['support']} | active_edges={payload.get('active_edges', 0)}",
    ]
    if payload["edge_label"] == "NO JUGAR":
        report_lines.append("NO JUGAR ESTE PICK")
    report_text = "\n".join(report_lines)
    save_text(REPORT_TXT, report_text)

    send_ok, reason = should_send_notification(state, signal_key, fp, payload)
    notified_flag = False

    if send_ok:
        text = build_telegram_message(payload, signal_key)
        sent = send_telegram(text)
        if sent:
            notified_flag = True
            state["last_signal_key"] = signal_key
            state["last_fingerprint"] = fp
            state["last_target_hora"] = signal_key
            state["last_sent_at_rd"] = now_rd.strftime("%Y-%m-%d %H:%M:%S")
            print(f"NOTIFY: enviado -> {signal_key}")
        else:
            print("NOTIFY: fallo el envío")
    else:
        print(f"NOTIFY SKIPPED: {reason}")

    upsert_pick_log(PICK_LOG_CSV, payload, notified=notified_flag)

    graded_df = grade_pending_picks(
        history_path=HISTORY_CSV,
        pick_log_path=PICK_LOG_CSV,
        performance_log_path=PERFORMANCE_LOG_CSV,
    )
    if not graded_df.empty:
        print(f"GRADED: {len(graded_df)} pick(s) evaluados")

    perf_summary = summarize_performance(PERFORMANCE_LOG_CSV)
    print("PERFORMANCE SUMMARY:", perf_summary)

    state["last_run_at_rd"] = now_rd.strftime("%Y-%m-%d %H:%M:%S")
    state["last_target_dt"] = str(target_dt)
    state["last_edge_label"] = payload["edge_label"]
    state["last_top3"] = payload["top3"]
    state["last_status"] = "OK"
    state["ok_draws"] = ok_count
    save_state(state)

    print(report_text)


if __name__ == "__main__":
    main()