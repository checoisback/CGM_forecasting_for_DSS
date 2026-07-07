"""
debug_reactive_handler.py
─────────────────────────
Standalone debugging harness for the reactive (ADA-style) hypotreatment handler
in a ReplayBG multi-meal replay.

What it does, in order:
  1. Runs a replay with the LIBRARY DEFAULT `ada_hypotreatments_handler` and shows
     that it double-fires at each hypo onset (the off-by-one guard bug).
  2. Runs the same replay with the FIXED `reactive_hypotreatment_handler`
     (dss-memory cooldown) and shows the double-fire is gone.
  3. Instruments the handler to log exactly what it sees each call in a chosen
     window, so you can watch the guard/threshold decision step by step.
  4. Reports the index offset between the handler's internal glucose frame and
     res["glucose"]["median"] (the array the plot draws), which explains why a
     naive datetime axis mislabels the fire time.

Expected files in the working dir (adjust PATHS below):
  - patient_95_critical_day.csv
  - results/map/map_patient_95.pkl        (the MAP digital twin)

Run:  python debug_reactive_handler.py
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_CSV   = "data/simulated dataset/patient_95_critical_day.csv"
SAVE_NAME  = "patient_95"
SAVE_FOLDER = "digital twin"
BW         = 85.0
SIM_START  = pd.Timestamp("2000-01-01 06:00")   # only for human-readable clock in logs
HYPO_THR   = 70.0
DOSE_G     = 15.0
COOLDOWN   = 15
# window (in sample index) to log the handler's step-by-step decisions
LOG_WINDOWS = [(258, 278), (1140, 1170)]


# ─────────────────────────────────────────────────────────────────────────────
# The fixed reactive handler (dss-memory cooldown, no array readback)
# ─────────────────────────────────────────────────────────────────────────────
def reactive_hypotreatment_handler(
        glucose, meal_announcement, meal_type,
        hypotreatments, bolus, basal,
        time, time_index, dss):
    
    ht = 0.0
    p = dss.hypotreatments_handler_params
    threshold    = p.get("threshold",    HYPO_THR)
    dose_g       = p.get("dose_g",       DOSE_G)
    cooldown_min = p.get("cooldown_min", COOLDOWN)
    last_fire    = p.get("_last_fire_time", -10**9)

    if glucose[time_index] < threshold and (time_index - last_fire) >= cooldown_min:
        ht = dose_g
        p["_last_fire_time"] = time_index
    return ht, dss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def clock(k):
    return (SIM_START + pd.Timedelta(minutes=int(k))).strftime("%H:%M")


def load_data():
    data = pd.read_csv(DATA_CSV)
    data["t"] = pd.to_datetime(data["t"], dayfirst=True)
    return data


def make_rbg():
    from py_replay_bg.py_replay_bg import ReplayBG
    return ReplayBG(blueprint="multi-meal", save_folder=SAVE_FOLDER,
                    yts=5, exercise=False, seed=1, verbose=False, plot_mode=True)


def run_replay(rbg, data, handler, params=None, suffix="_dbg"):
    return rbg.replay(
        data=data, bw=BW, save_name=SAVE_NAME, twinning_method="map",
        enable_hypotreatments=True,
        hypotreatments_handler=handler,
        hypotreatments_handler_params=(params or {}),
        save_suffix=suffix, n_replay=1,
    )


def fired_report(res, label):
    ht = np.array(res["hypotreatments"]["realizations"])[0]
    g  = np.array(res["glucose"]["median"])
    idx = np.where(ht > 0)[0]
    print(f"\n[{label}]  {len(idx)} treatment(s), {ht.sum():.0f} g CHO total")
    for k in idx:
        print(f"    sample={k:5d}  clock~{clock(k)}  g_median={g[k]:5.1f}  ht={ht[k]:.0f} g")
    gaps = np.diff(idx)
    if len(gaps):
        print(f"    min gap between fires: {gaps.min()} min "
              f"({'OK' if gaps.min() >= COOLDOWN else 'DOUBLE-FIRE < cooldown!'})")
    return ht, g, idx


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(os.path.join(SAVE_FOLDER, "results", "map",
                                        f"map_{SAVE_NAME}.pkl")):
        raise FileNotFoundError(
            f"Twin not found at {SAVE_FOLDER}/results/map/map_{SAVE_NAME}.pkl — "
            "check SAVE_FOLDER / SAVE_NAME.")

    data = load_data()

    # 1) Library default -> should double-fire ---------------------------------
    from py_replay_bg.dss.default_dss_handlers import ada_hypotreatments_handler
    rbg = make_rbg()
    res_default = run_replay(rbg, data, ada_hypotreatments_handler, suffix="_default")
    fired_report(res_default, "LIBRARY DEFAULT ada_hypotreatments_handler")

    # 2) Fixed handler -> no double-fire ---------------------------------------
    res_fixed = run_replay(
        rbg, data, reactive_hypotreatment_handler,
        params={"_last_fire_time": -10**9,
                "threshold": HYPO_THR, "dose_g": DOSE_G, "cooldown_min": COOLDOWN},
        suffix="_fixed")
    fired_report(res_fixed, "FIXED reactive_hypotreatment_handler")

    # 3) Step-by-step log of what the handler sees -----------------------------
    seen = []

    def logging_handler(glucose, meal_announcement, meal_type,
                        hypotreatments, bolus, basal, time, time_index, dss):
        ht = 0.0
        p = dss.hypotreatments_handler_params
        last = p.get("_last_fire_time", -10**9)
        g_here = glucose[time_index]
        cond_thr = g_here < HYPO_THR
        cond_cd  = (time_index - last) >= COOLDOWN
        if cond_thr and cond_cd:
            ht = DOSE_G
            p["_last_fire_time"] = time_index
        for lo, hi in LOG_WINDOWS:
            if lo <= time_index <= hi:
                seen.append((time_index, float(g_here), bool(cond_thr),
                             bool(cond_cd), float(ht)))
        return ht, dss

    run_replay(rbg, data, logging_handler,
               params={"_last_fire_time": -10**9}, suffix="_log")

    print("\n[STEP-BY-STEP handler view]  (g is glucose[time_index] as the handler sees it)")
    print("  sample  clock   g_seen   <70   cooldown_ok   ht")
    for ti, g, c_thr, c_cd, ht in seen:
        mark = "  <-- FIRE" if ht > 0 else ""
        print(f"  {ti:5d}  {clock(ti)}  {g:6.1f}   {str(c_thr):5}  "
              f"{str(c_cd):5}        {ht:.0f}{mark}")

    # 4) Handler-frame vs reported-median glucose offset -----------------------
    #    (why a reconstructed datetime axis mislabels the fire time)
    captured = {"gh": None}

    def probe(glucose, *a):
        dss = a[-1]; ti = a[-2]
        if captured["gh"] is None or len(glucose) > len(captured["gh"]):
            captured["gh"] = np.array(glucose)
        ht = 0.0
        p = dss.hypotreatments_handler_params
        last = p.get("_last_fire_time", -10**9)
        if glucose[ti] < HYPO_THR and (ti - last) >= COOLDOWN:
            ht = DOSE_G; p["_last_fire_time"] = ti
        return ht, dss

    res_probe = run_replay(rbg, data, probe,
                           params={"_last_fire_time": -10**9}, suffix="_probe")
    gh = captured["gh"]
    gm = np.array(res_probe["glucose"]["median"])

    def first_below(arr, lo=250, hi=290):
        for i in range(lo, hi):
            if arr[i] < HYPO_THR:
                return i
        return None

    ch_h, ch_m = first_below(gh), first_below(gm)
    print("\n[GLUCOSE FRAME OFFSET]")
    print(f"    handler-frame glucose crosses {HYPO_THR:.0f} at sample {ch_h}")
    print(f"    res['glucose']['median']  crosses {HYPO_THR:.0f} at sample {ch_m}")
    if ch_h is not None and ch_m is not None:
        print(f"    offset = {ch_h - ch_m} samples "
              f"(handler sees the same value ~{ch_h - ch_m} samples later than the "
              f"plotted median array)")
        print("    -> this is why a reconstructed clock axis mislabels the fire time; "
              "plot against sample index instead.")


if __name__ == "__main__":
    main()