# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:37:49 2026

@author: prend
"""

"""
Digital twinning of patient_95_critical_day.csv using py_replay_bg
Blueprint: multi-meal | Twinning method: MAP
F5-runnable in Spyder. Requires: pip install py-replay-bg
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

from py_replay_bg.py_replay_bg import ReplayBG
from py_replay_bg.visualizer import Visualizer
from py_replay_bg.analyzer import Analyzer
from py_replay_bg.dss.default_dss_handlers import ada_hypotreatments_handler

# =====================================================================
# CONFIG
# =====================================================================
DATA_PATH = "data/simulated dataset/patient_95_critical_day.csv"   # adjust path as needed
SAVE_FOLDER = os.path.join(os.path.abspath(''))
SAVE_NAME = "patient_95_critical_day"

BLUEPRINT = "multi-meal"
TWINNING_METHOD = "map"      # MAP: fast, single point estimate (no CI)

BW = 85.0    # <-- REQUIRED: patient's body weight in kg. Must be set manually.
U2SS = None  # optional: mU/(kg*min). If None, computed as the average basal in the data.

YTS = 5          # CGM sampling period (min), matches the 5-min grid in the data
EXERCISE = False
SEED = 1
VERBOSE = True
PLOT_MODE = True
PARALLELIZE = True

# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    freeze_support()

    if BW is None:
        raise ValueError("Set BW (patient body weight in kg) before running.")

    # --- Load and prepare data -------------------------------------------------
    data = pd.read_csv(DATA_PATH)
    data.t = pd.to_datetime(data['t'], format='%d-%b-%Y %H:%M:%S')

    # keep only the columns py_replay_bg needs (extra columns are ignored anyway,
    # but this keeps the dataframe clean)
    required_cols = ['t', 'glucose', 'bolus', 'bolus_label', 'basal', 'cho', 'cho_label']
    data = data[required_cols]

    print(f"Loaded {len(data)} rows spanning {data.t.iloc[0]} -> {data.t.iloc[-1]}")

    # --- Instantiate ReplayBG ----------------------------------------------------
    rbg = ReplayBG(
        blueprint=BLUEPRINT,
        save_folder=SAVE_FOLDER,
        yts=YTS,
        exercise=EXERCISE,
        seed=SEED,
        verbose=VERBOSE,
        plot_mode=PLOT_MODE,
    )

    # --- Step 1: twin (MAP) -------------------------------------------------
    # print(f"\nTwinning {SAVE_NAME} using MAP...")
    # rbg.twin(
    #     data=data,
    #     bw=BW,
    #     save_name=SAVE_NAME,
    #     twinning_method=TWINNING_METHOD,
    #     parallelize=PARALLELIZE,
    #     u2ss=U2SS,
    # )
    # print(f"Digital twin saved to results/{TWINNING_METHOD}/{TWINNING_METHOD}_{SAVE_NAME}.pkl")

    # --- Step 2: replay using the same inputs used for twinning ---------------
    print("\nRunning replay with the original data...")
    replay_results = rbg.replay(
        data=data,
        bw=BW,
        save_name=SAVE_NAME,
        twinning_method=TWINNING_METHOD,
        save_workspace=True,
        save_suffix='_replay',
    )

 # --- Step 3: test the ADA hypotreatment handler -----------------------------
    # This re-runs the replay on the same twin, but this time enabling the
    # default ADA rescue-carb policy: "take a 15 g hypotreatment if glucose < 70
    # and no hypotreatment was given in the last 15 minutes". Insulin/CHO from
    # the original data are kept as-is (bolus_source/cho_source default to
    # 'data'); only the hypotreatment layer is switched on, so any difference in
    # hypo exposure vs. the Step-2 replay is attributable to the ADA policy.
    print("\nRunning replay with the ADA hypotreatment handler enabled...")
    replay_results_ada = rbg.replay(
        data=data,
        bw=BW,
        save_name=SAVE_NAME,
        twinning_method=TWINNING_METHOD,
        enable_hypotreatments=True,
        hypotreatments_handler=ada_hypotreatments_handler,
        hypotreatments_handler_params=None,   # ada handler has no tunable params
        save_workspace=True,
        save_suffix='_ada_hypo',
    )
 
    Visualizer.plot_replay_results(replay_results_ada, data=data)
 
    analysis_ada = Analyzer.analyze_replay_results(replay_results_ada, data=data)
    print('Fit MARD (ADA hypo run): %.2f %%' % analysis_ada['median']['twin']['mard'])
    print('Mean glucose (ADA hypo run): %.2f mg/dl'
          % analysis_ada['median']['glucose']['variability']['mean_glucose'])
 
    # total grams of hypotreatment carbs administered during the simulation
    ht_total = np.sum(replay_results_ada['hypotreatments']['realizations'][0])
    print('Total hypotreatment CHO given: %.1f g' % ht_total)