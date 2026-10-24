"""
generate_data.py - creates data/student_data.csv (synthetic practice dataset)
============================================================================

    python data/generate_data.py

WHY SYNTHETIC?
The four inputs this project needs are available in a public Kaggle dataset
("Student Performance Factors", StudentPerformanceFactors.csv), but Kaggle needs
an account to download. So that the project runs out of the box, this script
generates a realistic *synthetic* dataset from simple, written-down rules.
If you place the Kaggle file in data/, the project uses it instead
(see src/data_prep.py).

THE RULES USED TO CREATE THE DATA (1,000 students)
  study_hours      average study hours per day, mostly 1-8          (gamma)
  attendance       percentage of classes attended, mostly 60-100    (beta)
  sleep_hours      average sleep per night, mostly 5-9              (normal)
  previous_marks   marks in the previous exam, 0-100, a little
                   higher for students who study more               (normal)

  final_marks = 19
              + 11 * log(1 + study_hours)       more study helps, with diminishing returns
              + 0.22 * (attendance - 75)        attending class helps
              - 1.6 * (sleep_hours - 8)^2       too little OR too much sleep hurts
              + 0.42 * previous_marks           past performance carries over
              + 0.04 * study_hours * (attendance - 75) / 10   study pays off more with attendance
              + random noise (standard deviation 4 marks)
  then limited to the 0-100 range.

DELIBERATE DATA-QUALITY PROBLEMS (so the cleaning step has real work to do)
  * about 3% of values in each input column are blanked (missing values)
  * 25 rows are duplicated
  * 12 impossible values are planted (negative hours, attendance > 100,
    0 hours of sleep, marks above 100)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 1000
OUT = Path(__file__).resolve().parent / "student_data.csv"


def main():
    rng = np.random.default_rng(SEED)

    study = np.clip(rng.gamma(shape=4.0, scale=1.0, size=N), 0.2, 12)
    attendance = np.clip(55 + 45 * rng.beta(4.0, 1.6, size=N), 50, 100)
    sleep = np.clip(rng.normal(7.0, 1.1, size=N), 4, 10.5)
    previous = np.clip(rng.normal(55 + 2.5 * study, 13), 15, 100)

    marks = (19
             + 11 * np.log1p(study)
             + 0.22 * (attendance - 75)
             - 1.6 * (sleep - 8) ** 2
             + 0.42 * previous
             + 0.04 * study * (attendance - 75) / 10
             + rng.normal(0, 4, size=N))
    marks = np.clip(marks, 0, 100)

    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, N + 1)],
        "study_hours": study.round(1),
        "attendance": attendance.round(1),
        "sleep_hours": sleep.round(1),
        "previous_marks": previous.round(0),
        "final_marks": marks.round(1),
    })

    # --- deliberate quality problems ---------------------------------------
    for col in ["study_hours", "attendance", "sleep_hours", "previous_marks"]:
        idx = rng.choice(N, size=int(0.03 * N), replace=False)
        df.loc[idx, col] = np.nan

    bad = rng.choice(N, size=12, replace=False)
    df.loc[bad[0:3], "study_hours"] = [-2.0, -1.0, -3.5]
    df.loc[bad[3:6], "attendance"] = [104.0, 120.0, 135.0]
    df.loc[bad[6:9], "sleep_hours"] = [0.0, 0.0, 0.0]
    df.loc[bad[9:12], "previous_marks"] = [110.0, 125.0, 140.0]

    dups = df.sample(25, random_state=SEED)
    df = pd.concat([df, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows (includes 25 duplicates and planted errors).")


if __name__ == "__main__":
    main()
