from pathlib import Path

import pandas as pd

files = sorted(Path(".").glob("*.csv"))
df = pd.concat(([pd.read_csv(f) for f in files]), ignore_index=True)
df.to_csv("combined.csv", index=False)
