import pandas as pd
from pathlib import Path
import random

# read in list of transcriptions
all_samples = pd.read_excel("C4PrePostDialogueInfo.xlsx")

# Read in utterance files
ufs = Path("Dialog_Utterances_2024").rglob("*Utterances.xlsx")

# Associate sample IDs
dfs = []
merge_cols = ["site", "test", "study_id"]
for uf in ufs:
    df = pd.read_excel(uf)
    df = df.loc[:, merge_cols + ["sample_id"]].drop_duplicates()
    dfs.append(df)

all_utts = pd.concat(dfs, ignore_index=True)
all_samples = all_samples.merge(all_utts, on=merge_cols, how="left")

# prepare table of selected samples
group_cols = ["cycle", "site", "test"]
selected = pd.DataFrame(columns=group_cols + ["study_id", "sample_id", "communication", "file", "stratum_no", "coder1", "coder2"])

# Set number to select from each stratum
n = 5

# Determine coder assignments
coders = ["FK", "SV"]
coders1 = coders * 3
random.shuffle(coders1)

# Stratify by severity (Cycle), test, & site
i = 0
for tup, group in all_samples.groupby(by=group_cols):
    subdf = group.sample(n=n, random_state=98)
    subdf["stratum_no"] = list(range(1, n+1))
    subdf["coder1"] = coders1[i]
    coders_copy = coders.copy()
    coders_copy.remove(coders1[i])
    subdf["coder2"] = coders_copy[0]
    selected = pd.concat([selected, subdf])
    i += 1

selected.to_excel("StratifiedRandomC4PrePostDialogSelection_250924.xlsx", index=False)
