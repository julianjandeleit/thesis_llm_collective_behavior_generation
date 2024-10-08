#%%
## just a workspace to do ipython stuff in, nothing persistent


# %%
import pandas as pd
df = pd.read_pickle("dataset.pkl")
# %%
df
# %%
from collections import defaultdict
mtypes = defaultdict(list)
for index, row in df.iterrows():
    mtypes[row["parameters"].objective_params.__class__.__name__].append(index)
mtypes
# %%

for t, p in mtypes.items():
    print(t)
    print(df["argos"][p[0]])
    print("\n\n")
# %%
