# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
# ]
# ///
import pickle
import pandas as pd

# Load the results
with open("results.pkl", "rb") as f:
    results = pickle.load(f)
    # print(all_results)
result_table = []
for k,v in results.items():
    result_table.append((k, v[0], v[1], v[2]))
result_table = pd.DataFrame(result_table, columns=["name", "flops", "out", "err"]).sort_values(by="flops", ascending=False)
print(result_table.head())