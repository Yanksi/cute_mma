import pickle
import pandas as pd

# Load the results
with open("results.pkl", "rb") as f:
    all_results = pickle.load(f)
    # print(all_results)

for dtype in all_results:
    print(f"\nResults for {dtype}")
    results = all_results[dtype]
    result_table = []
    for k, v in results.items():
        result_table.append((k, v.get(("TN"), float('nan')), v.get(("NT"), float('nan'))))

    result_table = pd.DataFrame(result_table, columns=["name", "TN", "NT"])

    # get top performed TN setting by sorting the TN column
    result_table_TN = result_table.sort_values(by="TN", ascending=False)
    print(result_table_TN.head())

    # get top performed NT setting by sorting the NT column
    result_table_NT = result_table.sort_values(by="NT", ascending=False)
    print(result_table_NT.head())
    print("\n")