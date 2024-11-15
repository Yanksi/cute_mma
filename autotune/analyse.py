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
        tn_result = v.get(("TN"), (float('nan'), None, None))
        nt_result = v.get(("NT"), (float('nan'), None, None))
        result_table.append((k, tn_result[0], nt_result[0], tn_result[1], nt_result[1], tn_result[2], nt_result[2]))
        # result_table.append((k, v.get(("TN"), float('nan')), v.get(("NT"), float('nan'))))

    result_table = pd.DataFrame(result_table, columns=["name", "TN", "NT", "TN_out", "NT_out", "TN_err", "NT_err"])

    # get top performed TN setting by sorting the TN column
    result_table_TN = result_table.sort_values(by="TN", ascending=False)
    print(result_table_TN.head())

    # get top performed NT setting by sorting the NT column
    result_table_NT = result_table.sort_values(by="NT", ascending=False)
    print(result_table_NT.head())
    print("\n")