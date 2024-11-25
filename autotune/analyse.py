import pickle
import pandas as pd

# Load the results
with open("results.pkl", "rb") as f:
    all_results = pickle.load(f)
    # print(all_results)

for dtype in all_results:
    print(f"\nResults for {dtype}")
    results = all_results[dtype]
    categories = ["TN", "NT", "tn", "nt"]
    result_tables = {c: [] for c in categories}
    for k, v in results.items():        
        for c in categories:
            curr_result = v.get(c, (float('nan'), None, None))
            result_tables[c].append((k, curr_result[0], curr_result[1], curr_result[2]))
    result_tables = {k: pd.DataFrame(v, columns=["name", "flops", "out", "err"]).sort_values(by="flops", ascending=False) for k, v in result_tables.items()}
    #     tn_result = v.get(("TN"), (float('nan'), None, None))
    #     nt_result = v.get(("NT"), (float('nan'), None, None))
    #     result_table.append((k, tn_result[0], nt_result[0], tn_result[1], nt_result[1], tn_result[2], nt_result[2]))
    #     # result_table.append((k, v.get(("TN"), float('nan')), v.get(("NT"), float('nan'))))

    # result_table = pd.DataFrame(result_table, columns=["name", "TN", "NT", "TN_out", "NT_out", "TN_err", "NT_err"])
    print("tn")
    print(result_tables["tn"].head())
    # # get top performed TN setting by sorting the TN column
    # result_table_TN = result_table.sort_values(by="TN", ascending=False)
    # print(result_table_TN.head())

    # # get top performed NT setting by sorting the NT column
    # result_table_NT = result_table.sort_values(by="NT", ascending=False)
    # print(result_table_NT.head())
    # print("\n")