import pandas as pd
import json
import os

def export_csv(path, name):
    filename = os.path.join(path, name)
    print(filename)
    df = pd.read_json(filename)
    print(df)
    trace = df['traceEvents'].apply(pd.Series)
    trace.to_csv(filename.replace('.json','.csv'))

def read_json(path, name):
    filename = os.path.join(path, name)
    with open(filename) as f:
        data = json.load(f)['traceEvents']
        trace = []
        for (k, v) in enumerate(data):
            if "cat" in v.keys():
                if v["cat"] != "Kernel":
                    trace.append({"categary": v["cat"], "name": v["name"], "Occupancy %": 0})
                else:
                    print(v)
                    trace.append({"categary": v["cat"], "name": v["name"], "Occupancy %": v["args"]["est. achieved occupancy %"]})
            else:
                trace.append({"categary": "None", "name": v["name"], "Occupancy %": 0})
        df = pd.DataFrame(trace)
        df.to_csv(filename.replace('.json','.csv'), index_label="Kernel_id")


if __name__ == "__main__":
    read_json("../train/trace","trace.json")