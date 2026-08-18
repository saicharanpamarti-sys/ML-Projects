import os
from idlelib.pyshell import usage_msg

import pandas as pd
from numpy.ma.core import shape

#df = pd.read_csv(r"C:\Users\saich\Downloads\ML\PythonProject12\placement_predict_50k Dataset (3) 1(in).csv")
#print(df)
#print(df.dtypes)
#print(df['CGPA'].describe(),df['Gender'].describe())#remember the syntax,2nd way
#print(df[['CGPA','AttendancePercent']].describe())#1st way
#print(list(df))
#print(shape(df))
#print(df.head())
DATA_PATH = r"C:\Users\saich\Downloads\ML\PythonProject12\placement_predict_50k Dataset (3) 1(in).csv"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    return df

def get_data_summary(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(10).to_dict("records"),
    }
    return summary

if __name__ == "__main__":
    df = load_data()
    print(get_data_summary(df))

