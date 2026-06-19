import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import ast

# 1. odczyt
table = pq.read_table(
    "garch_prognozy_oos.parquet",
    use_pandas_metadata=False
)

df = table.to_pandas(ignore_metadata=True)


# 2. naprawa kolumn
new_cols = []

for c in df.columns:

    if isinstance(c, tuple):
        model = str(c[0])
        ticker = str(c[1])

    else:
        txt = str(c)

        txt = txt.replace("np.str_(", "")
        txt = txt.replace(")", "")

        try:
            parsed = ast.literal_eval(txt)

            if isinstance(parsed, tuple):
                model = str(parsed[0])
                ticker = str(parsed[1])

            else:
                model = txt
                ticker = ""

        except:
            model = txt
            ticker = ""

    new_cols.append((model, ticker))


df.columns = pd.MultiIndex.from_tuples(new_cols)


# 3. indeks
if "index" in df.columns:
    df = df.set_index("index")

df.index = pd.to_datetime(df.index)


print("Pierwsze kolumny po naprawie:")
print(df.columns[:5])


# 4. zapis bez pandas metadata
table_fixed = pa.Table.from_pandas(
    df,
    preserve_index=True
)

table_fixed = table_fixed.replace_schema_metadata(None)

pq.write_table(
    table_fixed,
    "garch_prognozy_oos_fixed.parquet"
)


print("Gotowe")
