import json
import duckdb

JSONL_FILE = "openbeautyfacts-products.jsonl"
DB_FILE = "openbeautyfacts.duckdb"
SAMPLE_SIZE = 10000

# 1. Infer columns from sample rows
columns = {}

with open(JSONL_FILE, "r") as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break

        obj = json.loads(line)

        for key, value in obj.items():
            if key not in columns:
                # Keep nested objects and arrays as JSON
                if isinstance(value, (dict, list)):
                    columns[key] = "JSON"
                elif isinstance(value, bool):
                    columns[key] = "BOOLEAN"
                elif isinstance(value, int):
                    columns[key] = "BIGINT"
                elif isinstance(value, float):
                    columns[key] = "DOUBLE"
                else:
                    columns[key] = "VARCHAR"

# 2. Create DuckDB columns
con = duckdb.connect(DB_FILE)

columns_sql = ", ".join(f'"{name}" {dtype}' for name, dtype in columns.items())

con.execute(f"""
    CREATE OR REPLACE TABLE products ({columns_sql})
""")

# 3. Load JSONL data
con.execute(f"""
    INSERT INTO products
    SELECT *
    FROM read_json(
        '{JSONL_FILE}',
        format = 'newline_delimited',
        records = true,
        columns = {{{", ".join(repr(k) + ": " + repr(v) for k, v in columns.items())}}}
    )
""")

print("Rows:", con.execute("SELECT COUNT(*) FROM products").fetchone())
print(con.execute("DESCRIBE products").fetchdf())
