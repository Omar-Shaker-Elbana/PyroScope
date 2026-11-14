from core.src.config import RAW_DIR
from core.src.io_utils import find_files, read_firms_csv

SAMPLE_ROWS = 200_000

def inspect(label, path):
    df = read_firms_csv(path, nrows=SAMPLE_ROWS)
    print(f"\n===== {label} =====")
    print("File:", path.name)
    print(f"Rows read (first {SAMPLE_ROWS:,} at most), columns:", df.shape)
    print("Columns:", list(df.columns))
    if "satellite" in df.columns:
        print("Satellites:", df["satellite"].value_counts().to_dict())
    if "version" in df.columns:
        print("Versions:", df["version"].value_counts().to_dict())
    print("Dtypes:\n", df.dtypes)
    print("First 5 rows:\n", df.head())
    print("Missing values:\n", df.isna().sum())

if __name__ == "__main__":
    for keyword, label in [("modis", "MODIS"), ("viirs", "VIIRS")]:
        for path in find_files(RAW_DIR, keyword):
            inspect(label, path)
