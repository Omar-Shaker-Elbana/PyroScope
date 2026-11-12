from pathlib import Path
import pandas as pd

# Read text-like columns as strings so zero-padded times ("0005") and mixed
# confidence values survive chunked reading.
TEXT_COLS = {
    "acq_date": str, "acq_time": str, "satellite": str, "instrument": str,
    "confidence": str, "version": str, "daynight": str, "type": str,
}

def find_files(directory: Path, keyword: str) -> list[Path]:
    """All .csv/.txt files in `directory` whose name contains `keyword` (case-insensitive)."""
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".csv", ".txt"}
        and keyword.lower() in p.name.lower()
    )
    if not files:
        raise FileNotFoundError(
            f"No file with '{keyword}' in its name found in {directory}\n"
            "Download the dataset from NASA FIRMS and save it in data/raw/ "
            "with 'modis' or 'viirs' in the filename (see README.md)."
        )
    return files

def read_firms_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a FIRMS CSV/TXT export with a header row (optionally only the first rows)."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, low_memory=False, nrows=nrows, dtype=TEXT_COLS)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def iter_firms_csv(path: Path, chunksize: int):
    """Yield a FIRMS CSV in chunks so global multi-year files fit in memory."""
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, dtype=TEXT_COLS):
        chunk.columns = [str(c).strip().lower() for c in chunk.columns]
        yield chunk
