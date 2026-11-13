from pathlib import Path
from datetime import date
from getpass import getpass
import csv
import re
import subprocess
import json
import time

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path("MCD64A1").resolve()
ROOT.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://ladsweb.modaps.eosdis.nasa.gov/archive/"
    "allData/61/MCD64A1"
)

COLLECTION = "061"
START_YEAR = 2000

CONFIG = ROOT / ".mcd64_download.conf"
COOKIES = ROOT / ".mcd64_cookies.txt"
MANIFEST = ROOT / "download_manifest.csv"
REQUIRED_TILES_FILE = ROOT / "required_tiles.json"

# Match Collection 061 monthly HDF granules.
FILENAME_PATTERN = re.compile(
    r"^MCD64A1\.A(\d{4})(\d{3})\."
    r"(h\d{2}v\d{2})\.061\.(\d+)\.hdf$"
)

HDF4_SIGNATURE = bytes.fromhex("0E 03 13 01")

# ------------------------------------------------------------
# Load the required tiles
# ------------------------------------------------------------

if REQUIRED_TILES_FILE.exists():
    with REQUIRED_TILES_FILE.open("r", encoding="utf-8") as f:
        settings = json.load(f)

    TILES = settings.get(
        "tiles",
        ["h20v05", "h20v06", "h21v05", "h21v06"]
    )
else:
    TILES = ["h20v05", "h20v06", "h21v05", "h21v06"]

TILES = sorted(set(TILES))

print("\nMCD64A1 Collection 061 downloader")
print("---------------------------------")
print("Destination:", ROOT)
print("Tiles:", ", ".join(TILES))
print("Start year:", START_YEAR)
print("End month:", date.today().strftime("%Y-%m"))
print()

# ------------------------------------------------------------
# Authentication
# ------------------------------------------------------------

token = getpass("Enter your Earthdata token: ").strip()

if not token:
    raise SystemExit("ERROR: Empty Earthdata token.")

safe_token = token.replace("\\", "\\\\").replace('"', '\\"')

CONFIG.write_text(
    f'header = "Authorization: Bearer {safe_token}"\n',
    encoding="utf-8"
)

# Do not keep an additional copy of the token in memory.
del token
del safe_token

# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

manifest_fields = [
    "year",
    "doy",
    "tile",
    "filename",
    "path",
    "size_bytes",
    "status",
    "http_status",
    "final_url"
]

with MANIFEST.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=manifest_fields)
    writer.writeheader()

def record(**values):
    row = {field: "" for field in manifest_fields}
    row.update(values)

    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writerow(row)

def curl_download(url, output_path):
    """
    Download using curl and the authenticated Earthdata session.
    Return HTTP status, final URL, content type, and curl error.
    """

    result = subprocess.run(
        [
            "curl.exe",
            "-L",
            "-sS",
            "--retry", "3",
            "--retry-delay", "2",
            "--config", str(CONFIG),
            "-o", str(output_path),
            "-w",
            "HTTP=%{http_code}\n"
            "FINAL_URL=%{url_effective}\n"
            "TYPE=%{content_type}\n",
            url
        ],
        capture_output=True,
        text=True,
        errors="replace"
    )

    metadata = {}

    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value

    return {
        "http": metadata.get("HTTP", ""),
        "final_url": metadata.get("FINAL_URL", ""),
        "content_type": metadata.get("TYPE", ""),
        "error": result.stderr.strip(),
        "returncode": result.returncode
    }

def valid_hdf(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False

    with path.open("rb") as f:
        signature = f.read(4)

    return signature == HDF4_SIGNATURE

# ------------------------------------------------------------
# Main download
# ------------------------------------------------------------

total_found = 0
downloaded = 0
already_present = 0
failed = 0
missing_months = 0

today = date.today()

try:
    for year in range(START_YEAR, today.year + 1):

        last_month = today.month if year == today.year else 12

        for month in (6, 9, 10):

            month_date = date(year, month, 1)
            doy = month_date.timetuple().tm_yday

            doy_text = f"{doy:03d}"

            listing_url = (
                f"{BASE_URL}/{year}/{doy_text}.csv"
            )

            listing_file = (
                ROOT / f"_listing_{year}_{doy_text}.csv"
            )

            print(
                f"\n[{year}-{month:02d}] "
                f"Checking archive listing..."
            )

            listing_info = curl_download(
                listing_url,
                listing_file
            )

            if listing_info["http"] != "200":
                missing_months += 1

                print(
                    "  Listing unavailable:",
                    listing_info["http"] or "no HTTP response"
                )

                listing_file.unlink(missing_ok=True)
                time.sleep(0.15)
                continue

            if "oauth/authorize" in listing_info["final_url"]:
                raise RuntimeError(
                    "NASA redirected to Earthdata Login. "
                    "Authentication was rejected."
                )

            if not listing_file.exists():
                failed += 1
                print("  ERROR: Listing file was not created.")
                continue

            # ------------------------------------------------
            # Read the monthly CSV and select required tiles
            # ------------------------------------------------

            matches = {}

            try:
                with listing_file.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig"
                ) as f:

                    reader = csv.DictReader(f)

                    if not reader.fieldnames:
                        raise ValueError("CSV has no header.")

                    name_column = next(
                        (
                            name for name in reader.fieldnames
                            if name.strip().lower() == "name"
                        ),
                        None
                    )

                    if name_column is None:
                        raise ValueError(
                            "CSV does not contain a name column."
                        )

                    for row in reader:
                        filename = (
                            row.get(name_column) or ""
                        ).strip()

                        match = FILENAME_PATTERN.match(filename)

                        if not match:
                            continue

                        file_year = int(match.group(1))
                        file_doy = int(match.group(2))
                        tile = match.group(3)

                        if (
                            file_year == year
                            and file_doy == doy
                            and tile in TILES
                        ):
                            # Keep the latest production version
                            # if more than one version is listed.
                            if (
                                tile not in matches
                                or filename > matches[tile]
                            ):
                                matches[tile] = filename

            except Exception as exc:
                failed += 1
                print("  ERROR reading CSV:", exc)
                listing_file.unlink(missing_ok=True)
                continue

            listing_file.unlink(missing_ok=True)

            if not matches:
                print("  No matching granules for required tiles.")
                time.sleep(0.15)
                continue

            # ------------------------------------------------
            # Download matching granules
            # ------------------------------------------------

            for tile in TILES:

                filename = matches.get(tile)

                if not filename:
                    print(f"  {tile}: not listed for this month")
                    continue

                total_found += 1

                destination_dir = ROOT / str(year) / doy_text
                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                destination = destination_dir / filename
                part_file = destination.with_suffix(
                    destination.suffix + ".part"
                )

                # Do not download a valid file again.
                if valid_hdf(destination):
                    already_present += 1

                    print(f"  {tile}: already downloaded and valid")

                    record(
                        year=year,
                        doy=doy_text,
                        tile=tile,
                        filename=filename,
                        path=str(destination.relative_to(ROOT)),
                        size_bytes=destination.stat().st_size,
                        status="ALREADY_VALID",
                        http_status="200",
                        final_url=""
                    )
                    continue

                # Remove an invalid previous file if present.
                destination.unlink(missing_ok=True)
                part_file.unlink(missing_ok=True)

                file_url = (
                    f"{BASE_URL}/{year}/{doy_text}/{filename}"
                )

                print(f"  {tile}: downloading {filename}")

                info = curl_download(file_url, part_file)

                if info["http"] != "200":
                    failed += 1
                    part_file.unlink(missing_ok=True)

                    print(
                        f"    FAILED: HTTP {info['http'] or 'unknown'}"
                    )

                    record(
                        year=year,
                        doy=doy_text,
                        tile=tile,
                        filename=filename,
                        path=str(destination.relative_to(ROOT)),
                        status="DOWNLOAD_FAILED",
                        http_status=info["http"],
                        final_url=info["final_url"]
                    )
                    continue

                if "oauth/authorize" in info["final_url"]:
                    part_file.unlink(missing_ok=True)
                    raise RuntimeError(
                        "NASA redirected to Earthdata Login "
                        "during an HDF download."
                    )

                if not valid_hdf(part_file):
                    failed += 1
                    part_file.unlink(missing_ok=True)

                    print(
                        "    FAILED: Downloaded file is not "
                        "a valid HDF4 file."
                    )

                    record(
                        year=year,
                        doy=doy_text,
                        tile=tile,
                        filename=filename,
                        path=str(destination.relative_to(ROOT)),
                        status="INVALID_HDF",
                        http_status=info["http"],
                        final_url=info["final_url"]
                    )
                    continue

                part_file.replace(destination)
                downloaded += 1

                size = destination.stat().st_size

                print(
                    f"    OK: {size:,} bytes; HDF4 signature valid"
                )

                record(
                    year=year,
                    doy=doy_text,
                    tile=tile,
                    filename=filename,
                    path=str(destination.relative_to(ROOT)),
                    size_bytes=size,
                    status="DOWNLOADED",
                    http_status=info["http"],
                    final_url=info["final_url"]
                )

                time.sleep(0.15)

            print(
                f"  Progress: {downloaded} downloaded, "
                f"{already_present} already valid, "
                f"{failed} failed"
            )

finally:
    CONFIG.unlink(missing_ok=True)
    COOKIES.unlink(missing_ok=True)

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

hdf_files = list(ROOT.rglob("*.hdf"))

valid_files = sum(
    1 for path in hdf_files if valid_hdf(path)
)

print("\n")
print("=" * 58)
print("MCD64A1 DOWNLOAD SUMMARY")
print("=" * 58)
print("Tiles:", ", ".join(TILES))
print("Matching granules found:", total_found)
print("New downloads:", downloaded)
print("Already valid:", already_present)
print("Failed downloads:", failed)
print("Months without available listings:", missing_months)
print("HDF files currently present:", len(hdf_files))
print("HDF files with valid signatures:", valid_files)
print("Manifest:", MANIFEST)
print("=" * 58)

if failed:
    print(
        "\nWARNING: Some files failed. Check download_manifest.csv "
        "for their status."
    )
else:
    print("\nDownload scan completed without recorded file failures.")
