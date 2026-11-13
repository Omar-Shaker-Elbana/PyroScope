import os
import re
import time
import getpass
import requests
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote

BASE = "https://ladsweb.modaps.eosdis.nasa.gov"
SEARCH_URL = BASE + "/api/v2/content/archives"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "MCD64A1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Egypt bounding box
REGION = "[BBOX]N31.7 S21.7 E36.9 W24.7"

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# MCD64A1 collection 061 HDF filenames
FILE_PATTERN = re.compile(
    r"MCD64A1\.A(\d{7})\.h\d{2}v\d{2}\.061\.[A-Za-z0-9_.-]+\.hdf",
    re.IGNORECASE
)

class HrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self.hrefs.append(value)

def is_valid_hdf4(path):
    try:
        with open(path, "rb") as f:
            signature = f.read(4)
        return signature == bytes.fromhex("0E 03 13 01")
    except OSError:
        return False

def main():
    print("=" * 65)
    print("NASA MCD64A1 COLLECTION 061 DOWNLOADER")
    print("=" * 65)
    print(f"Project: {PROJECT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Dates:   {START_DATE} to {END_DATE}")
    print("Region:  Egypt bounding box")
    print()

    token = getpass.getpass(
        "Paste your NASA LAADS / Earthdata Download Token: "
    ).strip()

    if not token or any(ord(c) < 32 for c in token):
        print("ERROR: Token is empty or contains invalid control characters.")
        return

    session = requests.Session()
    session.headers.update({
        "Authorization": "Bearer " + token,
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "PyroScope-MCD64A1-Downloader/1.0"
    })

    filenames = set()
    print("\nSearching NASA LAADS archive...")

    page = 1
    max_pages = 500

    while page <= max_pages:
        params = {
            "products": "MCD64A1",
            "collections": "61",
            "temporalRanges": START_DATE + ".." + END_DATE,
            "regions": REGION,
            "page": page
        }

        try:
            response = session.get(
                SEARCH_URL,
                params=params,
                timeout=90
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Search failed on page {page}: {exc}")
            break

        parser = HrefParser()
        parser.feed(response.text)

        page_files = set()

        # Search links and page text for matching HDF filenames
        for href in parser.hrefs:
            decoded = unquote(href)
            match = FILE_PATTERN.search(decoded)
            if match:
                page_files.add(match.group(0))

        for match in FILE_PATTERN.finditer(response.text):
            page_files.add(match.group(0))

        new_files = page_files - filenames

        print(
            f"Page {page}: {len(page_files)} matching files, "
            f"{len(new_files)} new"
        )

        if not new_files:
            break

        filenames.update(new_files)
        page += 1

        time.sleep(0.3)

    if not filenames:
        print("\nNo matching HDF files were found.")
        print("Check your token, dates, region, and NASA LAADS access.")
        return

    files = sorted(filenames)
    print(f"\nTotal unique files found: {len(files)}")

    downloaded = 0
    skipped = 0
    failed = 0

    for index, filename in enumerate(files, start=1):
        destination = OUTPUT_DIR / filename
        temporary = OUTPUT_DIR / (filename + ".part")

        print(f"\n[{index}/{len(files)}] {filename}")

        if destination.exists() and is_valid_hdf4(destination):
            print("Already downloaded and verified. Skipping.")
            skipped += 1
            continue

        if destination.exists():
            destination.unlink()

        match = FILE_PATTERN.search(filename)
        year_day = match.group(1)
        year = year_day[:4]
        day_of_year = year_day[4:]

        # LAADS archive directory structure:
        # allData/61/MCD64A1/YYYY/DDD/FILENAME
        relative_path = (
            f"allData/61/MCD64A1/{year}/{day_of_year}/{filename}"
        )

        download_url = (
            BASE + "/api/v2/content/archives/" + relative_path
        )

        try:
            with session.get(
                download_url,
                stream=True,
                timeout=(30, 300)
            ) as response:

                response.raise_for_status()

                content_type = response.headers.get(
                    "Content-Type", "unknown"
                )

                total_bytes = 0
                with open(temporary, "wb") as output:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            output.write(chunk)
                            total_bytes += len(chunk)

            if total_bytes < 1000 or not is_valid_hdf4(temporary):
                with open(temporary, "rb") as f:
                    first_bytes = f.read(200)

                diagnostic = OUTPUT_DIR / (filename + ".failed.txt")
                diagnostic.write_text(
                    "Download did not pass HDF4 validation.\n"
                    f"URL: {download_url}\n"
                    f"Bytes received: {total_bytes}\n"
                    f"Content-Type: {content_type}\n"
                    f"First bytes: {first_bytes!r}\n",
                    encoding="utf-8"
                )

                temporary.unlink(missing_ok=True)
                print(
                    "FAILED validation. Diagnostic saved to "
                    + diagnostic.name
                )
                failed += 1
                continue

            os.replace(temporary, destination)
            print(f"SUCCESS: {total_bytes / (1024 * 1024):.2f} MB")
            downloaded += 1

        except requests.RequestException as exc:
            temporary.unlink(missing_ok=True)
            diagnostic = OUTPUT_DIR / (filename + ".failed.txt")
            diagnostic.write_text(
                f"Download error:\n{exc}\nURL: {download_url}\n",
                encoding="utf-8"
            )
            print(f"DOWNLOAD ERROR: {exc}")
            failed += 1

        except OSError as exc:
            temporary.unlink(missing_ok=True)
            print(f"FILE ERROR: {exc}")
            failed += 1

        time.sleep(0.2)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"New downloads:       {downloaded}")
    print(f"Already verified:    {skipped}")
    print(f"Failed:              {failed}")
    print(f"Output directory:    {OUTPUT_DIR}")

    valid_files = [
        p for p in OUTPUT_DIR.glob("*.hdf")
        if is_valid_hdf4(p)
    ]

    print(f"Verified HDF4 files: {len(valid_files)}")
    print("=" * 65)

    if failed:
        print(
            "\nSome downloads failed. Check the .failed.txt files "
            "inside MCD64A1 for details."
        )

if __name__ == "__main__":
    main()
