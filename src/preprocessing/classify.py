from pathlib import Path
"""This decides what kind of raw file each CSV is. Will need to change when we add different raw files 
beyond the original ASCC logs."""

def classify_file(path: Path):
    low = path.name.lower()
    if "ros" in low:
        return "ros"
    if "operator" in low or "check" in low or "sheet" in low:
        return "operator"
    return "print"


def iter_real_csv_files(raw_dir: Path):
    for path in raw_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".csv":
            continue
        if "zone.identifier" in path.name.lower():
            continue
        yield path