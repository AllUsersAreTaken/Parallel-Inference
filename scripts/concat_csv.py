"""Concatenate linter CSVs into one header-once file.

Usage:
    python scripts/concat_csv.py out.csv a.csv b.csv ...

The linter CSVs contain newlines inside quoted fields, so this uses the csv
module rather than line-level concatenation.
"""
import csv
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: concat_csv.py out.csv in1.csv [in2.csv ...]")
    out_path = sys.argv[1]
    in_paths = sys.argv[2:]
    header = None
    n = 0
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        for path in in_paths:
            with open(path, encoding="utf-8", newline="") as fin:
                reader = csv.DictReader(fin)
                if header is None:
                    header = reader.fieldnames or []
                    writer.writerow(header)
                for row in reader:
                    writer.writerow([row.get(k, "") for k in header])
                    n += 1
    print("wrote %d rows -> %s" % (n, out_path))


if __name__ == "__main__":
    main()
