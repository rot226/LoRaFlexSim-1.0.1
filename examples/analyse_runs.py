import sys
import pandas as pd


def main(file):
    df = pd.read_csv(file)
    grouped = df.groupby("run").mean(numeric_only=True)
    print(grouped)
    out = file.replace(".csv", "_means.csv")
    grouped.to_csv(out)
    print(f"Moyennes sauvegard\u00e9es dans {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyse_runs.py resultats.csv")
    else:
        main(sys.argv[1])
