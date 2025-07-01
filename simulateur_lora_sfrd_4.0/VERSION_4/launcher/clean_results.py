import argparse
import os

import pandas as pd


def clean_csv(input_path: str, output_path: str | None = None) -> str:
    """Load CSV file, clean it and save to new file.

    Parameters
    ----------
    input_path : str
        Path to CSV file to clean.
    output_path : str | None
        Destination path for cleaned CSV. If None, ``input_path`` is used with
        ``_clean" suffix.

    Returns
    -------
    str
        Path to the cleaned CSV file.
    """
    df = pd.read_csv(input_path)

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # Remove rows with any missing values
    df = df.dropna(how="any")

    # If an ``event_id`` column exists, sort by it for consistency
    if "event_id" in df.columns:
        df = df.sort_values(by="event_id")
    else:
        df = df.sort_index()

    cleaned_path = (
        output_path if output_path is not None else os.path.splitext(input_path)[0] + "_clean.csv"
    )
    df.to_csv(cleaned_path, index=False)

    return cleaned_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Nettoie un fichier CSV de résultats")
    parser.add_argument("csv_file", help="Chemin du fichier CSV à nettoyer")
    parser.add_argument(
        "--output",
        "-o",
        help="Chemin du fichier nettoyé (par défaut <csv_file>_clean.csv)",
    )
    args = parser.parse_args()

    cleaned = clean_csv(args.csv_file, args.output)
    print(f"Fichier nettoyé enregistré dans {cleaned}")


if __name__ == "__main__":
    main()
