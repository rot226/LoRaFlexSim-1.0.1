"""Génère tous les graphes de l'article C (placeholder)."""

from pathlib import Path

from article_c.common.utils import ensure_dir


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    for step in ("step1", "step2"):
        output_dir = base_dir / step / "plots" / "output"
        ensure_dir(output_dir)
        # Placeholder: aucun graphe n'est généré pour le moment.


if __name__ == "__main__":
    main()
