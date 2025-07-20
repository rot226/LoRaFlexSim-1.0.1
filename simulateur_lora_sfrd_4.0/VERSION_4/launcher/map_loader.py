import json
from pathlib import Path
from typing import Iterable, Union, List


def load_map(source: Union[str, Path, Iterable[Iterable[float]]]) -> List[List[float]]:
    """Load a 2D float matrix from a JSON file, plain text matrix or iterable.

    ``source`` can be a path to a JSON file or a simple whitespace separated
    text file. It can also directly be an iterable of iterables of numbers.
    Returns a list of list of floats.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        text = path.read_text()
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            data = [
                [float(v) for v in line.replace(",", " ").split()]
                for line in lines
            ]
    else:
        data = [list(row) for row in source]
    return [[float(v) for v in row] for row in data]
