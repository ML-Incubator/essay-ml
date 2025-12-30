"""File I/O utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def read_essay(file_path: Path) -> str:
    """Read an essay from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_results(results: Dict[str, Any], output_path: Optional[Path] = None) -> None:
    """Write scoring results to a file or stdout."""
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        result: Dict[str, Any] = json.load(f)
        return result


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
