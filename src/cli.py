"""Typer-based CLI interface for essay-ml."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from src import __version__

app = typer.Typer(
    name="essay-ml",
    help="ML-powered essay scoring and feedback system",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"essay-ml version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Essay-ML: ML-powered essay scoring and feedback system."""
    pass


@app.command()
def score(
    essay_file: Path = typer.Argument(
        ...,
        help="Path to the essay file to score.",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON format).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show detailed output.",
    ),
) -> None:
    """Score an essay and provide feedback."""
    console.print(
        Panel.fit(
            "[bold blue]Essay Scoring[/bold blue]\n" f"File: {essay_file}",
            title="essay-ml",
        )
    )
    # TODO: Implement scoring logic in Phase 2
    console.print(
        "[yellow]Scoring functionality will be implemented in Phase 2[/yellow]"
    )


@app.command()
def train(
    data_file: Path = typer.Argument(
        ...,
        help="Path to training data (CSV format).",
        exists=True,
        readable=True,
    ),
    output_model: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for trained model.",
    ),
) -> None:
    """Train a new scoring model."""
    console.print(
        Panel.fit(
            "[bold green]Model Training[/bold green]\n" f"Data: {data_file}",
            title="essay-ml",
        )
    )
    # TODO: Implement training logic in Phase 2
    console.print(
        "[yellow]Training functionality will be implemented in Phase 2[/yellow]"
    )


@app.command()
def info() -> None:
    """Show system information and configuration."""
    from src.core.config import MODEL_CONFIG, PATH_CONFIG, SCORING_CONFIG

    console.print(Panel.fit("[bold]System Information[/bold]", title="essay-ml"))
    console.print(f"Version: {__version__}")
    console.print(f"Project Root: {PATH_CONFIG.project_root}")
    console.print(f"Models Dir: {PATH_CONFIG.models_dir}")
    console.print(f"Data Dir: {PATH_CONFIG.data_dir}")
    console.print("\n[bold]Model Config:[/bold]")
    console.print(f"  Estimators: {MODEL_CONFIG.n_estimators}")
    console.print(f"  Max Depth: {MODEL_CONFIG.max_depth}")
    console.print("\n[bold]Scoring Weights:[/bold]")
    for category, weight in SCORING_CONFIG.weights.items():
        console.print(f"  {category}: {weight:.0%}")


if __name__ == "__main__":
    app()
