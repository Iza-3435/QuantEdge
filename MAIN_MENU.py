#!/usr/bin/env python3
"""
AI Market Intelligence - Main Menu
Production-grade terminal menu following big tech standards.
"""
import os
import sys
import subprocess
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

THEME = {
    'header_bg': 'on grey23',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass(frozen=True)
class Feature:
    """Immutable feature configuration."""
    id: int
    name: str
    file: str
    desc: str
    args: bool

    def __post_init__(self) -> None:
        if not Path(self.file).exists():
            logger.warning(f"Feature file not found: {self.file}")


FEATURES: List[Feature] = [
    Feature(1, 'Institutional Terminal (Ultra-Dense)', 'apps/INSTITUTIONAL_TERMINAL.py',
            'Bloomberg-level AI analysis (LSTM, Sentiment, Risk, Options, Fundamentals)', True),
    Feature(2, 'Market Overview', 'apps/MARKET_OVERVIEW.py',
            'Major markets & news (clickable)', False),
    Feature(3, 'Portfolio Manager Pro', 'apps/PORTFOLIO_PRO.py',
            'Track & manage portfolio', False),
    Feature(4, 'Watchlist Pro', 'apps/WATCHLIST_PRO.py',
            'Monitor watchlist stocks', False),
    Feature(5, 'Stock Screener', 'apps/STOCK_SCREENER.py',
            'Find stocks by criteria', False),
    Feature(6, 'AI Stock Picker', 'apps/AI_STOCK_PICKER.py',
            'AI stock recommendations', False),
    Feature(7, 'Historical Context', 'apps/HISTORICAL_CONTEXT.py',
            'Historical patterns & trends', True),
    Feature(8, 'Investment Projection', 'apps/INVESTMENT_PROJECTION.py',
            'Future price projections', True),
    Feature(9, 'Comparison Matrix', 'apps/COMPARISON_MATRIX.py',
            'Compare stocks (25+ metrics)', True),
    Feature(10, 'Earnings Calendar', 'apps/EARNINGS_CALENDAR.py',
            'Upcoming earnings & surprises', False),
    Feature(11, 'Dividend Tracker', 'apps/DIVIDEND_TRACKER.py',
            'Dividend yields & calendar', False),
    Feature(12, 'Sector Analyzer', 'apps/SECTOR_ANALYZER.py',
            'Sector performance & leaders', False),
    Feature(13, 'Technical Screener', 'apps/TECHNICAL_SCREENER.py',
            'Find bullish technical setups', False),
    Feature(14, 'Correlation Matrix', 'apps/CORRELATION_MATRIX.py',
            'Find correlations & hedges', True),
    Feature(15, 'Backtesting Engine', 'apps/BACKTESTING_ENGINE.py',
            'Test trading strategies with walk-forward validation', True),
]


def render_header() -> None:
    """Render the application header."""
    from datetime import datetime
    header = Text()
    header.append("AI MARKET INTELLIGENCE SYSTEM\n", style="bold white")
    header.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="bright_black")
    console.print(Panel(header, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()


def render_menu() -> None:
    """Render the feature menu table."""
    table = Table(show_header=True, header_style="bold white", border_style=THEME['border'],
                  box=box.SQUARE, style=THEME['panel_bg'])

    table.add_column("#", justify="right", style="white", width=4)
    table.add_column("Feature", justify="left", style="white", width=45)
    table.add_column("Description", justify="left", style="white")

    for f in FEATURES:
        table.add_row(str(f.id), f.name, f.desc)

    console.print(Panel(table, title="[bold white]AVAILABLE FEATURES[/bold white]",
                       border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()


def render_footer() -> None:
    """Render the menu footer."""
    footer = Text()
    footer.append("Enter feature number", style="white")
    footer.append(" │ ", style="bright_black")
    footer.append("Type 'q' to quit", style="white")
    console.print(Panel(footer, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
    console.print()


def get_stock_symbols() -> List[str]:
    """
    Get and validate stock symbols from user input.

    Returns:
        List of validated uppercase stock symbols
    """
    symbol_input = Prompt.ask(
        "[white]Enter stock symbol(s) (space-separated)[/white]",
        default="AAPL"
    )

    symbols = [s.strip().upper() for s in symbol_input.split() if s.strip()]

    if not symbols:
        console.print("[red]Error: No symbols provided[/red]")
        logger.warning(f"Invalid symbols: {symbol_input}")
        return get_stock_symbols()

    return symbols


def execute_feature(feature: Feature) -> None:
    """
    Execute a feature application.

    Args:
        feature: Feature to execute
    """
    console.clear()

    if not Path(feature.file).exists():
        console.print(f"\n[red]Error: Feature file not found: {feature.file}[/red]\n")
        logger.error(f"Missing feature file: {feature.file}")
        input("Press Enter to continue...")
        return

    args: List[str] = []
    if feature.args:
        text = Text(f"Running: {feature.name}", style="white")
        console.print(Panel(text, title="[bold white]Feature Launcher[/bold white]",
                           border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
        console.print()
        symbols = get_stock_symbols()

        # Handle different argument formats
        if 'advanced_dashboard.py' in feature.file:
            # advanced_dashboard requires --symbols flag
            args = ['--symbols'] + symbols
        else:
            # Other scripts accept positional arguments
            args = symbols

    try:
        cmd = [sys.executable, feature.file] + args
        logger.info(f"Executing: {' '.join(cmd)}")
        console.print(f"\n[bright_black]Launching: {' '.join(cmd)}[/bright_black]\n")

        result = subprocess.run(cmd, check=True)

        if result.returncode != 0:
            logger.warning(f"Feature exited with code {result.returncode}")

    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Error: Feature execution failed: {e}[/red]\n")
        logger.error(f"Subprocess error: {e}", exc_info=True)

    except FileNotFoundError as e:
        console.print(f"\n[red]Error: Python interpreter not found: {e}[/red]\n")
        logger.error(f"Python not found: {e}", exc_info=True)

    except Exception as e:
        console.print(f"\n[red]Error: Unexpected error: {e}[/red]\n")
        logger.error(f"Unexpected error: {e}", exc_info=True)

    console.print()
    input("Press Enter to return to menu...")


def validate_choice(choice: str) -> Optional[Feature]:
    """
    Validate user's feature choice.

    Args:
        choice: User input string

    Returns:
        Feature if valid, None otherwise
    """
    if choice.lower() in ['q', 'quit', 'exit']:
        return None

    try:
        feature_id = int(choice)
        if 1 <= feature_id <= len(FEATURES):
            return FEATURES[feature_id - 1]
        else:
            console.print(f"\n[red]Error: Please enter a number between 1 and {len(FEATURES)}[/red]")
            logger.warning(f"Invalid choice: {choice}")
            input("Press Enter to continue...")
            return None
    except ValueError:
        console.print(f"\n[red]Error: Invalid input. Please enter a number.[/red]")
        logger.warning(f"Invalid choice: {choice}")
        input("Press Enter to continue...")
        return None


def run_menu_loop() -> None:
    """Main menu loop."""
    while True:
        console.clear()
        render_header()
        render_menu()
        render_footer()

        choice = Prompt.ask("[white]Select feature[/white]", default="1")

        if choice.lower() in ['q', 'quit', 'exit']:
            console.print("\n[white]Goodbye![/white]\n")
            logger.info("Application exited by user")
            break

        feature = validate_choice(choice)
        if feature:
            execute_feature(feature)


def main() -> None:
    """Main entry point."""
    try:
        Path('logs').mkdir(exist_ok=True)
        logger.info("Application started")
        run_menu_loop()

    except KeyboardInterrupt:
        console.print("\n\n[white]Goodbye![/white]\n")
        logger.info("Application interrupted by user")
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]\n")
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
