"""
Reusable UI components for all applications.
Follows DRY principle and ensures consistent styling.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .config import THEME, CONFIG


def create_header(
    title: str,
    subtitle: Optional[str] = None,
    show_timestamp: bool = True
) -> Panel:
    """
    Create a consistent professional header panel.

    Args:
        title: Main header text
        subtitle: Optional subtitle text
        show_timestamp: Whether to show current time

    Returns:
        Formatted header panel
    """
    text = Text()
    text.append(f"{title}\n", style="bold white")

    if subtitle:
        text.append(f"{subtitle}\n", style="white")

    if show_timestamp:
        timestamp = datetime.now().strftime(CONFIG.datetime_format)
        text.append(timestamp, style="bright_black")

    return Panel(
        text,
        border_style=THEME.border,
        box=box.SQUARE,
        padding=(1, 2),
        style=THEME.panel_bg
    )


def create_table(
    columns: List[Dict[str, Any]],
    rows: List[List[str]],
    title: Optional[str] = None,
    show_header: bool = True
) -> Table:
    """
    Create a consistently styled table.

    Args:
        columns: List of column dicts with 'name', 'style', 'width', 'justify'
        rows: List of row data
        title: Optional table title
        show_header: Whether to show column headers

    Returns:
        Formatted table
    """
    table = Table(
        show_header=show_header,
        header_style=f"bold white {THEME.header_bg}",
        border_style=THEME.border,
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=True,
        row_styles=[THEME.row_even, THEME.row_odd],
        title=title
    )

    # Add columns
    for col in columns:
        table.add_column(
            col.get('name', ''),
            style=col.get('style', 'white'),
            width=col.get('width'),
            justify=col.get('justify', 'left')
        )

    # Add rows
    for row in rows:
        table.add_row(*row)

    return table


def create_panel(
    content: Any,
    title: Optional[str] = None,
    subtitle: Optional[str] = None
) -> Panel:
    """
    Create a consistently styled panel.

    Args:
        content: Panel content (can be Text, Table, or string)
        title: Optional panel title
        subtitle: Optional panel subtitle

    Returns:
        Formatted panel
    """
    return Panel(
        content,
        title=f"[bold white]{title}[/bold white]" if title else None,
        subtitle=f"[bright_black]{subtitle}[/bright_black]" if subtitle else None,
        border_style=THEME.border,
        box=box.SQUARE,
        style=THEME.panel_bg
    )


def format_change(value: float, show_sign: bool = True) -> tuple[str, str]:
    """
    Format a numeric change value with color coding.

    Args:
        value: Numeric value to format
        show_sign: Whether to show +/- sign

    Returns:
        Tuple of (formatted_text, color)
    """
    from .config import COLORS

    if value > 0:
        sign = "+" if show_sign else ""
        return f"{sign}{value:.2f}", COLORS.positive
    elif value < 0:
        return f"{value:.2f}", COLORS.negative
    else:
        return f"{value:.2f}", COLORS.neutral


def format_percentage(value: float, decimals: int = 2) -> tuple[str, str]:
    """
    Format a percentage value with color coding.

    Args:
        value: Percentage value (e.g., 5.2 for 5.2%)
        decimals: Number of decimal places

    Returns:
        Tuple of (formatted_text, color)
    """
    from .config import COLORS

    if value > 0:
        return f"+{value:.{decimals}f}%", COLORS.positive
    elif value < 0:
        return f"{value:.{decimals}f}%", COLORS.negative
    else:
        return f"{value:.{decimals}f}%", COLORS.neutral


def create_footer(items: List[str]) -> Panel:
    """
    Create a consistent footer with multiple items.

    Args:
        items: List of footer text items

    Returns:
        Formatted footer panel
    """
    text = Text()
    for i, item in enumerate(items):
        if i > 0:
            text.append(" │ ", style="bright_black")
        text.append(item, style="white" if i == 0 else "bright_black")

    return Panel(
        text,
        border_style=THEME.border,
        box=box.SQUARE,
        style=THEME.panel_bg
    )


def show_error(console: Console, message: str, title: str = "Error") -> None:
    """
    Display a consistent error message.

    Args:
        console: Rich console instance
        message: Error message to display
        title: Error panel title
    """
    console.print()
    console.print(
        Panel(
            f"[red]{message}[/red]",
            title=f"[bold red]{title}[/bold red]",
            border_style="red",
            box=box.SQUARE
        )
    )
    console.print()


def show_success(console: Console, message: str, title: str = "Success") -> None:
    """
    Display a consistent success message.

    Args:
        console: Rich console instance
        message: Success message to display
        title: Success panel title
    """
    console.print()
    console.print(
        Panel(
            f"[green]{message}[/green]",
            title=f"[bold green]{title}[/bold green]",
            border_style="green",
            box=box.SQUARE
        )
    )
    console.print()


def show_warning(console: Console, message: str, title: str = "Warning") -> None:
    """
    Display a consistent warning message.

    Args:
        console: Rich console instance
        message: Warning message to display
        title: Warning panel title
    """
    console.print()
    console.print(
        Panel(
            f"[yellow]{message}[/yellow]",
            title=f"[bold yellow]{title}[/bold yellow]",
            border_style="yellow",
            box=box.SQUARE
        )
    )
    console.print()
