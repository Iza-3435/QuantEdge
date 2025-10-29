"""
Professional Table Styling for Trading Terminals
Light gray backgrounds like real trading platforms
Gray theme applied to everything
"""

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Professional color scheme
COLORS = {
    'up': 'green',
    'down': 'red',
    'neutral': 'white',
    'dim': 'bright_black'
}

# Gray theme settings
THEME = {
    'header_bg': 'on grey23',
    'row_even': 'on grey15',
    'row_odd': 'on grey11',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}


def create_professional_table(title, rows, show_header=True, expand=False):
    """
    Create professional table with light gray background like real terminals

    Args:
        title: Table title
        rows: List of [metric, value] pairs
        show_header: Whether to show the header row
        expand: Whether table should expand to fill space

    Returns:
        Panel containing the styled table
    """
    table = Table(
        show_header=show_header,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=expand,
        row_styles=[THEME['row_even'], THEME['row_odd']]
    )

    # Add columns
    table.add_column("Metric", style="white", no_wrap=True, width=18)
    table.add_column("Value", justify="right", style="white", width=15)

    # Add rows
    for row in rows:
        metric = row[0]
        value = row[1]
        table.add_row(metric, value)

    return Panel(
        table,
        title=f"[bold white]{title}[/bold white]",
        border_style=THEME['border'],
        box=box.SQUARE,
        style=THEME['panel_bg']
    )


def create_professional_grid_table(title, columns, rows, show_header=True, expand=True):
    """
    Create professional multi-column table with light gray background

    Args:
        title: Table title
        columns: List of column headers
        rows: List of row data (each row is a list matching column count)
        show_header: Whether to show header row
        expand: Whether table should expand to fill space

    Returns:
        Panel containing the styled table
    """
    table = Table(
        show_header=show_header,
        header_style=f"bold white {THEME['header_bg']}",
        border_style=THEME['border'],
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
        expand=expand,
        row_styles=[THEME['row_even'], THEME['row_odd']]
    )

    # Add columns
    for i, col in enumerate(columns):
        justify = "right" if i > 0 else "left"
        table.add_column(col, style="white", justify=justify)

    # Add rows
    for row in rows:
        table.add_row(*row)

    return Panel(
        table,
        title=f"[bold white]{title}[/bold white]",
        border_style=THEME['border'],
        box=box.SQUARE,
        style=THEME['panel_bg']
    )


def create_professional_header(symbol, name, sector, price, change, change_pct):
    """Create professional header with gray theme"""
    arrow = '▲' if change_pct >= 0 else '▼'
    price_color = COLORS['up'] if change_pct >= 0 else COLORS['down']

    from datetime import datetime
    text = Text()
    text.append(f"{symbol}", style="bold white")
    text.append(" │ ", style="bright_black")
    text.append(f"{name}", style="white")
    text.append(" │ ", style="bright_black")
    text.append(f"{sector}", style="bright_black")
    text.append("\n")

    text.append(f"${price:.2f}", style=f"bold {price_color}")
    text.append(f"  {arrow} ", style=price_color)
    text.append(f"{abs(change):+.2f} ", style=price_color)
    text.append(f"({change_pct:+.2f}%)", style=price_color)
    text.append("\n")

    text.append(datetime.now().strftime('%Y-%m-%d %I:%M:%S %p ET'), style="bright_black")

    return Panel(text, border_style=THEME['border'], box=box.SQUARE, padding=(1, 2), style=THEME['panel_bg'])


def create_professional_footer(text_content):
    """Create professional footer with gray theme"""
    from datetime import datetime
    if text_content is None:
        text_content = f"Professional Research Terminal • {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}"

    return Panel(
        f"[bright_black]{text_content}[/bright_black]",
        border_style=THEME['border'],
        box=box.SQUARE,
        style=THEME['panel_bg']
    )


def format_return(val):
    """Format return value with color"""
    color = COLORS['up'] if val >= 0 else COLORS['down']
    return f"[{color}]{val:+.2f}%[/]"


def format_price_change(change, change_pct):
    """Format price change with color"""
    color = COLORS['up'] if change >= 0 else COLORS['down']
    return f"[{color}]{change:+.2f} ({change_pct:+.2f}%)[/]"


def format_value_with_color(value, threshold=0, reverse=False):
    """Format value with color based on threshold"""
    if reverse:
        color = COLORS['down'] if value > threshold else COLORS['up']
    else:
        color = COLORS['up'] if value > threshold else COLORS['down']
    return f"[{color}]{value}[/]"
