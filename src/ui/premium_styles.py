"""
PREMIUM UI STYLES
Professional colored block headers and styling for all tools
"""

from rich.text import Text
from rich.panel import Panel
from rich import box
from datetime import datetime


class PremiumStyles:
    """
    Premium UI components with colored block headers
    """

    @staticmethod
    def create_header(title: str, subtitle: str = None, color_scheme: str = "blue"):
        """
        Create premium header with colored block title

        Args:
            title: Main title text
            subtitle: Optional subtitle
            color_scheme: Color scheme (blue, green, cyan, magenta, yellow)

        Returns:
            Rich Panel with formatted header
        """
        # Color scheme mapping
        schemes = {
            'blue': {
                'block': 'bold white on blue',
                'accent': 'bright_cyan',
                'border': 'bright_cyan'
            },
            'green': {
                'block': 'bold white on green',
                'accent': 'bright_green',
                'border': 'bright_green'
            },
            'cyan': {
                'block': 'bold white on cyan',
                'accent': 'bright_cyan',
                'border': 'cyan'
            },
            'magenta': {
                'block': 'bold white on magenta',
                'accent': 'bright_magenta',
                'border': 'magenta'
            },
            'yellow': {
                'block': 'bold black on yellow',
                'accent': 'yellow',
                'border': 'yellow'
            },
            'red': {
                'block': 'bold white on red',
                'accent': 'bright_red',
                'border': 'red'
            }
        }

        scheme = schemes.get(color_scheme, schemes['blue'])
        now = datetime.now()

        header = Text()
        header.append("═" * 80, style=scheme['accent'])
        header.append("\n")
        header.append("                    ", style=scheme['accent'])
        header.append(f" {title} ", style=scheme['block'])
        header.append("\n", style=scheme['accent'])
        header.append("═" * 80, style=scheme['accent'])

        if subtitle:
            header.append("\n\n")
            header.append("  ", style="dim")
            header.append(subtitle, style=f"bold {scheme['accent']}")

        header.append("\n\n")
        header.append("  Updated: ", style="dim")
        header.append(now.strftime('%A, %B %d, %Y at %I:%M:%S %p ET'), style=scheme['accent'])

        return Panel(header, box=box.DOUBLE, border_style=scheme['border'], padding=(1, 2))

    @staticmethod
    def create_section_header(title: str, color: str = "cyan", emoji: str = ""):
        """
        Create colored section header

        Args:
            title: Section title
            color: Header color
            emoji: Optional emoji

        Returns:
            Text with formatted section header
        """
        text = Text()
        text.append(f"\n{emoji} ", style=f"bold {color}")
        text.append(title.upper(), style=f"bold {color}")
        text.append("\n")
        return text

    @staticmethod
    def create_table_title(title: str, live: bool = False, color_scheme: str = "blue"):
        """
        Create colored table title

        Args:
            title: Table title
            live: Show LIVE indicator
            color_scheme: Color scheme

        Returns:
            Formatted title string for Rich Table
        """
        schemes = {
            'blue': 'bold white on blue',
            'green': 'bold white on green',
            'cyan': 'bold white on cyan',
            'magenta': 'bold white on magenta',
            'yellow': 'bold black on yellow',
            'red': 'bold white on red'
        }

        style = schemes.get(color_scheme, schemes['blue'])
        live_indicator = " 🟢 LIVE" if live else ""

        return f"[{style}] {title}{live_indicator} [/{style}]"

    @staticmethod
    def format_price(price: float, change_pct: float, show_dollar: bool = True) -> str:
        """
        Format price with colors based on change

        Args:
            price: Price value
            change_pct: Percentage change
            show_dollar: Include $ sign

        Returns:
            Formatted price string with color
        """
        # Color based on intensity
        if change_pct > 2:
            color = 'bold bright_green'
        elif change_pct > 0:
            color = 'green'
        elif change_pct < -2:
            color = 'bold bright_red'
        elif change_pct < 0:
            color = 'red'
        else:
            color = 'yellow'

        # Arrow
        if change_pct > 0:
            arrow = '▲'
        elif change_pct < 0:
            arrow = '▼'
        else:
            arrow = '━'

        dollar = '$' if show_dollar else ''
        return f"[{color}]{arrow} {dollar}{abs(price):,.2f} ({change_pct:+.2f}%)[/{color}]"

    @staticmethod
    def format_metric(label: str, value: str, good: bool = None) -> str:
        """
        Format metric with optional color coding

        Args:
            label: Metric label
            value: Metric value
            good: True=green, False=red, None=white

        Returns:
            Formatted metric string
        """
        if good is True:
            color = 'bright_green'
        elif good is False:
            color = 'bright_red'
        else:
            color = 'white'

        return f"[bold white]{label}:[/bold white] [{color}]{value}[/{color}]"

    @staticmethod
    def create_footer(message: str = None, color: str = "cyan") -> Panel:
        """
        Create premium footer

        Args:
            message: Optional custom message
            color: Border color

        Returns:
            Rich Panel footer
        """
        default_msg = "Data provided by Yahoo Finance | Press Ctrl+C to exit"
        content = message or default_msg

        return Panel(
            f"[dim]{content}[/dim]",
            box=box.ROUNDED,
            border_style=color
        )


# Quick access functions
def header(title: str, subtitle: str = None, color: str = "blue"):
    """Quick header creation"""
    return PremiumStyles.create_header(title, subtitle, color)


def table_title(title: str, live: bool = False, color: str = "blue"):
    """Quick table title creation"""
    return PremiumStyles.create_table_title(title, live, color)


def section(title: str, emoji: str = "", color: str = "cyan"):
    """Quick section header creation"""
    return PremiumStyles.create_section_header(title, color, emoji)


def footer(message: str = None, color: str = "cyan"):
    """Quick footer creation"""
    return PremiumStyles.create_footer(message, color)


# Example color schemes for different tools
TOOL_COLORS = {
    'MASTER_DASHBOARD': 'blue',
    'MARKET_OVERVIEW': 'cyan',
    'PORTFOLIO_PRO': 'green',
    'WATCHLIST_PRO': 'magenta',
    'STOCK_SCREENER': 'blue',
    'STOCK_COMPARISON': 'cyan',
    'ULTIMATE_TERMINAL_PRO': 'blue',
    'MASTER_STOCK_ANALYZER': 'magenta',
    'ALERTS_SYSTEM': 'red'
}
