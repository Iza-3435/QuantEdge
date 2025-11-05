# TERMINAL VISUALIZATION ENHANCEMENTS

## Problem: Current ASCII Charts Look Basic

Your current sparklines: `▁▂▃▄▅▆▇█` are functional but limited.

## Solution: Professional Terminal Graphics (All FREE)

---

## 1. PLOTEXT - Matplotlib for Terminal
**Cost:** FREE
**Quality:** ⭐⭐⭐⭐⭐ (Best terminal plotting library)

### Installation
```bash
pip install plotext
```

### Example: Candlestick Charts in Terminal

```python
# Add to src/terminal_charts.py
import plotext as plt
import pandas as pd

class TerminalCharts:
    """Professional terminal charts using plotext"""

    @staticmethod
    def candlestick_chart(df, ticker, width=120, height=25):
        """
        Beautiful candlestick chart directly in terminal
        Much better than ASCII sparklines!
        """
        plt.clear_figure()

        # Extract OHLC
        dates = df.index.strftime('%Y-%m-%d').tolist()
        opens = df['Open'].tolist()
        highs = df['High'].tolist()
        lows = df['Low'].tolist()
        closes = df['Close'].tolist()

        # Create candlestick
        plt.candlestick(dates, opens, closes, highs, lows)

        plt.title(f"{ticker} Price Chart")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.theme('dark')  # Bloomberg-style dark theme
        plt.plotsize(width, height)

        plt.show()

    @staticmethod
    def line_chart_with_indicators(df, ticker):
        """Price with SMA overlays"""
        plt.clear_figure()

        dates = df.index.strftime('%Y-%m-%d').tolist()

        # Plot price
        plt.plot(dates, df['Close'].tolist(),
                label='Price', color='cyan')

        # Plot moving averages
        if 'SMA_20' in df.columns:
            plt.plot(dates, df['SMA_20'].tolist(),
                    label='SMA 20', color='yellow')

        if 'SMA_50' in df.columns:
            plt.plot(dates, df['SMA_50'].tolist(),
                    label='SMA 50', color='magenta')

        plt.title(f"{ticker} Technical Analysis")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.theme('dark')
        plt.plotsize(120, 25)

        plt.show()

    @staticmethod
    def volume_chart(df):
        """Volume bars in terminal"""
        plt.clear_figure()

        dates = df.index.strftime('%Y-%m-%d').tolist()
        volumes = df['Volume'].tolist()

        # Color bars: green if close > open, red otherwise
        colors = ['green+' if df['Close'].iloc[i] > df['Open'].iloc[i]
                 else 'red+' for i in range(len(df))]

        plt.bar(dates, volumes, color=colors, orientation='v')

        plt.title("Volume Analysis")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.theme('dark')
        plt.plotsize(120, 15)

        plt.show()

    @staticmethod
    def rsi_chart(df):
        """RSI indicator in terminal"""
        plt.clear_figure()

        dates = df.index.strftime('%Y-%m-%d').tolist()
        rsi = df['RSI'].tolist()

        plt.plot(dates, rsi, color='cyan', label='RSI')

        # Add overbought/oversold lines
        plt.hline(70, color='red', label='Overbought')
        plt.hline(30, color='green', label='Oversold')

        plt.title("RSI (Relative Strength Index)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.ylim(0, 100)
        plt.theme('dark')
        plt.plotsize(120, 15)

        plt.show()

    @staticmethod
    def correlation_heatmap(corr_matrix, tickers):
        """Correlation heatmap in terminal"""
        plt.clear_figure()

        # Create matrix plot
        plt.matrix_plot(corr_matrix.values.tolist())
        plt.title("Stock Correlation Matrix")
        plt.plotsize(80, 40)

        plt.show()

        # Also print numeric table
        return corr_matrix

    @staticmethod
    def scatter_plot(returns, volatility, labels):
        """Risk-Return scatter (efficient frontier)"""
        plt.clear_figure()

        plt.scatter(volatility, returns,
                   color='cyan', marker='●')

        plt.title("Risk-Return Profile")
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Expected Return")
        plt.theme('dark')
        plt.plotsize(100, 30)

        plt.show()

    @staticmethod
    def histogram(data, title, bins=50):
        """Distribution histogram"""
        plt.clear_figure()

        plt.hist(data, bins=bins, color='cyan')
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.theme('dark')
        plt.plotsize(100, 25)

        plt.show()

    @staticmethod
    def multi_line_comparison(df_dict):
        """Compare multiple stocks on one chart"""
        plt.clear_figure()

        colors = ['cyan', 'yellow', 'magenta', 'green', 'red', 'blue']

        for idx, (ticker, df) in enumerate(df_dict.items()):
            dates = df.index.strftime('%Y-%m-%d').tolist()
            # Normalize to percentage change
            normalized = (df['Close'] / df['Close'].iloc[0] - 1) * 100
            plt.plot(dates, normalized.tolist(),
                    label=ticker,
                    color=colors[idx % len(colors)])

        plt.title("Stock Comparison (% Change)")
        plt.xlabel("Date")
        plt.ylabel("% Change")
        plt.theme('dark')
        plt.plotsize(120, 30)

        plt.show()
```

### Usage in Your Apps

```python
# In apps/INSTITUTIONAL_TERMINAL.py, replace sparklines:

from src.terminal_charts import TerminalCharts

# After showing data table:
console.print("\n[cyan]View Chart?[/cyan] [1] Candlestick [2] Price+Indicators [3] Volume [4] Skip")
choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="4")

if choice == "1":
    TerminalCharts.candlestick_chart(df, ticker, width=140, height=30)
elif choice == "2":
    TerminalCharts.line_chart_with_indicators(df, ticker)
elif choice == "3":
    TerminalCharts.volume_chart(df)
```

---

## 2. ENHANCED RICH VISUALIZATIONS

### Better Tables with Gradients

```python
# Add to src/ui_enhancements.py
from rich.table import Table
from rich.text import Text
import colorsys

class EnhancedRichUI:
    """Enhanced Rich library visualizations"""

    @staticmethod
    def create_gradient_table(data, title):
        """Table with color gradients based on values"""
        table = Table(title=title, box=box.ROUNDED,
                     header_style="bold white on grey23",
                     border_style="grey35")

        # Add columns
        for col in data.columns:
            table.add_column(col, justify="right")

        # Add rows with color gradients
        for _, row in data.iterrows():
            styled_row = []
            for col in data.columns:
                value = row[col]

                # Apply color based on value
                if isinstance(value, (int, float)):
                    color = EnhancedRichUI.value_to_color(value,
                                                          data[col].min(),
                                                          data[col].max())
                    styled_row.append(f"[{color}]{value:.2f}[/{color}]")
                else:
                    styled_row.append(str(value))

            table.add_row(*styled_row)

        return table

    @staticmethod
    def value_to_color(value, min_val, max_val):
        """Map value to color gradient (red -> yellow -> green)"""
        if max_val == min_val:
            return "white"

        # Normalize to 0-1
        normalized = (value - min_val) / (max_val - min_val)

        if normalized < 0.5:
            # Red to yellow
            return f"rgb({255},{int(normalized*2*255)},0)"
        else:
            # Yellow to green
            return f"rgb({int(255-((normalized-0.5)*2*255))},255,0)"

    @staticmethod
    def create_sparkline_advanced(data, width=50):
        """Enhanced sparkline with colors"""
        if len(data) == 0:
            return "No data"

        min_val = min(data)
        max_val = max(data)

        if max_val == min_val:
            return "─" * width

        # More granular characters
        chars = ' ▁▂▃▄▅▆▇█'

        # Resample data to fit width
        if len(data) > width:
            step = len(data) / width
            data = [data[int(i * step)] for i in range(width)]

        sparkline = ""
        for i, val in enumerate(data):
            normalized = (val - min_val) / (max_val - min_val)
            char_idx = int(normalized * (len(chars) - 1))
            char = chars[char_idx]

            # Color based on trend
            if i > 0:
                if data[i] > data[i-1]:
                    sparkline += f"[green]{char}[/green]"
                elif data[i] < data[i-1]:
                    sparkline += f"[red]{char}[/red]"
                else:
                    sparkline += f"[yellow]{char}[/yellow]"
            else:
                sparkline += char

        return sparkline

    @staticmethod
    def create_horizontal_bar_chart(labels, values, title, max_width=60):
        """Horizontal bar chart in terminal"""
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan", width=20)
        table.add_column("Bar", width=max_width)
        table.add_column("Value", justify="right", style="yellow")

        max_val = max(values)

        for label, value in zip(labels, values):
            # Create bar
            bar_length = int((value / max_val) * max_width)
            bar = "█" * bar_length

            # Color based on value
            if value > 0:
                bar = f"[green]{bar}[/green]"
            elif value < 0:
                bar = f"[red]{bar}[/red]"
            else:
                bar = f"[grey]{bar}[/grey]"

            table.add_row(label, bar, f"{value:.2f}")

        return Panel(table, title=title, border_style="grey35")

    @staticmethod
    def create_gauge(value, min_val, max_val, title, width=50):
        """Gauge chart (like speedometer)"""
        normalized = (value - min_val) / (max_val - min_val)
        position = int(normalized * width)

        # Create gauge bar
        bar = "─" * width
        bar_list = list(bar)

        # Add marker
        bar_list[position] = "┃"

        # Color sections
        third = width // 3
        gauge = (
            f"[red]{''.join(bar_list[:third])}[/red]"
            f"[yellow]{''.join(bar_list[third:2*third])}[/yellow]"
            f"[green]{''.join(bar_list[2*third:])}[/green]"
        )

        # Add labels
        result = f"{title}\n"
        result += f"{min_val:.1f} {gauge} {max_val:.1f}\n"
        result += f"Current: {value:.2f}"

        return Panel(result, border_style="grey35")

    @staticmethod
    def create_box_plot_terminal(data, labels):
        """Box plot visualization in terminal"""
        from rich.table import Table

        table = Table(title="Distribution Analysis", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")

        for label in labels:
            table.add_column(label, justify="right")

        stats = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean', 'Std']

        for stat in stats:
            row = [stat]
            for series in data:
                if stat == 'Min':
                    val = series.min()
                elif stat == 'Q1':
                    val = series.quantile(0.25)
                elif stat == 'Median':
                    val = series.median()
                elif stat == 'Q3':
                    val = series.quantile(0.75)
                elif stat == 'Max':
                    val = series.max()
                elif stat == 'Mean':
                    val = series.mean()
                elif stat == 'Std':
                    val = series.std()

                row.append(f"{val:.2f}")

            table.add_row(*row)

        return table
```

---

## 3. UNICODE BLOCKS & BRAILLE PATTERNS

### High-Resolution Terminal Graphics

```python
# Add to src/terminal_charts.py

class UnicodeCharts:
    """High-res charts using Unicode"""

    # Braille patterns for 2x4 pixel blocks
    BRAILLE = [
        ' ', '⠁', '⠂', '⠃', '⠄', '⠅', '⠆', '⠇',
        '⠈', '⠉', '⠊', '⠋', '⠌', '⠍', '⠎', '⠏',
        # ... (full braille unicode range)
    ]

    @staticmethod
    def create_braille_chart(data, width=100, height=30):
        """Ultra high-res chart using braille characters"""
        # Each braille char = 2x4 pixels
        # Gives 8x resolution of normal ASCII
        import numpy as np

        # Create pixel grid
        grid = np.zeros((height * 4, width * 2))

        # Map data to grid
        # ... (implementation)

        # Convert to braille
        # ... (implementation)

        return braille_output

    @staticmethod
    def block_chart(data, width=60):
        """Smooth chart using block characters"""
        blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

        min_val = min(data)
        max_val = max(data)

        chart = ""
        for val in data:
            normalized = (val - min_val) / (max_val - min_val)
            idx = int(normalized * (len(blocks) - 1))
            chart += blocks[idx]

        return chart
```

---

## 4. LAYOUT IMPROVEMENTS

### Multi-Panel Dashboard Layout

```python
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console

def create_bloomberg_style_layout():
    """Bloomberg Terminal-style multi-panel layout"""

    layout = Layout()

    # Split screen into sections
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )

    # Split main into columns
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1)
    )

    # Split left into top/bottom
    layout["left"].split_column(
        Layout(name="chart", ratio=2),
        Layout(name="indicators", ratio=1)
    )

    # Populate panels
    layout["header"].update(Panel("[bold white]QUANTEDGE INSTITUTIONAL TERMINAL[/bold white]",
                                  style="on grey23"))

    layout["chart"].update(Panel("Price Chart Here", title="Price", border_style="cyan"))
    layout["indicators"].update(Panel("Technical Indicators", title="Indicators"))
    layout["right"].update(Panel("News Feed", title="News", border_style="yellow"))

    layout["footer"].update(Panel("Status Bar", style="on grey15"))

    return layout

# Usage
console = Console()
layout = create_bloomberg_style_layout()
console.print(layout)
```

---

## 5. LIVE UPDATING DISPLAYS

### Real-time Terminal Updates

```python
from rich.live import Live
from rich.table import Table
import time

def create_live_dashboard(symbols):
    """Live updating dashboard"""

    with Live(generate_table(symbols), refresh_per_second=1) as live:
        while True:
            time.sleep(1)
            live.update(generate_table(symbols))

def generate_table(symbols):
    """Generate updated table"""
    table = Table(title="Live Market Data")
    table.add_column("Symbol")
    table.add_column("Price")
    table.add_column("Change")
    table.add_column("Volume")

    for symbol in symbols:
        # Fetch latest data
        price, change, volume = get_latest_quote(symbol)

        # Color code
        change_color = "green" if change > 0 else "red"

        table.add_row(
            symbol,
            f"${price:.2f}",
            f"[{change_color}]{change:+.2f}%[/{change_color}]",
            f"{volume:,.0f}"
        )

    return table
```

---

## IMPLEMENTATION PRIORITY

### Week 1: Add Plotext
```bash
pip install plotext
```
- Replace ASCII sparklines with real charts
- Add candlestick, line, and volume charts
- **Impact: Huge visual upgrade**

### Week 2: Enhanced Rich UI
- Gradient tables
- Horizontal bar charts
- Gauges for scores/metrics
- **Impact: Professional appearance**

### Week 3: Multi-Panel Layouts
- Bloomberg-style dashboard layout
- Split screen views
- **Impact: Information density**

### Week 4: Live Updates
- Real-time refreshing tables
- Auto-updating charts
- **Impact: Dynamic feel**

---

## BEFORE/AFTER COMPARISON

### BEFORE (Current)
```
Price: $150.25 (+2.5%)
Chart: ▁▂▃▄▅▆▇█▇▆▅▄
```

### AFTER (With Plotext)
```
    AAPL - Price Chart
    │
160 │                        ╭──╮
155 │                   ╭────╯  ╰─╮
150 │              ╭────╯         ╰─╮
145 │         ╭────╯                ╰─
140 │    ╭────╯
    └─────────────────────────────────
    Jan   Feb   Mar   Apr   May   Jun
```

Much more professional and informative!

---

## TOOLS SUMMARY

| Tool | Purpose | Cost | Quality |
|------|---------|------|---------|
| **plotext** | Terminal matplotlib | FREE | ⭐⭐⭐⭐⭐ |
| **Rich** | UI framework (you have it!) | FREE | ⭐⭐⭐⭐⭐ |
| **Unicode blocks** | High-res charts | FREE | ⭐⭐⭐⭐ |
| **Braille patterns** | Ultra high-res | FREE | ⭐⭐⭐⭐ |

**Total cost: $0** for massive visual upgrade!
