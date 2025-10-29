"""
RESEARCH NOTES & BOOKMARKS SYSTEM
Save investment thesis, bookmark findings, track research progress

Features:
- Add research notes for any stock
- Bookmark important metrics/news
- Tag concerns and opportunities
- View all research in one place
- Export notes to markdown
- Search through notes
"""

import sqlite3
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
import json

console = Console()

DB_PATH = 'investment_platform.db'

COLORS = {
    'up': 'green',
    'down': 'red',
    'neutral': 'white',
    'dim': 'bright_black'
}

THEME = {
    'header_bg': 'on grey23',
    'row_even': 'on grey15',
    'row_odd': 'on grey11',
    'border': 'grey35',
    'panel_bg': 'on grey11'
}


def init_research_db():
    """Initialize research notes database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Research notes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            note_type TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Bookmarks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_bookmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bookmark_type TEXT NOT NULL,
            title TEXT NOT NULL,
            value TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_note(symbol, note_type, content, tags=None):
    """Add a research note"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO research_notes (symbol, note_type, content, tags)
        VALUES (?, ?, ?, ?)
    """, (symbol.upper(), note_type, content, tags))

    conn.commit()
    note_id = cursor.lastrowid
    conn.close()

    return note_id


def add_bookmark(symbol, bookmark_type, title, value=None, notes=None):
    """Add a bookmark"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO research_bookmarks (symbol, bookmark_type, title, value, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (symbol.upper(), bookmark_type, title, value, notes))

    conn.commit()
    bookmark_id = cursor.lastrowid
    conn.close()

    return bookmark_id


def get_notes_for_symbol(symbol):
    """Get all notes for a symbol"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, note_type, content, tags, created_at, updated_at
        FROM research_notes
        WHERE symbol = ?
        ORDER BY updated_at DESC
    """, (symbol.upper(),))

    notes = cursor.fetchall()
    conn.close()

    return notes


def get_bookmarks_for_symbol(symbol):
    """Get all bookmarks for a symbol"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, bookmark_type, title, value, notes, created_at
        FROM research_bookmarks
        WHERE symbol = ?
        ORDER BY created_at DESC
    """, (symbol.upper(),))

    bookmarks = cursor.fetchall()
    conn.close()

    return bookmarks


def get_all_researched_stocks():
    """Get list of all stocks with research"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT symbol FROM (
            SELECT symbol FROM research_notes
            UNION
            SELECT symbol FROM research_bookmarks
        )
        ORDER BY symbol
    """)

    stocks = [row[0] for row in cursor.fetchall()]
    conn.close()

    return stocks


def delete_note(note_id):
    """Delete a note"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM research_notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()


def delete_bookmark(bookmark_id):
    """Delete a bookmark"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM research_bookmarks WHERE id = ?", (bookmark_id,))
    conn.commit()
    conn.close()


def export_notes_to_markdown(symbol):
    """Export research notes to markdown file"""
    notes = get_notes_for_symbol(symbol)
    bookmarks = get_bookmarks_for_symbol(symbol)

    if not notes and not bookmarks:
        return None

    filename = f"{symbol}_research_notes_{datetime.now().strftime('%Y%m%d')}.md"

    with open(filename, 'w') as f:
        f.write(f"# {symbol} Research Notes\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n")
        f.write("---\n\n")

        # Investment Thesis
        thesis_notes = [n for n in notes if n[1] == 'THESIS']
        if thesis_notes:
            f.write("##  Investment Thesis\n\n")
            for note in thesis_notes:
                f.write(f"{note[2]}\n\n")
                if note[3]:  # tags
                    f.write(f"*Tags: {note[3]}*\n\n")

        # Concerns
        concern_notes = [n for n in notes if n[1] == 'CONCERN']
        if concern_notes:
            f.write("##   Concerns\n\n")
            for note in concern_notes:
                f.write(f"- {note[2]}\n")
            f.write("\n")

        # Opportunities
        opp_notes = [n for n in notes if n[1] == 'OPPORTUNITY']
        if opp_notes:
            f.write("##  Opportunities\n\n")
            for note in opp_notes:
                f.write(f"- {note[2]}\n")
            f.write("\n")

        # Key Metrics
        metric_bookmarks = [b for b in bookmarks if b[1] == 'METRIC']
        if metric_bookmarks:
            f.write("##  Key Metrics to Watch\n\n")
            for bm in metric_bookmarks:
                f.write(f"- **{bm[2]}:** {bm[3]}")
                if bm[4]:
                    f.write(f" — {bm[4]}")
                f.write("\n")
            f.write("\n")

        # Bookmarked News
        news_bookmarks = [b for b in bookmarks if b[1] == 'NEWS']
        if news_bookmarks:
            f.write("## 📰 Important News\n\n")
            for bm in news_bookmarks:
                f.write(f"- {bm[2]}")
                if bm[4]:
                    f.write(f" — {bm[4]}")
                f.write(f" ({bm[5][:10]})\n")
            f.write("\n")

        # General Notes
        general_notes = [n for n in notes if n[1] == 'NOTE']
        if general_notes:
            f.write("## 📝 Research Notes\n\n")
            for note in general_notes:
                f.write(f"**{note[4][:10]}:** {note[2]}\n\n")

    return filename


def create_header():
    """Create header"""
    header = Text()
    header.append("RESEARCH NOTES & BOOKMARKS\n\n", style="bold white")
    header.append("Build Your Investment Thesis", style="white")
    header.append(" │ ", style="bright_black")
    header.append("Save Findings", style="white")
    header.append(" │ ", style="bright_black")
    header.append("Track Research Progress", style="white")

    return Panel(header, box=box.SQUARE, border_style=THEME['border'], padding=(1, 2), style=THEME['panel_bg'])


def display_stock_research(symbol):
    """Display all research for a stock"""
    console.clear()
    console.print()

    # Header
    title_text = Text()
    title_text.append(f"Research Notes: ", style="bold white")
    title_text.append(symbol, style="bold white")

    console.print(Panel(title_text, border_style=THEME['border'], style=THEME['panel_bg']))
    console.print()

    notes = get_notes_for_symbol(symbol)
    bookmarks = get_bookmarks_for_symbol(symbol)

    if not notes and not bookmarks:
        console.print(f"[yellow]No research notes found for {symbol}[/yellow]\n")
        return

    # Display notes by type
    note_types = {
        'THESIS': ('Investment Thesis', 'green'),
        'CONCERN': ('Concerns', 'red'),
        'OPPORTUNITY': ('Opportunities', 'white'),
        'NOTE': ('General Notes', 'white')
    }

    for note_type, (title, color) in note_types.items():
        type_notes = [n for n in notes if n[1] == note_type]
        if type_notes:
            console.print(Panel(title, border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))
            for note in type_notes:
                text = Text()
                text.append(f"[{note[4][:10]}] ", style="bright_black")
                text.append(note[2], style="white")
                if note[3]:
                    text.append(f"\nTags: {note[3]}", style="bright_black")
                console.print(text)
                console.print()

    # Display bookmarks
    if bookmarks:
        console.print(Panel("Bookmarks", border_style=THEME['border'], box=box.SQUARE, style=THEME['panel_bg']))

        bookmark_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style=f"bold white {THEME['header_bg']}", border_style=THEME['border'], row_styles=[THEME['row_even'], THEME['row_odd']], padding=(0, 1))
        bookmark_table.add_column("Type", style="white", width=10)
        bookmark_table.add_column("Item", style="white", width=40)
        bookmark_table.add_column("Value/Notes", style="white", width=30)

        for bm in bookmarks[:10]:
            bm_type = bm[1]
            title = bm[2]
            value = bm[3] or ""
            notes_text = bm[4] or ""
            display_value = f"{value} {notes_text}".strip()

            bookmark_table.add_row(bm_type, title, display_value)

        console.print(bookmark_table)
        console.print()


def interactive_add_note(symbol):
    """Interactive note addition"""
    console.print(f"\n[bold white]Add Research Note for {symbol}[/bold white]\n")

    # Note type
    console.print("[white]Note Type:[/white]")
    console.print("  1. Investment Thesis")
    console.print("  2. Concern")
    console.print("  3. Opportunity")
    console.print("  4. General Note")

    choice = Prompt.ask("Choose type", choices=["1", "2", "3", "4"], default="4")

    type_map = {
        "1": "THESIS",
        "2": "CONCERN",
        "3": "OPPORTUNITY",
        "4": "NOTE"
    }

    note_type = type_map[choice]

    # Content
    content = Prompt.ask("\n[white]Note content[/white]")

    # Tags (optional)
    tags = Prompt.ask("[bright_black]Tags (optional, comma-separated)[/bright_black]", default="")

    # Save
    note_id = add_note(symbol, note_type, content, tags if tags else None)

    console.print(f"\n[green]✓ Note saved (ID: {note_id})[/green]\n")


def interactive_add_bookmark(symbol):
    """Interactive bookmark addition"""
    console.print(f"\n[bold white]Add Bookmark for {symbol}[/bold white]\n")

    # Bookmark type
    console.print("[white]Bookmark Type:[/white]")
    console.print("  1. Key Metric")
    console.print("  2. News/Article")
    console.print("  3. Event/Catalyst")
    console.print("  4. Other")

    choice = Prompt.ask("Choose type", choices=["1", "2", "3", "4"], default="1")

    type_map = {
        "1": "METRIC",
        "2": "NEWS",
        "3": "EVENT",
        "4": "OTHER"
    }

    bookmark_type = type_map[choice]

    # Title
    title = Prompt.ask("[white]Title[/white]")

    # Value
    value = Prompt.ask("[bright_black]Value (optional)[/bright_black]", default="")

    # Notes
    notes = Prompt.ask("[bright_black]Notes (optional)[/bright_black]", default="")

    # Save
    bookmark_id = add_bookmark(
        symbol,
        bookmark_type,
        title,
        value if value else None,
        notes if notes else None
    )

    console.print(f"\n[green]✓ Bookmark saved (ID: {bookmark_id})[/green]\n")


def main_menu():
    """Main research notes menu"""
    init_research_db()

    while True:
        console.clear()
        console.print(create_header())
        console.print()

        # Get researched stocks
        stocks = get_all_researched_stocks()

        console.print("[bold white]Research Menu:[/bold white]\n")
        console.print("  [white]1[/white]  Add Note/Bookmark")
        console.print("  [white]2[/white]  View Stock Research")
        console.print("  [white]3[/white]  List All Researched Stocks")
        console.print("  [white]4[/white]  Export Notes to Markdown")
        console.print("  [white]5[/white]  Delete Note/Bookmark")
        console.print("  [red]6[/red]  Exit")

        if stocks:
            console.print(f"\n[bright_black]You have research on {len(stocks)} stock(s)[/bright_black]")

        console.print()

        choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "6"], default="2")

        if choice == "1":
            symbol = Prompt.ask("\n[white]Stock symbol[/white]").upper()

            console.print("\n[white]Add:[/white]")
            console.print("  1. Research Note")
            console.print("  2. Bookmark")

            add_choice = Prompt.ask("Choose", choices=["1", "2"])

            if add_choice == "1":
                interactive_add_note(symbol)
            else:
                interactive_add_bookmark(symbol)

            Prompt.ask("\nPress Enter to continue")

        elif choice == "2":
            if not stocks:
                console.print("\n[yellow]No research notes yet. Add some first![/yellow]\n")
                Prompt.ask("Press Enter to continue")
                continue

            symbol = Prompt.ask("\n[white]Stock symbol[/white]", default=stocks[0] if stocks else "").upper()
            display_stock_research(symbol)
            Prompt.ask("\nPress Enter to continue")

        elif choice == "3":
            console.clear()
            console.print("\n[bold white]All Researched Stocks:[/bold white]\n")

            if not stocks:
                console.print("[yellow]No research notes yet[/yellow]\n")
            else:
                for symbol in stocks:
                    notes_count = len(get_notes_for_symbol(symbol))
                    bookmarks_count = len(get_bookmarks_for_symbol(symbol))
                    console.print(f"  [white]{symbol}[/white] — {notes_count} notes, {bookmarks_count} bookmarks")

            console.print()
            Prompt.ask("Press Enter to continue")

        elif choice == "4":
            symbol = Prompt.ask("\n[white]Stock symbol to export[/white]").upper()
            filename = export_notes_to_markdown(symbol)

            if filename:
                console.print(f"\n[green]✓ Exported to {filename}[/green]\n")
            else:
                console.print(f"\n[yellow]No research found for {symbol}[/yellow]\n")

            Prompt.ask("Press Enter to continue")

        elif choice == "5":
            console.print("\n[yellow]Delete functionality - Coming soon![/yellow]\n")
            Prompt.ask("Press Enter to continue")

        elif choice == "6":
            console.print("\n[white]Goodbye![/white]\n")
            break


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]\n")
        sys.exit(1)
