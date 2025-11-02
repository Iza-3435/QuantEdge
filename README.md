# QuantEdge

Professional quantitative analysis platform for stock market intelligence.

## Features

### Core Applications
- **Market Overview** - Real-time market data and analytics
- **Stock Screener** - Advanced filtering with quality metrics
- **Portfolio Manager** - Track holdings with performance analytics
- **Dividend Tracker** - Income investing dashboard
- **Correlation Matrix** - Portfolio diversification analysis
- **Comparison Matrix** - Side-by-side stock comparison
- **Earnings Calendar** - Track upcoming earnings
- **Watchlist Manager** - Monitor favorite stocks
- **Research Terminal** - Professional-grade research tools
- **AI Stock Picker** - ML-powered stock recommendations

### Advanced Analytics
- Technical Indicators (MACD, RSI, Bollinger Bands, etc.)
- Quality Scores (Piotroski F-Score, Altman Z-Score)
- 30-day price sparklines
- Color-coded performance metrics
- Professional data visualizations

## Installation

```bash
git clone https://github.com/Iza-3435/QuantEdge.git
cd QuantEdge

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
# Launch main menu
python MAIN_MENU.py

# Or run individual apps
python apps/MARKET_OVERVIEW.py
python apps/STOCK_SCREENER.py
python apps/PORTFOLIO_PRO.py
```

## Project Structure

```
QuantEdge/
├── apps/                          # Core applications
│   ├── MARKET_OVERVIEW.py
│   ├── STOCK_SCREENER.py
│   ├── PORTFOLIO_PRO.py
│   ├── DIVIDEND_TRACKER.py
│   ├── CORRELATION_MATRIX.py
│   ├── COMPARISON_MATRIX.py
│   ├── EARNINGS_CALENDAR.py
│   ├── WATCHLIST_PRO.py
│   ├── PROFESSIONAL_RESEARCH_TERMINAL.py
│   ├── AI_STOCK_PICKER.py
│   └── ...
│
├── src/                           # Shared modules
│   ├── quality_scores.py          # Piotroski, Altman Z-Score
│   ├── technical_indicators.py    # MACD, RSI, Bollinger Bands
│   └── ui_enhancements.py         # Visualizations
│
├── utils/                         # Utilities
│   └── data_cache.py              # Data caching
│
├── MAIN_MENU.py                   # Main entry point
└── requirements.txt               # Dependencies
```

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- rich
- requests

## License

MIT License - See LICENSE file for details.

## Disclaimer

This platform is for educational purposes only. Always conduct your own due diligence before making investment decisions.
