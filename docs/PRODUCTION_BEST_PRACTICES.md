# Production Code Best Practices

This document outlines the big tech production standards implemented in this codebase.

## Code Quality Standards

### 1. Type Hints (PEP 484)
**ALWAYS use type hints** for function parameters and return values.

```python
# ❌ Bad
def get_stock_data(symbol):
    return data

# ✅ Good
def get_stock_data(symbol: str) -> Dict[str, Any]:
    return data
```

### 2. Minimal Comments
**Code should be self-documenting.** Only comment when necessary.

```python
# ❌ Bad - Obvious comments
# Loop through symbols
for symbol in symbols:
    # Get the data
    data = fetch_data(symbol)
    # Process the data
    process(data)

# ✅ Good - Self-documenting code
for symbol in symbols:
    data = fetch_data(symbol)
    process(data)

# ✅ Good - Comment only for complex logic
# Using Black-Scholes for European options only; American options require binomial model
price = black_scholes_price(strike, spot, volatility)
```

### 3. Error Handling
**Specific exceptions, proper logging, graceful degradation.**

```python
# ❌ Bad
try:
    data = fetch_data()
except:
    pass

# ✅ Good
try:
    data = fetch_data(symbol)
except requests.HTTPError as e:
    logger.error(f"HTTP error fetching {symbol}: {e}", exc_info=True)
    return None
except requests.Timeout:
    logger.warning(f"Timeout fetching {symbol}, retrying...")
    return retry_fetch(symbol)
```

### 4. DRY Principle
**Never repeat yourself.** Extract common patterns into reusable utilities.

```python
# ❌ Bad - Repeated in every file
table = Table(
    show_header=True,
    header_style="bold white on grey23",
    border_style="grey35",
    ...
)

# ✅ Good - Centralized utility
from src.ui.components import create_table
table = create_table(columns, rows, title="Data")
```

### 5. Immutability
**Use immutable data structures where possible.**

```python
# ❌ Bad - Mutable config
THEME = {
    'border': 'grey35',
    'bg': 'on grey11'
}

# ✅ Good - Immutable config
@dataclass(frozen=True)
class Theme:
    border: str = 'grey35'
    bg: str = 'on grey11'

THEME = Theme()
```

### 6. Input Validation
**Validate and sanitize ALL user inputs.**

```python
# ❌ Bad - Direct usage
symbol = input("Enter symbol: ")
query_db(f"SELECT * FROM stocks WHERE symbol = '{symbol}'")

# ✅ Good - Validated and parameterized
from src.core.validators import StockSymbolValidator

result = StockSymbolValidator.validate(symbol)
if result.is_valid:
    query_db("SELECT * FROM stocks WHERE symbol = ?", (result.value,))
```

### 7. Logging Over Print
**Use structured logging, not print statements.**

```python
# ❌ Bad
print(f"Processing {symbol}...")
print(f"Error: {error}")

# ✅ Good
logger.info(f"Processing symbol: {symbol}")
logger.error(f"Failed to process {symbol}: {error}", exc_info=True)
```

### 8. Configuration Management
**Centralize configuration, support environment variables.**

```python
# ❌ Bad - Hardcoded everywhere
refresh_interval = 60
cache_ttl = 300

# ✅ Good - Centralized config
from src.ui.config import CONFIG
refresh_interval = CONFIG.refresh_interval
cache_ttl = CONFIG.cache_ttl
```

### 9. Function Length
**Keep functions short and focused.** Max 50 lines is a good target.

```python
# ❌ Bad - 200 line function doing everything
def process_stock_data():
    # Fetch data
    # Clean data
    # Calculate metrics
    # Generate charts
    # Send notifications
    # Update database
    ...

# ✅ Good - Single responsibility
def process_stock_data(symbol: str) -> ProcessedData:
    raw_data = fetch_data(symbol)
    cleaned = clean_data(raw_data)
    return calculate_metrics(cleaned)
```

### 10. Docstrings
**Use Google-style docstrings for public functions.**

```python
def calculate_return(
    initial: float,
    final: float,
    periods: int
) -> float:
    """
    Calculate annualized return.

    Args:
        initial: Initial investment value
        final: Final investment value
        periods: Number of periods

    Returns:
        Annualized return as decimal (e.g., 0.15 for 15%)

    Raises:
        ValueError: If initial <= 0 or periods <= 0
    """
    if initial <= 0 or periods <= 0:
        raise ValueError("Initial and periods must be positive")

    return (final / initial) ** (1 / periods) - 1
```

## Architecture Patterns

### Separation of Concerns

```
apps/           # User interface layer
src/
  ├── core/     # Cross-cutting concerns (logging, config, validation)
  ├── data/     # Data fetching and processing
  ├── models/   # Machine learning models
  └── ui/       # UI components and utilities
```

### Dependency Injection

```python
# ✅ Good - Testable, flexible
class StockAnalyzer:
    def __init__(self, data_provider: DataProvider, logger: Logger):
        self.data_provider = data_provider
        self.logger = logger

    def analyze(self, symbol: str) -> Analysis:
        data = self.data_provider.fetch(symbol)
        return self._process(data)
```

### Resource Management

```python
# ❌ Bad
file = open('data.csv')
data = file.read()
file.close()

# ✅ Good
with open('data.csv') as file:
    data = file.read()

# ✅ Good - Database connections
with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
```

## Performance

### Async I/O for Network Calls

```python
# ❌ Bad - Sequential
for symbol in symbols:
    data = fetch_data(symbol)  # Blocks

# ✅ Good - Concurrent
async def fetch_all(symbols: List[str]) -> List[Data]:
    tasks = [fetch_data(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param: str) -> Result:
    ...
```

### Lazy Loading

```python
# ✅ Use generators for large datasets
def read_large_file(path: Path) -> Iterator[str]:
    with open(path) as f:
        for line in f:
            yield process_line(line)
```

## Security

### SQL Injection Prevention
```python
# ❌ Bad
query = f"SELECT * FROM users WHERE id = '{user_id}'"

# ✅ Good
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

### Environment Variables for Secrets
```python
# ❌ Bad
API_KEY = "abc123secret"

# ✅ Good
from os import getenv
API_KEY = getenv("API_KEY")
if not API_KEY:
    raise EnvironmentError("API_KEY not set")
```

## Testing

### Unit Tests
```python
def test_stock_validator():
    # Arrange
    validator = StockSymbolValidator()

    # Act
    result = validator.validate("AAPL")

    # Assert
    assert result.is_valid
    assert result.value == "AAPL"
```

### Test Coverage
- Aim for 80%+ code coverage
- Test edge cases and error paths
- Mock external dependencies

## Git Practices

### Commit Messages
```
feat: Add stock symbol validation
fix: Handle network timeout in data fetcher
refactor: Extract table creation to UI components
docs: Update API documentation
test: Add unit tests for validators
```

### Branch Strategy
- `main` - Production code
- `develop` - Integration branch
- `feature/xxx` - Feature branches
- `hotfix/xxx` - Critical fixes

## Code Review Checklist

- [ ] Type hints on all functions
- [ ] No verbose/obvious comments
- [ ] Proper error handling
- [ ] Input validation
- [ ] Logging instead of prints
- [ ] No hardcoded values
- [ ] Functions < 50 lines
- [ ] DRY principle followed
- [ ] Tests added/updated
- [ ] Documentation updated

## Tools

### Code Quality
```bash
# Format code
black .

# Lint
flake8 .
pylint src/

# Type checking
mypy src/

# Security
bandit -r src/
```

### Performance
```bash
# Profile code
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats

# Memory profiling
python -m memory_profiler script.py
```

## References

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 484 – Type Hints](https://peps.python.org/pep-0484/)
- [The Zen of Python (PEP 20)](https://peps.python.org/pep-0020/)
