# Contributing to AI Market Intelligence Platform

Thank you for your interest in contributing to the AI Market Intelligence Platform! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ai-market-intelligence.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit with clear messages: `git commit -m "Add: description of your changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- API keys (see docs/GET_API_KEYS.md)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ai-market-intelligence.git
cd ai-market-intelligence

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Code Standards

### Python Style Guide
- Follow PEP 8 style guidelines
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write docstrings for all functions and classes

### Code Formatting
We use the following tools for code quality:
```bash
# Format code with black
black .

# Check for linting issues
flake8 .

# Type checking
mypy .
```

### Testing
- Write tests for all new features
- Maintain test coverage above 80%
- Run tests before submitting PR:
```bash
pytest tests/
```

## Project Structure

```
ai-market-intelligence/
├── apps/           # User-facing applications
├── src/            # Core library code
│   ├── data/       # Data fetching and processing
│   ├── ml/         # Machine learning models
│   ├── api/        # API integrations
│   ├── analysis/   # Analysis tools
│   └── core/       # Core utilities
├── tests/          # Test files
├── scripts/        # Utility scripts
├── docs/           # Documentation
└── config/         # Configuration files
```

## Contribution Guidelines

### Adding New Features

1. **Research Tools** (in `apps/`)
   - Use consistent UI styling (rich library)
   - Follow the Bloomberg-inspired gray theme
   - Add error handling for API failures
   - Include data caching where appropriate

2. **ML Models** (in `src/ml/`)
   - Document model architecture and parameters
   - Include training and evaluation code
   - Add model versioning with MLflow
   - Provide example usage

3. **Data Sources** (in `src/data/`)
   - Handle rate limiting appropriately
   - Implement caching to reduce API calls
   - Add comprehensive error handling
   - Document data schemas

### Code Review Process

All submissions require review before merging:
1. Code must pass all tests
2. Code must follow style guidelines
3. New features must include tests
4. Documentation must be updated
5. At least one maintainer approval required

### Commit Message Format

Use clear, descriptive commit messages:
```
Type: Brief description (50 chars or less)

More detailed explanation if needed (72 chars per line)

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

Types:
- `Add:` New feature or functionality
- `Fix:` Bug fix
- `Update:` Improvement to existing feature
- `Refactor:` Code restructuring
- `Docs:` Documentation changes
- `Test:` Test additions or modifications
- `Style:` Formatting changes

## Areas for Contribution

### High Priority
- Additional data sources and API integrations
- More technical indicators and analysis tools
- Enhanced AI/ML models for prediction
- Performance optimizations
- Additional test coverage

### Medium Priority
- UI/UX improvements
- Additional screening strategies
- More visualization options
- Better error messages
- Code documentation

### Good First Issues
- Fix typos in documentation
- Add docstrings to functions
- Improve error handling
- Add unit tests for existing code
- Update screenshots in README

## Bug Reports

When reporting bugs, include:
1. Python version and OS
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Error messages and stack traces
6. API keys used (without sharing the actual keys)

## Feature Requests

When requesting features, include:
1. Clear description of the feature
2. Use cases and benefits
3. Potential implementation approach
4. Any relevant examples or references

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome diverse perspectives
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior
- Harassment or discriminatory language
- Trolling or insulting comments
- Personal or political attacks
- Publishing others' private information
- Other unprofessional conduct

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Review documentation in `/docs` directory
- Look at example scripts in `/scripts` directory

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions help make this platform better for everyone. We appreciate your time and effort!
