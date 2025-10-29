"""Verify installation and dependencies."""
import sys
from pathlib import Path

def check_imports():
    """Check all required imports."""
    print("Checking imports...")

    errors = []
    required = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('sklearn', 'scikit-learn'),
        ('faiss', 'FAISS'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('yfinance', 'yfinance'),
        ('prometheus_client', 'Prometheus Client'),
        ('loguru', 'Loguru'),
        ('mlflow', 'MLflow'),
        ('jwt', 'PyJWT'),
        ('yaml', 'PyYAML'),
        ('pytest', 'pytest'),
    ]

    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            errors.append(name)

    return errors

def check_directories():
    """Check required directories."""
    print("\nChecking directories...")

    required_dirs = [
        'data/raw',
        'data/processed',
        'data/cache',
        'data/feedback',
        'logs',
        'mlruns',
        'config',
        'src/api',
        'src/core',
        'src/retrieval',
        'src/models',
        'src/data',
        'src/ml',
        'src/evaluation',
        'tests',
        'scripts',
    ]

    missing = []
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            missing.append(dir_path)

    return missing

def check_files():
    """Check key files exist."""
    print("\nChecking key files...")

    required_files = [
        'config/config.yaml',
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'Makefile',
        '.env.example',
        'src/api/main.py',
        'src/core/config.py',
        'src/core/logging.py',
        'src/core/cache.py',
        'src/core/metrics.py',
        'src/ml/ensemble.py',
        'src/ml/drift_detection.py',
        'src/ml/mlflow_tracking.py',
        'src/data/realtime_stream.py',
        'src/evaluation/backtesting.py',
        'tests/test_api.py',
        'tests/test_ml.py',
        'scripts/train_models.py',
        'scripts/quick_start.py',
    ]

    missing = []
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing.append(file_path)

    return missing

def check_env():
    """Check environment configuration."""
    print("\nChecking environment...")

    env_file = Path('.env')
    if env_file.exists():
        print("  ✅ .env file exists")
        with open(env_file) as f:
            content = f.read()
            if 'ALPHA_VANTAGE_KEY' in content:
                print("  ✅ ALPHA_VANTAGE_KEY configured")
            else:
                print("  ⚠️  ALPHA_VANTAGE_KEY not set")

            if 'JWT_SECRET' in content:
                print("  ✅ JWT_SECRET configured")
            else:
                print("  ⚠️  JWT_SECRET not set")
    else:
        print("  ⚠️  .env file not found (copy from .env.example)")

def main():
    print("="*70)
    print("AI MARKET INTELLIGENCE - INSTALLATION VERIFICATION")
    print("="*70)

    # Check imports
    import_errors = check_imports()

    # Check directories
    missing_dirs = check_directories()

    # Check files
    missing_files = check_files()

    # Check environment
    check_env()

    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    if import_errors:
        print(f"\n❌ Missing packages: {', '.join(import_errors)}")
        print("   Run: pip install -r requirements.txt")
    else:
        print("\n✅ All packages installed")

    if missing_dirs:
        print(f"\n❌ Missing directories: {len(missing_dirs)}")
        print("   Run: make setup-dev")
    else:
        print("✅ All directories present")

    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
    else:
        print("✅ All files present")

    if not import_errors and not missing_dirs and not missing_files:
        print("\n" + "="*70)
        print("🎉 INSTALLATION VERIFIED - READY TO USE!")
        print("="*70)
        print("\nNext steps:")
        print("1. make train          # Train models")
        print("2. make docker-up      # Start services")
        print("3. make run-api        # Start API server")
        print("4. make demo           # Run demo")
        return 0
    else:
        print("\n⚠️  Some issues found. Please fix and re-run verification.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
