#!/bin/bash
# API Keys Setup Script
# Run this to configure your API keys

echo "=================================="
echo "  API KEYS SETUP"
echo "=================================="
echo ""
echo "This script will help you set up API keys for the AI Market Intelligence System."
echo ""

# Detect shell
if [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

echo "Detected shell config: $SHELL_RC"
echo ""

# Function to add key
add_key() {
    local key_name=$1
    local key_value=$2

    # Check if key already exists
    if grep -q "export $key_name=" "$SHELL_RC" 2>/dev/null; then
        echo "⚠️  $key_name already exists in $SHELL_RC"
        read -p "Overwrite? (y/n): " overwrite
        if [ "$overwrite" = "y" ]; then
            # Remove old line
            sed -i.bak "/export $key_name=/d" "$SHELL_RC"
            echo "export $key_name='$key_value'" >> "$SHELL_RC"
            echo "✅ Updated $key_name"
        else
            echo "⏭️  Skipped $key_name"
        fi
    else
        echo "export $key_name='$key_value'" >> "$SHELL_RC"
        echo "✅ Added $key_name"
    fi
}

echo "=================================="
echo "  STEP 1: NewsAPI"
echo "=================================="
echo "Sign up (free): https://newsapi.org/register"
echo "Free tier: 100 requests/day"
echo ""
read -p "Enter your NewsAPI key (or press Enter to skip): " newsapi_key

if [ ! -z "$newsapi_key" ]; then
    add_key "NEWSAPI_KEY" "$newsapi_key"
else
    echo "⏭️  Skipped NewsAPI"
fi

echo ""
echo "=================================="
echo "  STEP 2: Alpha Vantage"
echo "=================================="
echo "Get free key: https://www.alphavantage.co/support/#api-key"
echo "Free tier: 5 calls/min, 500 calls/day"
echo ""
read -p "Enter your Alpha Vantage key (or press Enter to skip): " alphavantage_key

if [ ! -z "$alphavantage_key" ]; then
    add_key "ALPHAVANTAGE_KEY" "$alphavantage_key"
else
    echo "⏭️  Skipped Alpha Vantage"
fi

echo ""
echo "=================================="
echo "  STEP 3: Finnhub"
echo "=================================="
echo "Sign up (free): https://finnhub.io/register"
echo "Free tier: 60 calls/min"
echo ""
read -p "Enter your Finnhub key (or press Enter to skip): " finnhub_key

if [ ! -z "$finnhub_key" ]; then
    add_key "FINNHUB_KEY" "$finnhub_key"
else
    echo "⏭️  Skipped Finnhub"
fi

echo ""
echo "=================================="
echo "  STEP 4: Twitter API (Optional)"
echo "=================================="
echo "Sign up: https://developer.twitter.com/"
echo "Free tier: 500k tweets/month"
echo ""
read -p "Enter your Twitter Bearer Token (or press Enter to skip): " twitter_token

if [ ! -z "$twitter_token" ]; then
    add_key "TWITTER_BEARER_TOKEN" "$twitter_token"
else
    echo "⏭️  Skipped Twitter API"
fi

echo ""
echo "=================================="
echo "  SETUP COMPLETE!"
echo "=================================="
echo ""
echo "API keys have been added to: $SHELL_RC"
echo ""
echo "To activate the keys, run:"
echo "  source $SHELL_RC"
echo ""
echo "Or restart your terminal."
echo ""
echo "To verify:"
echo "  echo \$NEWSAPI_KEY"
echo "  echo \$ALPHAVANTAGE_KEY"
echo "  echo \$FINNHUB_KEY"
echo ""
echo "Next steps:"
echo "  1. source $SHELL_RC"
echo "  2. python bloomberg_dashboard.py --symbol AAPL"
echo "  3. python scripts/finetune_finbert.py"
echo ""
