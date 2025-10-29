"""
Quick test script to verify API keys are working
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("🔑 API KEY VERIFICATION TEST")
print("=" * 80)
print()

# Check what keys are loaded
newsapi_key = os.getenv('NEWSAPI_KEY')
alphavantage_key = os.getenv('ALPHAVANTAGE_KEY')
sec_user_agent = os.getenv('SEC_USER_AGENT')

print("📋 Configuration Status:")
print()
print(f"  NewsAPI Key:       {'✅ SET' if newsapi_key else '❌ NOT SET'}")
if newsapi_key:
    print(f"                     {newsapi_key[:10]}...{newsapi_key[-4:]}")
print()
print(f"  Alpha Vantage Key: {'✅ SET' if alphavantage_key else '❌ NOT SET'}")
if alphavantage_key:
    print(f"                     {alphavantage_key[:10]}...{alphavantage_key[-4:]}")
print()
print(f"  SEC User Agent:    {'✅ SET' if sec_user_agent else '❌ NOT SET'}")
if sec_user_agent:
    print(f"                     {sec_user_agent}")
print()

# Test NewsAPI
if newsapi_key:
    print("-" * 80)
    print("📰 Testing NewsAPI (Free tier: 100 requests/day)")
    print("-" * 80)
    try:
        import requests
        url = f"https://newsapi.org/v2/everything?q=Apple&apiKey={newsapi_key}&pageSize=3"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"✅ NewsAPI Working! Found {len(articles)} articles")
            if articles:
                print(f"   Latest: {articles[0].get('title', 'N/A')[:60]}...")
        elif response.status_code == 429:
            print("⚠️  Rate limit reached (100 requests/day for free tier)")
        elif response.status_code == 401:
            print("❌ Invalid API key")
        else:
            print(f"⚠️  Status: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️  NewsAPI key not configured")

print()

# Test Alpha Vantage
if alphavantage_key:
    print("-" * 80)
    print("📊 Testing Alpha Vantage (Free tier: 25 requests/day)")
    print("-" * 80)
    try:
        import requests
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={alphavantage_key}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data and data['Global Quote']:
                print(f"✅ Alpha Vantage Working!")
                quote = data['Global Quote']
                print(f"   AAPL Price: ${quote.get('05. price', 'N/A')}")
            elif 'Note' in data:
                print("⚠️  Rate limit reached (25 requests/day for free tier)")
                print(f"   Message: {data['Note'][:80]}...")
            elif 'Information' in data:
                print("⚠️  Rate limit reached")
                print(f"   Message: {data['Information'][:80]}...")
            else:
                print(f"⚠️  Unexpected response: {data}")
        else:
            print(f"⚠️  Status: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️  Alpha Vantage key not configured")

print()

# Test SEC EDGAR
if sec_user_agent:
    print("-" * 80)
    print("📄 Testing SEC EDGAR (Free: 10 requests/second)")
    print("-" * 80)
    try:
        import requests
        session = requests.Session()
        session.headers.update({
            'User-Agent': sec_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })

        url = "https://data.sec.gov/files/company_tickers.json"
        response = session.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SEC EDGAR Working!")
            print(f"   Can access {len(data)} company tickers")
            # Find Apple
            for entry in data.values():
                if entry['ticker'] == 'AAPL':
                    print(f"   Example: AAPL CIK = {entry['cik_str']}")
                    break
        else:
            print(f"⚠️  Status: {response.status_code}")

    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("⚠️  SEC user agent not configured")

print()
print("=" * 80)
print("✅ API KEY TEST COMPLETE")
print("=" * 80)
print()
print("📊 RATE LIMITS TO REMEMBER:")
print()
print("  Free Tier Limits:")
print("  • NewsAPI:       100 requests/day")
print("  • Alpha Vantage: 25 requests/day (5 per minute)")
print("  • SEC EDGAR:     10 requests/second (FREE, unlimited daily)")
print()
print("💡 TIP: The system caches results to minimize API calls!")
print()
