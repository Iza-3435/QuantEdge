# 🔑 Complete API Keys Guide

## Required API Keys (All FREE - No Credit Card!)

### 1. NewsAPI ⭐ (Highest Priority)
**What it does:** Real-time news from 80,000+ sources

**Sign up:** https://newsapi.org/register
- Enter email
- Choose "Developer" plan (FREE)
- Get instant API key

**Free tier:**
- 100 requests/day
- 1 month of historical data
- Perfect for this system!

**Add to terminal:**
```bash
export NEWSAPI_KEY='your_key_here'
```

---

### 2. Alpha Vantage ⭐ (Highest Priority)
**What it does:** Financial news + sentiment analysis included!

**Get key:** https://www.alphavantage.co/support/#api-key
- Just click "Get Free API Key"
- Enter email
- Instant key!

**Free tier:**
- 5 API calls per minute
- 500 calls per day
- **Includes sentiment scores!**

**Add to terminal:**
```bash
export ALPHAVANTAGE_KEY='your_key_here'
```

---

### 3. Finnhub 🌟 (Recommended)
**What it does:** Stock-specific news and data

**Sign up:** https://finnhub.io/register
- Create free account
- Get API key from dashboard

**Free tier:**
- 60 API calls per minute
- Great for stock news!

**Add to terminal:**
```bash
export FINNHUB_KEY='your_key_here'
```

---

### 4. Twitter API (Optional - For Social Sentiment)
**What it does:** Real-time Twitter sentiment tracking

**Sign up:** https://developer.twitter.com/en/portal/petition/essential/basic-info
- Create Twitter Developer account
- Create "App"
- Get Bearer Token

**Free tier:**
- 500,000 tweets per month
- Essential access (free)

**Add to terminal:**
```bash
export TWITTER_BEARER_TOKEN='your_bearer_token_here'
```

---

### 5. No API Key Needed ✅
These work automatically:
- **Yahoo Finance** - Free, unlimited, built into yfinance
- **StockTwits** - Free public API
- **Reddit** - Free with praw (no authentication needed for read-only)

---

## 🚀 Quick Setup (Copy-Paste Ready)

### Step 1: Get Your Keys
1. Visit each URL above
2. Sign up (takes 2 min each)
3. Copy your API keys

### Step 2: Add to Your Shell

**For Mac/Linux (bash):**
```bash
# Open your bash config
nano ~/.bashrc

# Add these lines (replace with your actual keys):
export NEWSAPI_KEY='paste_your_newsapi_key_here'
export ALPHAVANTAGE_KEY='paste_your_alphavantage_key_here'
export FINNHUB_KEY='paste_your_finnhub_key_here'
export TWITTER_BEARER_TOKEN='paste_your_twitter_token_here'

# Save (Ctrl+X, then Y, then Enter)

# Reload your config
source ~/.bashrc
```

**For Mac with zsh:**
```bash
# Open your zsh config
nano ~/.zshrc

# Add the same export lines above

# Save and reload
source ~/.zshrc
```

### Step 3: Verify Keys Are Set
```bash
echo $NEWSAPI_KEY
echo $ALPHAVANTAGE_KEY
echo $FINNHUB_KEY
```

If you see your keys, you're ready! ✅

---

## 📊 What Each API Provides

| API | News Count | Sentiment | Special Features | Priority |
|-----|-----------|-----------|------------------|----------|
| **NewsAPI** | 50-100 | ❌ | 80k sources, global coverage | ⭐⭐⭐ |
| **Alpha Vantage** | 20-50 | ✅ | **Sentiment included!** | ⭐⭐⭐ |
| **Finnhub** | 10-20 | ❌ | Stock-specific, fast | ⭐⭐ |
| **Twitter** | Real-time | Manual | Social sentiment | ⭐ |
| **Yahoo Finance** | 5-10 | ❌ | Free, no key needed | ✅ Auto |

---

## 💡 Recommended Setup

### Minimum (Good Experience):
```bash
export NEWSAPI_KEY='your_key'
export ALPHAVANTAGE_KEY='your_key'
```
**Result:** 70-150 real articles with sentiment

### Recommended (Best Experience):
```bash
export NEWSAPI_KEY='your_key'
export ALPHAVANTAGE_KEY='your_key'
export FINNHUB_KEY='your_key'
```
**Result:** 80-170 real articles with sentiment + stock-specific news

### Maximum (Full Features):
```bash
export NEWSAPI_KEY='your_key'
export ALPHAVANTAGE_KEY='your_key'
export FINNHUB_KEY='your_key'
export TWITTER_BEARER_TOKEN='your_token'
```
**Result:** 100+ articles + real-time social sentiment

---

## 🎯 Priority Order

1. **MUST HAVE:**
   - NewsAPI (most articles)
   - Alpha Vantage (has sentiment!)

2. **RECOMMENDED:**
   - Finnhub (stock-specific)

3. **NICE TO HAVE:**
   - Twitter API (social sentiment)

4. **AUTOMATIC:**
   - Yahoo Finance (always works)
   - StockTwits (always works)
   - Reddit (always works)

---

## ⚡ Super Quick Setup Script

Run this interactive script:
```bash
chmod +x setup_api_keys.sh
./setup_api_keys.sh
```

It will prompt you for each key and set them up automatically!

---

## 🔍 Troubleshooting

### "No articles found"
**Solution:** Set at least NewsAPI or Alpha Vantage key

### "API rate limit exceeded"
**Solution:**
- NewsAPI: 100/day limit, wait or use other APIs
- Alpha Vantage: 5/minute limit, wait 60 seconds between calls
- Finnhub: 60/minute, rarely hit

### "Invalid API key"
**Solution:**
- Check for extra spaces
- Make sure you copied the full key
- Verify with: `echo $NEWSAPI_KEY`

### Keys not persisting after restart
**Solution:**
- Make sure you added to ~/.bashrc or ~/.zshrc
- Run `source ~/.bashrc` or `source ~/.zshrc`
- Or add to ~/.profile for permanent setup

---

## ✅ Verification

After setting keys, test them:

```bash
# Test NewsAPI
python -c "
from src.news.real_news_aggregator import RealNewsAggregator
agg = RealNewsAggregator()
articles = agg.get_real_news('AAPL', 'Apple Inc.', lookback_days=7)
print(f'✅ Found {len(articles)} articles')
for a in articles[:3]:
    print(f'  - {a.title[:60]}...')
"
```

**Expected output:**
```
✅ Fetched 29 articles from NewsAPI
✅ Fetched 50 articles from Alpha Vantage
✅ Fetched 10 articles from Yahoo Finance
✅ Found 89 articles
  - Apple reports strong earnings, beating Wall Street...
  - iPhone 16 sales exceed expectations in China...
  - Tim Cook discusses AI strategy in interview...
```

---

## 🎉 You're Ready!

Once you have at least **NewsAPI** and **Alpha Vantage** keys set, run:

```bash
# Bloomberg dashboard with real news
python bloomberg_dashboard.py --symbol AAPL

# Ultimate quant dashboard
python ultimate_quant_dashboard.py --symbol AAPL
```

**Total time to set up:** 10-15 minutes
**Total cost:** $0 (all free!)
**Result:** Professional-grade system with real data! 🚀
