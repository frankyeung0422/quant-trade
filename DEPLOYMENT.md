# Quant Trader Pro - Deployment Guide

## Overview
This guide covers deploying the Quant Trader Pro Streamlit application to various platforms.

## Prerequisites
- Python 3.8+
- All dependencies listed in `requirements.txt`
- Alpaca Trading API credentials (for live trading features)

## Environment Variables
Set the following environment variables for deployment:

```bash
# Debug mode
DEBUG=False

# Streamlit settings
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0

# Alpaca Trading API (Paper Trading)
APCA_API_KEY_ID=your_alpaca_api_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_key_here
APCA_API_BASE_URL=https://paper-api.alpaca.markets

# For live trading (use with caution)
# APCA_API_BASE_URL=https://api.alpaca.markets

# Optional: Set to True for live trading (default is paper trading)
LIVE_TRADING=False
```

## Deployment Options

### 1. Streamlit Cloud (Recommended)

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and set `web_app.py` as the main file
3. **Configure Environment Variables**:
   - In the Streamlit Cloud dashboard, go to your app settings
   - Add the environment variables listed above
4. **Deploy**: Click "Deploy" and wait for the build to complete

### 2. Heroku

1. **Install Heroku CLI** and login
2. **Create Heroku app**:
   ```bash
   heroku create your-app-name
   ```
3. **Set environment variables**:
   ```bash
   heroku config:set APCA_API_KEY_ID=your_key
   heroku config:set APCA_API_SECRET_KEY=your_secret
   heroku config:set APCA_API_BASE_URL=https://paper-api.alpaca.markets
   ```
4. **Deploy**:
   ```bash
   git push heroku main
   ```

### 3. Railway

1. **Connect GitHub repository** to Railway
2. **Set environment variables** in Railway dashboard
3. **Deploy automatically** on git push

### 4. Local Deployment

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables** (create `.env` file):
   ```bash
   export APCA_API_KEY_ID=your_key
   export APCA_API_SECRET_KEY=your_secret
   export APCA_API_BASE_URL=https://paper-api.alpaca.markets
   ```

3. **Run the app**:
   ```bash
   streamlit run web_app.py
   ```

## Configuration Files

### Streamlit Config (`.streamlit/config.toml`)
Already configured for deployment with:
- Headless mode enabled
- CORS disabled
- Custom theme
- Production settings

### Requirements (`requirements.txt`)
All necessary Python packages are included.

### System Dependencies (`packages.txt`)
Includes build tools for compilation.

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Paper Trading**: Use paper trading for testing
3. **Environment Variables**: Use platform-specific secret management
4. **HTTPS**: Ensure HTTPS is enabled in production

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are in `requirements.txt`
2. **API Connection**: Verify Alpaca API credentials
3. **Memory Issues**: Consider upgrading deployment plan
4. **Timeout Errors**: Optimize data fetching and caching

### Performance Optimization

1. **Data Caching**: The app includes built-in caching
2. **Lazy Loading**: Charts load on demand
3. **Session State**: Efficient state management
4. **Background Processing**: Heavy computations run in background

## Monitoring

- **Streamlit Cloud**: Built-in monitoring and logs
- **Heroku**: Use `heroku logs --tail`
- **Custom**: Add logging to `web_app.py`

## Support

For deployment issues:
1. Check the logs in your deployment platform
2. Verify environment variables are set correctly
3. Test locally first
4. Check Streamlit documentation for platform-specific issues 