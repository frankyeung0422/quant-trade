import os
from alpaca_trade_api.rest import REST, TimeFrame

def execute_trades(recommendation: str, ticker: str):
    """
    Execute trades on Alpaca based on recommendation.
    """
    key = os.getenv('APCA_API_KEY_ID')
    secret = os.getenv('APCA_API_SECRET_KEY')
    base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    if not key or not secret:
        print('Alpaca API keys not set. Skipping trade execution.')
        return
    api = REST(key, secret, base_url)
    position = None
    try:
        position = api.get_position(ticker)
    except Exception:
        pass
    qty = 1  # For demo, trade 1 share
    if recommendation == 'Buy' and not position:
        api.submit_order(symbol=ticker, qty=qty, side='buy', type='market', time_in_force='gtc')
        print(f'Placed BUY order for {qty} share(s) of {ticker}')
    elif recommendation == 'Sell' and position:
        api.submit_order(symbol=ticker, qty=qty, side='sell', type='market', time_in_force='gtc')
        print(f'Placed SELL order for {qty} share(s) of {ticker}')
    else:
        print('No trade executed.') 