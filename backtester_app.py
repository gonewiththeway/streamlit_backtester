import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import traceback
import numpy as np

st.set_page_config(layout="wide")

st.title("Trader_Joe80 Swing Options Backtester")

# --- User Inputs ---
st.sidebar.header("Backtest Configuration")

position_size = st.sidebar.number_input("Position Size ($)", min_value=1000, value=5000, step=500)
start_budget = st.sidebar.number_input("Starting Budget", min_value=1000, value=10000, step=1000)

default_tickers = [
    "NVDA", "TEM", "GOOG", "META", "AMZN", "ORCL", "CRM", "PLTR", "SNOW",
    "MSTR", "AMD", "AVGO", "TSM", "ASML", "INTC", "QCOM", "MU", "AAPL",
    "TSLA", "NFLX", "PYPL", "UBER", "SHOP", "SQ", "COIN"
]
selected_tickers = st.sidebar.multiselect("Select Stocks to Backtest", options=default_tickers, default=default_tickers)

st.sidebar.subheader("Strategy Parameters")
max_option_premium = st.sidebar.slider("Max Option Premium ($)", 1.0, 5.0, 3.0, 0.25)
option_expiration_days = st.sidebar.slider("Option Expiration (Days)", 30, 90, 60, 5)
stop_loss_pct = st.sidebar.slider("Stop-Loss (%)", 5, 50, 25, 1)
take_profit_pct = st.sidebar.slider("Take-Profit (%)", 10, 200, 50, 5)


col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("today") - pd.DateOffset(years=1))
with col2:
    end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.sidebar.button("Run Backtest"):
    
    if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days > 730:
        st.error("Date range cannot exceed 730 days. Please select a shorter period.")
        st.stop()

    @st.cache_data
    def get_data(ticker, start, end):
        """Fetches and prepares weekly, daily, and 4-hour data for a given ticker."""
        try:
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            
            stock = yf.Ticker(ticker)
            extended_start = start - pd.DateOffset(years=2)
            data_1w = stock.history(start=extended_start, end=end, interval="1wk", auto_adjust=True)
            data_1d = stock.history(start=extended_start, end=end, interval="1d", auto_adjust=True)
            data_1h = stock.history(start=start, end=end, interval="1h", auto_adjust=True)
            
            if data_1d.empty or data_1h.empty or data_1w.empty:
                st.warning(f"No data for {ticker}, it may be delisted.")
                return None, None, None

            for df in [data_1w, data_1d, data_1h]:
                df.columns = [col.title() for col in df.columns]
                if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in data_1h.columns for col in required_cols):
                return None, None, None

            data_1h = data_1h.between_time('09:30', '16:00')
            data_4h = data_1h.resample('4h', offset='30min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            # Add 50 EMA for trend confirmation and 200 EMA for entry
            for df in [data_1w, data_1d, data_4h]:
                df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

            data_1w = data_1w.loc[start:]
            data_1d = data_1d.loc[start:]
            data_4h = data_4h.loc[start:]

            return data_1w, data_1d, data_4h
        except Exception as e:
            st.error(f"Failed to get data for {ticker}: {e}")
            return None, None, None

    def get_trend_direction(weekly_data, daily_data, four_hour_data, current_date):
        """Confirms trend using 50 EMA on W, D, and 4h charts."""
        try:
            latest_w = weekly_data[weekly_data.index <= current_date].iloc[-1]
            latest_d = daily_data[daily_data.index <= current_date].iloc[-1]
            latest_4h = four_hour_data[four_hour_data.index <= current_date].iloc[-1]
            
            is_uptrend = latest_w['Close'] > latest_w['EMA50'] and \
                         latest_d['Close'] > latest_d['EMA50'] and \
                         latest_4h['Close'] > latest_4h['EMA50']
            
            is_downtrend = latest_w['Close'] < latest_w['EMA50'] and \
                           latest_d['Close'] < latest_d['EMA50'] and \
                           latest_4h['Close'] < latest_4h['EMA50']

            if is_uptrend: return "Up"
            elif is_downtrend: return "Down"
            else: return "None"
        except IndexError:
            return "None"

    def is_not_at_ath(daily_data, current_date):
        historical_data = daily_data.loc[:current_date]
        if historical_data.empty: return True
        return historical_data['Close'].iloc[-1] < historical_data['High'].max()
    
    def is_not_at_atl(daily_data, current_date):
        historical_data = daily_data.loc[:current_date]
        if historical_data.empty: return True
        return historical_data['Close'].iloc[-1] > historical_data['Low'].min()
    
    # --- Backtesting Engine ---
    all_trades = []
    portfolio_value = start_budget
    
    for ticker in selected_tickers:
        st.write(f"--- Processing {ticker} ---")
        
        weekly_data, daily_data, four_hour_data = get_data(ticker, start_date, end_date)
        
        if four_hour_data is None or four_hour_data.empty:
            st.warning(f"Could not fetch or process data for {ticker}. Skipping.")
            continue

        buy_signals_dates = []
        sell_signals_dates = []
        
        in_trade = False
        current_trade = {}

        for i in range(1, len(four_hour_data) - 1): # -1 to allow entry on next candle
            current_ts = four_hour_data.index[i]
            current_row = four_hour_data.iloc[i]
            prev_close = four_hour_data['Close'].iat[i-1]
            
            # --- Exit Logic ---
            if in_trade:
                exit_reason = None
                # Percentage Stop-Loss
                current_pl = 0
                if current_trade['Type'] == 'Call':
                    current_pl = (current_row['Close'] - current_trade['Entry Price']) * current_trade['Contracts'] * 100 * 0.5
                else:
                    current_pl = (current_trade['Entry Price'] - current_row['Close']) * current_trade['Contracts'] * 100 * 0.5
                
                if (current_pl / (current_trade['Contracts'] * current_trade['Premium'] * 100)) <= -(stop_loss_pct / 100):
                    exit_reason = f"{stop_loss_pct}% Stop Loss"
                # Technical Stop
                elif (current_trade['Type'] == 'Call' and current_row['Close'] < current_trade['Stop']) or \
                     (current_trade['Type'] == 'Put' and current_row['Close'] > current_trade['Stop']):
                    exit_reason = "Technical Stop"
                # Take Profit
                elif (current_pl / (current_trade['Contracts'] * current_trade['Premium'] * 100)) >= (take_profit_pct / 100):
                    exit_reason = f"{take_profit_pct}% Take Profit"
                # Expiration
                elif current_ts >= current_trade['Expiration']:
                    exit_reason = "Expiration"

                if exit_reason:
                    exit_price = current_row['Close']
                    profit_loss = current_pl
                    portfolio_value += profit_loss

                    all_trades.append({
                        "Ticker": ticker, "Strategy": current_trade['Type'], "Entry Date": current_trade['Entry Date'].date(),
                        "Exit Date": current_ts.date(), "Exit Reason": exit_reason,
                        "Entry Price": f"${current_trade['Entry Price']:.2f}", "Exit Price": f"${exit_price:.2f}",
                        "Contracts": current_trade['Contracts'], "Entry Premium": f"${current_trade['Premium']:.2f}",
                        "Options P/L": f"${profit_loss:.2f}"
                    })
                    in_trade = False
                    current_trade = {}

            # --- Entry Logic ---
            if not in_trade:
                trend_direction = get_trend_direction(weekly_data, daily_data, four_hour_data, current_ts)
                ema200 = current_row['EMA200']
                
                # Call Signal: Wait for close of breakout candle
                if trend_direction == "Up" and is_not_at_ath(daily_data, current_ts.date()):
                    if prev_close < ema200 and current_row['Close'] > ema200:
                        buy_signals_dates.append(current_ts)
                        entry_candle = four_hour_data.iloc[i+1]
                        entry_price = entry_candle['Open']
                        
                        # Dynamic Premium Proxy
                        proxy_premium = entry_price * 0.03 # 3% of stock price as premium proxy
                        if proxy_premium < max_option_premium:
                            num_contracts = position_size // (proxy_premium * 100)
                            if num_contracts > 0:
                                in_trade = True
                                current_trade = {
                                    'Type': 'Call', 'Entry Date': entry_candle.name, 'Entry Price': entry_price,
                                    'Stop': ema200, 'Expiration': entry_candle.name + pd.Timedelta(days=option_expiration_days),
                                    'Contracts': num_contracts, 'Premium': proxy_premium
                                }
                
                # Put Signal: Wait for close of breakout candle
                elif trend_direction == "Down" and is_not_at_atl(daily_data, current_ts.date()):
                    if prev_close > ema200 and current_row['Close'] < ema200:
                        sell_signals_dates.append(current_ts)
                        entry_candle = four_hour_data.iloc[i+1]
                        entry_price = entry_candle['Open']

                        # Dynamic Premium Proxy
                        proxy_premium = entry_price * 0.03 # 3% of stock price as premium proxy
                        if proxy_premium < max_option_premium:
                            num_contracts = position_size // (proxy_premium * 100)
                            if num_contracts > 0:
                                in_trade = True
                                current_trade = {
                                    'Type': 'Put', 'Entry Date': entry_candle.name, 'Entry Price': entry_price,
                                    'Stop': ema200, 'Expiration': entry_candle.name + pd.Timedelta(days=option_expiration_days),
                                    'Contracts': num_contracts, 'Premium': proxy_premium
                                }

        # --- Charting ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=four_hour_data.index,
                                 open=four_hour_data['Open'], high=four_hour_data['High'],
                                 low=four_hour_data['Low'], close=four_hour_data['Close'],
                                 name='Candlestick'))
    
        fig.add_trace(go.Scatter(x=four_hour_data.index, y=four_hour_data['EMA50'], 
                             line=dict(color='blue', width=1, dash='dot'), name='50 EMA'))
        fig.add_trace(go.Scatter(x=four_hour_data.index, y=four_hour_data['EMA200'], 
                             line=dict(color='orange', width=2), name='200 EMA'))
    
        if buy_signals_dates:
            fig.add_trace(go.Scatter(x=buy_signals_dates, y=four_hour_data.loc[buy_signals_dates]['Close'],
                                 mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                                 name='Buy Signal'))
        if sell_signals_dates:
            fig.add_trace(go.Scatter(x=sell_signals_dates, y=four_hour_data.loc[sell_signals_dates]['Close'],
                                 mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                                 name='Sell Signal'))

        fig.update_layout(title=f'{ticker} 4-Hour Chart with Signals', xaxis_title='Date', yaxis_title='Price',
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    # --- Display Results ---
    st.header("Backtest Results")
    
    if not all_trades:
        st.warning("No trades were executed during the backtest period.")
    else:
        total_pnl = sum(float(trade['Options P/L'].replace('$', '')) for trade in all_trades)
        final_portfolio_value = start_budget + total_pnl
        
        wins = [t for t in all_trades if float(t['Options P/L'].replace('$', '')) > 0]
        win_rate = len(wins) / len(all_trades) if all_trades else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Starting Budget", f"${start_budget:,.2f}")
        col2.metric("Final Portfolio Value", f"${final_portfolio_value:,.2f}", delta=f"{total_pnl:,.2f}")
        col3.metric("Total Trades", len(all_trades))
        col4.metric("Win Rate", f"{win_rate:.2%}")

        st.subheader("All Trades")
        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df[[
            "Ticker", "Strategy", "Entry Date", "Exit Date", "Exit Reason", 
            "Entry Price", "Exit Price", "Contracts", "Entry Premium", "Options P/L"
        ]]
        st.dataframe(trades_df)

else:
    st.info("Please configure your backtest on the left and click 'Run Backtest'.")