#!/usr/bin/env python3
"""
INNOVATIVE AI TRADING BOT - Scalping dengan Compounding 3%
"""

import asyncio
import aiohttp
import ccxt.async_support as ccxt
import json
import time
import hashlib
import hmac
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import ML modules
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import joblib
import pickle

# Import config
from bot_config import BotConfig

class AITradingBot:
    """Bot Trading AI dengan Machine Learning"""
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Inisialisasi bot trading
        
        Args:
            api_key: API Key Indodax
            secret_key: Secret Key Indodax
        """
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Konfigurasi
        self.config = BotConfig()
        
        # Inisialisasi Exchange
        self.exchange = ccxt.indodax({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
        
        # State variables
        self.running = True
        self.daily_profit = 0.0
        self.daily_trades = 0
        self.total_profit = 0.0
        self.active_trades = {}  # {trade_id: trade_info}
        self.trade_history = []
        
        # Machine Learning Models
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Data collectors
        self.market_data = {}
        self.ohlcv_data = {}
        self.technical_indicators = {}
        
        # Performance tracking
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'successful_trades': 0
        }
        
        # Load atau train models
        self.load_or_train_models()
        
        print(f"[{datetime.now()}] 🤖 AI Trading Bot Initialized")
        print(f"💰 Target Profit: {self.config.DAILY_TARGET_PROFIT*100}% per day")
        print(f"🎯 Max Concurrent Trades: {self.config.MAX_CONCURRENT_TRADES}")
        print(f"🧠 ML Model: {self.config.ML_MODEL_TYPE}")
    
    def load_or_train_models(self):
        """Load model ML yang sudah ada atau train baru"""
        try:
            # Coba load model yang sudah disimpan
            with open('ml_models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('ml_models/feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            # Load model ensemble
            model_files = {
                'rf': 'ml_models/random_forest.pkl',
                'xgb': 'ml_models/xgboost.pkl',
                'gb': 'ml_models/gradient_boosting.pkl'
            }
            
            for model_name, model_file in model_files.items():
                try:
                    with open(model_file, 'rb') as f:
                        self.ml_models[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name} model")
                except:
                    print(f"⚠ {model_name} model not found")
            
            if not self.ml_models:
                print("⚠ No ML models found, training new models...")
                self.train_initial_models()
        
        except FileNotFoundError:
            print("⚠ ML models not found, training new models...")
            self.train_initial_models()
    
    def train_initial_models(self):
        """Train initial ML models dengan data historis"""
        print("🔧 Training initial ML models...")
        
        # Simulasi data training (dalam implementasi real, ambil dari database)
        # Ini adalah placeholder - di production akan menggunakan data real
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic training data
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)  # 0: don't trade, 1: trade
        
        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.ml_models['rf'] = rf_model
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.ml_models['xgb'] = xgb_model
        
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.ml_models['gb'] = gb_model
        
        # Save models
        import os
        os.makedirs('ml_models', exist_ok=True)
        
        with open('ml_models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open('ml_models/feature_columns.pkl', 'wb') as f:
            pickle.dump([f'feature_{i}' for i in range(n_features)], f)
        
        for model_name, model in self.ml_models.items():
            with open(f'ml_models/{model_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        print("✅ ML models trained and saved")
    
    def calculate_technical_indicators(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Menghitung indikator teknis untuk analisis
        
        Args:
            ohlcv_df: DataFrame dengan kolom ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame dengan indikator tambahan
        """
        df = ohlcv_df.copy()
        
        # Moving Averages
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price changes
        df['price_change_1m'] = df['close'].pct_change(1)
        df['price_change_5m'] = df['close'].pct_change(5)
        df['price_change_15m'] = df['close'].pct_change(15)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        # Support Resistance
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def extract_features(self, symbol: str, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        Ekstrak features untuk model ML
        
        Args:
            symbol: Trading pair
            ohlcv_data: OHLCV data dengan indikator
        
        Returns:
            Array features untuk prediction
        """
        if ohlcv_data.empty or len(ohlcv_data) < 50:
            return None
        
        # Ambil data terbaru
        latest = ohlcv_data.iloc[-1]
        
        # Price features
        features = []
        
        # Price momentum
        features.append(latest['price_change_1m'])
        features.append(latest['price_change_5m'])
        features.append(latest['price_change_15m'])
        
        # RSI features
        features.append(latest['RSI'])
        features.append(1 if latest['RSI'] > 70 else (-1 if latest['RSI'] < 30 else 0))
        
        # MACD features
        features.append(latest['MACD'])
        features.append(latest['MACD_hist'])
        features.append(1 if latest['MACD'] > latest['MACD_signal'] else -1)
        
        # Moving Average features
        features.append(latest['close'] / latest['SMA_10'] - 1)
        features.append(latest['close'] / latest['SMA_30'] - 1)
        features.append(1 if latest['SMA_10'] > latest['SMA_30'] else -1)
        
        # Bollinger Bands
        bb_position = (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
        features.append(bb_position)
        features.append(latest['BB_width'])
        
        # Volume features
        features.append(latest['volume_ratio'])
        
        # Volatility
        features.append(latest['volatility'])
        
        # Market structure
        features.append(latest['higher_high'])
        features.append(latest['higher_low'])
        features.append(latest['lower_high'])
        features.append(latest['lower_low'])
        
        # Support Resistance
        resistance_distance = (latest['resistance'] - latest['close']) / latest['close']
        support_distance = (latest['close'] - latest['support']) / latest['close']
        features.append(resistance_distance)
        features.append(support_distance)
        
        # Time features
        hour = datetime.now().hour
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        return np.array(features).reshape(1, -1)
    
    def predict_trade_signal(self, symbol: str, ohlcv_data: pd.DataFrame) -> Dict:
        """
        Predict trade signal menggunakan ML models
        
        Args:
            symbol: Trading pair
            ohlcv_data: OHLCV data
        
        Returns:
            Dictionary dengan signal dan confidence
        """
        try:
            # Ekstrak features
            features = self.extract_features(symbol, ohlcv_data)
            if features is None:
                return {'signal': 'HOLD', 'confidence': 0.0}
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Ensemble prediction
            predictions = []
            confidences = []
            
            for model_name, model in self.ml_models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    prediction = model.predict(features_scaled)[0]
                    
                    # Confidence adalah probability dari predicted class
                    confidence = proba[prediction]
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
            
            # Voting ensemble
            if len(predictions) > 0:
                # Simple majority voting
                buy_votes = sum(1 for p in predictions if p == 1)
                sell_votes = sum(1 for p in predictions if p == 2)
                hold_votes = sum(1 for p in predictions if p == 0)
                
                # Average confidence
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                # Determine signal
                if buy_votes > sell_votes and buy_votes > hold_votes and avg_confidence > self.config.PREDICTION_CONFIDENCE:
                    return {'signal': 'BUY', 'confidence': avg_confidence, 'votes': {'BUY': buy_votes, 'SELL': sell_votes, 'HOLD': hold_votes}}
                elif sell_votes > buy_votes and sell_votes > hold_votes and avg_confidence > self.config.PREDICTION_CONFIDENCE:
                    return {'signal': 'SELL', 'confidence': avg_confidence, 'votes': {'BUY': buy_votes, 'SELL': sell_votes, 'HOLD': hold_votes}}
                else:
                    return {'signal': 'HOLD', 'confidence': avg_confidence, 'votes': {'BUY': buy_votes, 'SELL': sell_votes, 'HOLD': hold_votes}}
        
        except Exception as e:
            print(f"❌ Error in ML prediction for {symbol}: {e}")
        
        return {'signal': 'HOLD', 'confidence': 0.0}
    
    async def scan_markets(self) -> List[Dict]:
        """
        Scan semua market untuk peluang trading
        
        Returns:
            List of trading opportunities
        """
        opportunities = []
        
        try:
            # Load markets
            markets = await self.exchange.load_markets()
            
            # Filter untuk pairs IDR
            idr_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/IDR')]
            
            print(f"🔍 Scanning {len(idr_pairs)} pairs...")
            
            for symbol in idr_pairs[:50]:  # Batasi scan untuk performance
                try:
                    # Skip jika sudah ada active trade untuk pair ini
                    active_for_symbol = any(
                        trade['symbol'] == symbol for trade in self.active_trades.values()
                    )
                    if active_for_symbol:
                        continue
                    
                    # Fetch OHLCV data
                    ohlcv = await self.exchange.fetch_ohlcv(
                        symbol, 
                        timeframe='1m', 
                        limit=100
                    )
                    
                    if len(ohlcv) < 50:  # Minimal data
                        continue
                    
                    # Convert ke DataFrame
                    df = pd.DataFrame(
                        ohlcv, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # Calculate technical indicators
                    df = self.calculate_technical_indicators(df)
                    
                    # Get current ticker
                    ticker = await self.exchange.fetch_ticker(symbol)
                    
                    # Filter berdasarkan kriteria
                    current_price = ticker['last']
                    volume_24h = ticker['quoteVolume']
                    
                    # Kriteria screening
                    price_ok = (self.config.MIN_COIN_PRICE <= current_price <= self.config.MAX_COIN_PRICE)
                    volume_ok = (volume_24h >= self.config.MIN_VOLUME_THRESHOLD)
                    
                    # Calculate volatility
                    volatility = df['close'].pct_change().std() * np.sqrt(365*24*60)  # Annualized
                    volatility_ok = volatility >= self.config.VOLATILITY_THRESHOLD
                    
                    if price_ok and volume_ok and volatility_ok:
                        # ML Prediction
                        prediction = self.predict_trade_signal(symbol, df)
                        
                        if prediction['signal'] in ['BUY', 'SELL']:
                            opportunities.append({
                                'symbol': symbol,
                                'price': current_price,
                                'volume_24h': volume_24h,
                                'volatility': volatility,
                                'signal': prediction['signal'],
                                'confidence': prediction['confidence'],
                                'votes': prediction.get('votes', {}),
                                'timestamp': datetime.now().isoformat()
                            })
                
                except Exception as e:
                    print(f"⚠ Error scanning {symbol}: {e}")
                    continue
            
            # Sort opportunities by confidence
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)
            
            print(f"✅ Found {len(opportunities)} opportunities")
            
        except Exception as e:
            print(f"❌ Error in market scan: {e}")
        
        return opportunities
    
    def calculate_position_size(self, symbol: str, price: float, account_balance: Dict) -> Dict:
        """
        Calculate position size berdasarkan risk management
        
        Args:
            symbol: Trading pair
            price: Entry price
            account_balance: Account balance dictionary
        
        Returns:
            Position size details
        """
        try:
            # Get available balance
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            
            # Available balance untuk quote currency (IDR)
            available_idr = float(account_balance.get('idr', 0))
            
            # Compounding: sesuaikan dengan daily target
            if self.config.COMPOUNDING_ENABLED and self.daily_profit > 0:
                # Tambahkan profit hari ini ke modal
                effective_capital = available_idr * (1 + min(self.daily_profit, self.config.DAILY_TARGET_PROFIT))
            else:
                effective_capital = available_idr
            
            # Risk per trade (maksimal 2% dari modal)
            risk_per_trade = effective_capital * self.config.MAX_RISK_PER_TRADE
            
            # Position size berdasarkan stop loss
            position_size_idr = risk_per_trade / self.config.STOP_LOSS_PERCENT
            
            # Adjust untuk maksimal concurrent trades
            max_per_trade = effective_capital / self.config.MAX_CONCURRENT_TRADES
            position_size_idr = min(position_size_idr, max_per_trade)
            
            # Minimal position size
            min_position = 50000  # Minimal 50k IDR
            if position_size_idr < min_position:
                return None
            
            # Calculate coin amount
            coin_amount = position_size_idr / price
            
            return {
                'position_size_idr': position_size_idr,
                'coin_amount': coin_amount,
                'risk_amount': risk_per_trade,
                'stop_loss': price * (1 - self.config.STOP_LOSS_PERCENT),
                'take_profit': price * (1 + self.config.TAKE_PROFIT_PERCENT),
                'risk_reward_ratio': self.config.RISK_REWARD_RATIO
            }
        
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            return None
    
    async def execute_trade(self, opportunity: Dict):
        """
        Execute trading order
        
        Args:
            opportunity: Trading opportunity
        """
        try:
            symbol = opportunity['symbol']
            signal = opportunity['signal']
            
            # Check jika sudah mencapai daily target
            if self.daily_profit >= self.config.DAILY_TARGET_PROFIT:
                print(f"✅ Daily target reached ({self.daily_profit*100:.2f}%), skipping trade")
                return
            
            # Check max concurrent trades
            if len(self.active_trades) >= self.config.MAX_CONCURRENT_TRADES:
                print(f"⚠ Max concurrent trades reached ({self.config.MAX_CONCURRENT_TRADES})")
                return
            
            # Get account balance
            balance = await self.exchange.fetch_balance()
            
            # Calculate position size
            position = self.calculate_position_size(
                symbol, 
                opportunity['price'], 
                balance['free']
            )
            
            if not position:
                print(f"⚠ Position size too small for {symbol}")
                return
            
            # Prepare order
            if signal == 'BUY':
                order_type = 'limit'
                side = 'buy'
                amount = position['coin_amount']
                price = opportunity['price'] * 0.999  # Slightly below market for better fill
            else:  # SELL
                # Check if we have the coin
                base_currency = symbol.split('/')[0]
                available_coin = float(balance['free'].get(base_currency, 0))
                
                if available_coin < position['coin_amount']:
                    print(f"⚠ Insufficient {base_currency} for sell order")
                    return
                
                order_type = 'limit'
                side = 'sell'
                amount = position['coin_amount']
                price = opportunity['price'] * 1.001  # Slightly above market
            
            # Execute order
            print(f"🎯 Executing {side.upper()} order for {symbol}")
            print(f"   Amount: {amount:.8f} | Price: {price:,.0f} IDR")
            print(f"   Position Size: {position['position_size_idr']:,.0f} IDR")
            print(f"   Stop Loss: {position['stop_loss']:,.0f} IDR")
            print(f"   Take Profit: {position['take_profit']:,.0f} IDR")
            
            try:
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price
                )
                
                # Record trade
                trade_id = order['id']
                self.active_trades[trade_id] = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': price,
                    'amount': amount,
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit'],
                    'entry_time': datetime.now(),
                    'order_id': trade_id,
                    'status': 'OPEN',
                    'position_size': position['position_size_idr']
                }
                
                print(f"✅ Order executed: {trade_id}")
                
                # Increment daily trades counter
                self.daily_trades += 1
                
                # Schedule trade monitoring
                asyncio.create_task(self.monitor_trade(trade_id))
                
            except Exception as e:
                print(f"❌ Order execution failed: {e}")
        
        except Exception as e:
            print(f"❌ Error in execute_trade: {e}")
    
    async def monitor_trade(self, trade_id: str):
        """
        Monitor active trade untuk stop loss dan take profit
        
        Args:
            trade_id: ID trade yang aktif
        """
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return
            
            symbol = trade['symbol']
            side = trade['side']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']
            
            print(f"👁 Monitoring trade {trade_id} for {symbol}")
            
            # Monitor selama maksimal holding period
            start_time = datetime.now()
            max_duration = timedelta(minutes=self.config.HOLDING_PERIOD_MAX)
            
            while (datetime.now() - start_time) < max_duration:
                try:
                    # Get current ticker
                    ticker = await self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Check stop loss
                    if side == 'buy' and current_price <= stop_loss:
                        print(f"🛑 STOP LOSS triggered for {symbol}")
                        await self.close_trade(trade_id, current_price, 'STOP_LOSS')
                        break
                    elif side == 'sell' and current_price >= stop_loss:
                        print(f"🛑 STOP LOSS triggered for {symbol}")
                        await self.close_trade(trade_id, current_price, 'STOP_LOSS')
                        break
                    
                    # Check take profit
                    if side == 'buy' and current_price >= take_profit:
                        print(f"🎯 TAKE PROFIT triggered for {symbol}")
                        await self.close_trade(trade_id, current_price, 'TAKE_PROFIT')
                        break
                    elif side == 'sell' and current_price <= take_profit:
                        print(f"🎯 TAKE PROFIT triggered for {symbol}")
                        await self.close_trade(trade_id, current_price, 'TAKE_PROFIT')
                        break
                    
                    # Check trailing stop (opsional)
                    # Implement trailing stop logic di sini
                    
                    # Wait sebelum check lagi
                    await asyncio.sleep(5)  # Check setiap 5 detik
                
                except Exception as e:
                    print(f"⚠ Error monitoring trade {trade_id}: {e}")
                    await asyncio.sleep(10)
            
            # Force close jika sudah melewati holding period
            if trade_id in self.active_trades:
                print(f"⏰ Holding period expired for {symbol}, closing trade")
                ticker = await self.exchange.fetch_ticker(symbol)
                await self.close_trade(trade_id, ticker['last'], 'TIME_EXPIRED')
        
        except Exception as e:
            print(f"❌ Error in monitor_trade: {e}")
    
    async def close_trade(self, trade_id: str, close_price: float, reason: str):
        """
        Close trade dan hitung profit/loss
        
        Args:
            trade_id: ID trade
            close_price: Close price
            reason: Reason for closing
        """
        try:
            if trade_id not in self.active_trades:
                return
            
            trade = self.active_trades[trade_id]
            symbol = trade['symbol']
            side = trade['side']
            entry_price = trade['entry_price']
            amount = trade['amount']
            position_size = trade['position_size']
            
            # Calculate P&L
            if side == 'buy':
                pl_percent = (close_price - entry_price) / entry_price
                pl_idr = (close_price - entry_price) * amount
            else:  # sell
                pl_percent = (entry_price - close_price) / entry_price
                pl_idr = (entry_price - close_price) * amount
            
            # Update daily profit
            self.daily_profit += pl_idr / position_size if position_size > 0 else 0
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            if pl_idr > 0:
                self.performance_metrics['successful_trades'] += 1
            
            # Update win rate
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['successful_trades'] / 
                max(1, self.performance_metrics['total_trades'])
            )
            
            # Record trade history
            trade_history = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'close_price': close_price,
                'amount': amount,
                'pnl_percent': pl_percent,
                'pnl_idr': pl_idr,
                'entry_time': trade['entry_time'],
                'close_time': datetime.now(),
                'close_reason': reason,
                'position_size': position_size
            }
            
            self.trade_history.append(trade_history)
            
            # Remove from active trades
            del self.active_trades[trade_id]
            
            # Print trade result
            result_color = "🟢" if pl_idr > 0 else "🔴" if pl_idr < 0 else "⚪"
            print(f"\n{result_color} TRADE CLOSED: {symbol}")
            print(f"   Side: {side.upper()} | Reason: {reason}")
            print(f"   Entry: {entry_price:,.0f} | Close: {close_price:,.0f}")
            print(f"   P&L: {pl_idr:+,.0f} IDR ({pl_percent*100:+.2f}%)")
            print(f"   Daily Profit: {self.daily_profit*100:.2f}%")
            print(f"   Win Rate: {self.performance_metrics['win_rate']*100:.2f}%")
            
            # Update ML model dengan hasil trade
            await self.update_ml_model(trade_history)
            
            # Check jika sudah mencapai daily target
            if self.daily_profit >= self.config.DAILY_TARGET_PROFIT:
                print(f"\n🎉 DAILY TARGET ACHIEVED: {self.daily_profit*100:.2f}%")
                print("   Stopping new trades for today...")
            
        except Exception as e:
            print(f"❌ Error closing trade {trade_id}: {e}")
    
    async def update_ml_model(self, trade_history: Dict):
        """
        Update ML model dengan hasil trade terbaru
        
        Args:
            trade_history: Data hasil trade
        """
        try:
            # Ini adalah contoh sederhana
            # Di implementasi real, Anda akan:
            # 1. Simpan trade data ke database
            # 2. Retrain model periodically dengan data baru
            # 3. Implement online learning
            
            print(f"🧠 Updating ML model with trade result...")
            
            # Untuk sekarang, hanya log ke file
            with open('trade_log.json', 'a') as f:
                json.dump(trade_history, f)
                f.write('\n')
            
            # Periodic retraining bisa dilakukan di sini
            # Misalnya setiap 100 trade, retrain model
            
        except Exception as e:
            print(f"⚠ Error updating ML model: {e}")
    
    async def run_trading_cycle(self):
        """Satu siklus trading lengkap"""
        try:
            print(f"\n{'='*60}")
            print(f"🔄 TRADING CYCLE START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"💰 Daily Profit: {self.daily_profit*100:.2f}% | Target: {self.config.DAILY_TARGET_PROFIT*100}%")
            print(f"📊 Active Trades: {len(self.active_trades)}/{self.config.MAX_CONCURRENT_TRADES}")
            print(f"📈 Win Rate: {self.performance_metrics['win_rate']*100:.2f}%")
            print(f"{'='*60}")
            
            # Skip jika sudah mencapai target
            if self.daily_profit >= self.config.DAILY_TARGET_PROFIT:
                print("✅ Daily target already achieved, waiting...")
                return
            
            # Scan markets untuk opportunities
            opportunities = await self.scan_markets()
            
            # Filter dan eksekusi trades
            for opp in opportunities[:3]:  # Ambil 3 terbaik
                # Check confidence threshold
                if opp['confidence'] < self.config.PREDICTION_CONFIDENCE:
                    continue
                
                # Check jika pair ini sudah ada di active trades
                symbol_exists = any(
                    trade['symbol'] == opp['symbol'] for trade in self.active_trades.values()
                )
                if symbol_exists:
                    continue
                
                # Execute trade
                await self.execute_trade(opp)
                
                # Delay antara trades
                await asyncio.sleep(1)
        
        except Exception as e:
            print(f"❌ Error in trading cycle: {e}")
    
    async def run(self):
        """Main loop bot"""
        print("\n🚀 Starting AI Trading Bot...")
        
        try:
            # Test connection
            balance = await self.exchange.fetch_balance()
            total_idr = float(balance['IDR']['total']) if 'IDR' in balance else 0
            print(f"💰 Account Balance: {total_idr:,.0f} IDR")
            
            # Main trading loop
            while self.running:
                try:
                    # Run trading cycle
                    await self.run_trading_cycle()
                    
                    # Wait sebelum cycle berikutnya
                    wait_time = 30  # detik
                    print(f"\n⏳ Next scan in {wait_time} seconds...")
                    
                    for i in range(wait_time):
                        if not self.running:
                            break
                        await asyncio.sleep(1)
                        
                        # Print progress setiap 10 detik
                        if i % 10 == 0:
                            active_count = len(self.active_trades)
                            print(f"   Active trades: {active_count} | Daily P&L: {self.daily_profit*100:.2f}%")
                
                except KeyboardInterrupt:
                    print("\n🛑 Stopping bot...")
                    self.running = False
                    break
                
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 menit jika error
        
        finally:
            # Cleanup
            await self.exchange.close()
            print("✅ Bot stopped successfully")
    
    def stop(self):
        """Stop bot trading"""
        self.running = False
        print("\n🛑 Stopping bot...")


async def main():
    """Main function"""
    print("="*70)
    print("🤖 INNOVATIVE AI TRADING BOT")
    print("   Strategy: Scalping with 3% Daily Compounding")
    print("   Features: Machine Learning, Multi-Asset Scanning")
    print("="*70)
    
    # Load API credentials
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
            api_key = config['api_key']
            secret_key = config['secret_key']
    except FileNotFoundError:
        print("\n⚠ config.json not found!")
        print("Please create config.json with your API credentials:")
        print('''
{
    "api_key": "YOUR_API_KEY",
    "secret_key": "YOUR_SECRET_KEY"
}
        ''')
        return
    
    # Create and run bot
    bot = AITradingBot(api_key, secret_key)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()


if __name__ == "__main__":
    # Run bot
    asyncio.run(main())