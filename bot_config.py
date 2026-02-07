"""
CONFIGURASI BOT TRADING AI
"""

class BotConfig:
    """Konfigurasi utama bot trading"""
    
    # ========== TARGET PROFIT ==========
    DAILY_TARGET_PROFIT = 0.03  # 3% per hari
    WEEKLY_TARGET_PROFIT = 0.15  # 15% per minggu
    MONTHLY_TARGET_PROFIT = 0.50  # 50% per bulan
    
    # ========== RISK MANAGEMENT ==========
    MAX_CONCURRENT_TRADES = 5      # Maksimal 5 transaksi paralel
    MAX_RISK_PER_TRADE = 0.02      # Maksimal 2% risk per trade
    MAX_DAILY_RISK = 0.10          # Maksimal 10% risk per hari
    MAX_DRAWDOWN = 0.15            # Maksimal drawdown 15%
    
    # ========== TRADING PARAMETERS ==========
    STOP_LOSS_PERCENT = 0.015      # Stop Loss 1.5%
    TAKE_PROFIT_PERCENT = 0.02     # Take Profit 2%
    TRAILING_STOP_PERCENT = 0.01   # Trailing Stop 1%
    RISK_REWARD_RATIO = 1.5        # Minimal Risk:Reward 1:1.5
    
    # ========== SCALPING SETTINGS ==========
    SCALPING_TIMEFRAME = '1m'      # Timeframe utama
    HOLDING_PERIOD_MAX = 30        # Maksimal 30 menit hold (detik)
    MIN_PROFIT_PERCENT = 0.005     # Minimal profit 0.5%
    
    # ========== ASSET SELECTION ==========
    SCAN_ALL_COINS = True          # Scan semua coin
    MIN_COIN_PRICE = 100           # Minimal harga coin (IDR)
    MAX_COIN_PRICE = 10000000      # Maksimal harga coin (IDR)
    MIN_VOLUME_24H = 1000000       # Minimal volume 24h (IDR)
    VOLATILITY_THRESHOLD = 0.02    # Minimal volatilitas 2%
    
    # ========== MACHINE LEARNING ==========
    ML_MODEL_TYPE = 'ensemble'     # ensemble, neural_net, xgboost
    TRAINING_WINDOW = 500          # Data training window
    PREDICTION_CONFIDENCE = 0.65   # Minimal confidence untuk eksekusi
    MODEL_UPDATE_FREQUENCY = 100   # Update model setiap 100 trades
    
    # ========== TRADING HOURS ==========
    TRADING_HOURS = {
        'start': 9,    # 09:00 WIB
        'end': 17      # 17:00 WIB
    }
    
    # ========== ADVANCED FEATURES ==========
    COMPOUNDING_ENABLED = True     # Mode compounding
    HEDGING_ENABLED = False        # Mode hedging
    GRID_TRADING_ENABLED = False   # Mode grid trading
    DCA_ENABLED = True             # Dollar Cost Averaging
    MARTINGALE_ENABLED = False     # Martingale strategy (HATI-HATI!)
    
    # ========== NOTIFICATIONS ==========
    TELEGRAM_NOTIFICATIONS = True
    DISCORD_NOTIFICATIONS = False
    EMAIL_NOTIFICATIONS = False
    
    # ========== LOGGING ==========
    LOG_LEVEL = 'INFO'             # DEBUG, INFO, WARNING, ERROR
    SAVE_TRADE_LOGS = True
    PERFORMANCE_REPORT_INTERVAL = 3600  # 1 jam
    
    # ========== ADVANCED ML ==========
    FEATURE_ENGINEERING = True
    DEEP_LEARNING = False
    REINFORCEMENT_LEARNING = False
    SENTIMENT_ANALYSIS = False
    
    # ========== BACKTESTING ==========
    BACKTEST_PERIOD = '30d'        # 30 hari
    INITIAL_CAPITAL = 1000000      # 1 juta IDR
    COMMISSION_RATE = 0.002        # 0.2% commission
    

class RiskManagement:
    """Kelas untuk manajemen risiko"""
    
    @staticmethod
    def calculate_kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion untuk position sizing
        
        Formula: f* = (bp - q) / b
        dimana:
          f* = fraksi modal untuk dipertaruhkan
          b = odds (win/loss ratio)
          p = probability menang
          q = probability kalah (1-p)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        return max(0.0, min(kelly, 0.25))  # Batasi maksimal 25%
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sharpe Ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    

class TradingStrategies:
    """Kumpulan strategi trading"""
    
    @staticmethod
    def momentum_strategy(data: pd.DataFrame, lookback: int = 10) -> pd.Series:
        """Strategi momentum sederhana"""
        returns = data['close'].pct_change(lookback)
        signals = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
        return pd.Series(signals, index=data.index)
    
    @staticmethod
    def mean_reversion_strategy(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Strategi mean reversion"""
        sma = data['close'].rolling(lookback).mean()
        std = data['close'].rolling(lookback).std()
        
        z_score = (data['close'] - sma) / std
        signals = np.where(z_score < -1, 1, np.where(z_score > 1, -1, 0))
        return pd.Series(signals, index=data.index)
    
    @staticmethod
    def breakout_strategy(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Strategi breakout"""
        high_max = data['high'].rolling(lookback).max()
        low_min = data['low'].rolling(lookback).min()
        
        signals = np.zeros(len(data))
        signals[data['close'] > high_max.shift(1)] = 1
        signals[data['close'] < low_min.shift(1)] = -1
        
        return pd.Series(signals, index=data.index)
    

class PortfolioOptimizer:
    """Optimasi portfolio dengan Modern Portfolio Theory"""
    
    def __init__(self, returns_data: pd.DataFrame):
        self.returns = returns_data
    
    def efficient_frontier(self, num_portfolios: int = 10000) -> Dict:
        """Calculate efficient frontier"""
        n_assets = len(self.returns.columns)
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Portfolio statistics
            portfolio_return = np.sum(self.returns.mean() * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * 252, weights)))
            
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std
            results[2,i] = (portfolio_return - 0.05) / portfolio_std  # Sharpe ratio
        
        return {
            'returns': results[0],
            'volatility': results[1],
            'sharpe': results[2],
            'weights': weights_record
        }