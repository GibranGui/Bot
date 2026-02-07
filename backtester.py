#!/usr/bin/env python3
"""
BACKTESTING ENGINE - Untuk testing strategi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class Backtester:
    """Engine untuk backtesting strategi trading"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Inisialisasi backtester"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.backtest_config = self.config.get('backtesting', {})
        self.initial_capital = self.backtest_config.get('initial_capital', 1000000)
        self.commission_rate = self.backtest_config.get('commission_rate', 0.002)
        self.slippage = self.backtest_config.get('slippage', 0.001)
        
        self.results = {}
        self.trades = []
        
    def load_historical_data(self, symbol: str, period: str = '30d') -> pd.DataFrame:
        """
        Load data historis untuk backtesting
        
        Args:
            symbol: Trading pair
            period: Periode data (30d, 90d, 1y)
        
        Returns:
            DataFrame dengan data OHLCV
        """
        # Di implementasi real, ini akan mengambil data dari database atau API
        # Untuk contoh, kita generate synthetic data
        
        periods_map = {
            '30d': 30 * 24 * 60,  # 1 menit interval
            '90d': 90 * 24 * 60,
            '1y': 365 * 24 * 60
        }
        
        n_periods = periods_map.get(period, 30 * 24 * 60)
        
        # Generate synthetic price data (Geometric Brownian Motion)
        np.random.seed(42)
        dt = 1 / (24 * 60)  # 1 menit dalam hari
        mu = 0.0002  # Expected return per menit
        sigma = 0.005  # Volatility per menit
        
        # Generate random walk
        price_changes = np.random.normal((mu - 0.5 * sigma**2) * dt, 
                                        sigma * np.sqrt(dt), n_periods)
        prices = 1000000 * np.exp(np.cumsum(price_changes))
        
        # Create OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq='1min')
        
        df = pd.DataFrame(index=dates)
        df['open'] = prices * np.random.uniform(0.999, 1.001, n_periods)
        df['high'] = df['open'] * np.random.uniform(1.001, 1.005, n_periods)
        df['low'] = df['open'] * np.random.uniform(0.995, 0.999, n_periods)
        df['close'] = df['open'] * np.random.uniform(0.998, 1.002, n_periods)
        df['volume'] = np.random.lognormal(10, 1, n_periods)
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hitung indikator teknis untuk backtesting"""
        df = df.copy()
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df.fillna(method='bfill')
    
    def generate_signals(self, df: pd.DataFrame, strategy: str = 'ml') -> pd.Series:
        """
        Generate trading signals
        
        Args:
            df: DataFrame dengan indikator
            strategy: 'ml', 'momentum', 'mean_reversion', 'breakout'
        
        Returns:
            Series dengan signals (1=buy, -1=sell, 0=hold)
        """
        if strategy == 'momentum':
            # Momentum strategy
            returns = df['close'].pct_change(10)
            signals = np.where(returns > 0.02, 1, 
                              np.where(returns < -0.02, -1, 0))
            
        elif strategy == 'mean_reversion':
            # Mean reversion strategy
            sma_ratio = df['close'] / df['sma_30']
            signals = np.where(sma_ratio < 0.98, 1,
                              np.where(sma_ratio > 1.02, -1, 0))
            
        elif strategy == 'breakout':
            # Breakout strategy
            high_20 = df['high'].rolling(20).max()
            low_20 = df['low'].rolling(20).min()
            
            signals = np.zeros(len(df))
            signals[df['close'] > high_20.shift(1)] = 1
            signals[df['close'] < low_20.shift(1)] = -1
            
        else:  # ML-like strategy
            # Simulate ML predictions
            signals = np.zeros(len(df))
            
            # Buy when RSI < 30 and price below lower BB
            buy_condition = (df['rsi'] < 30) & (df['close'] < df['bb_lower'])
            signals[buy_condition] = 1
            
            # Sell when RSI > 70 and price above upper BB
            sell_condition = (df['rsi'] > 70) & (df['close'] > df['bb_upper'])
            signals[sell_condition] = -1
        
        return pd.Series(signals, index=df.index)
    
    def run_backtest(self, symbol: str, strategy: str = 'ml', 
                    initial_capital: float = None) -> Dict:
        """
        Run backtest untuk strategi tertentu
        
        Returns:
            Dictionary dengan hasil backtest
        """
        if initial_capital is None:
            initial_capital = self.initial_capital
        
        # Load data
        df = self.load_historical_data(symbol)
        df = self.calculate_indicators(df)
        
        # Generate signals
        signals = self.generate_signals(df, strategy)
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            signal = signals.iloc[i]
            
            # Check for entry
            if signal == 1 and position == 0:
                # Buy signal
                position_size = capital * 0.2  # 20% per trade
                shares = position_size / current_price
                commission = position_size * self.commission_rate
                
                capital -= position_size + commission
                position = shares
                
                trades.append({
                    'timestamp': df.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': position_size,
                    'commission': commission
                })
            
            elif signal == -1 and position > 0:
                # Sell signal
                position_value = position * current_price
                commission = position_value * self.commission_rate
                
                capital += position_value - commission
                
                trades.append({
                    'timestamp': df.index[i],
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': position_value,
                    'commission': commission
                })
                
                position = 0
            
            # Calculate equity
            position_value = position * current_price if position > 0 else 0
            equity = capital + position_value
            equity_curve.append(equity)
        
        # Close any open position at the end
        if position > 0:
            position_value = position * df['close'].iloc[-1]
            commission = position_value * self.commission_rate
            capital += position_value - commission
            
            trades.append({
                'timestamp': df.index[-1],
                'type': 'SELL',
                'price': df['close'].iloc[-1],
                'shares': position,
                'value': position_value,
                'commission': commission
            })
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve, index=df.index[1:])
        returns = equity_series.pct_change().fillna(0)
        
        total_return = (equity_series.iloc[-1] / initial_capital - 1)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(equity_series)
        win_rate = self.calculate_win_rate(trades)
        
        # Store results
        self.results[symbol] = {
            'initial_capital': initial_capital,
            'final_capital': equity_series.iloc[-1],
            'total_return': total_return,
            'total_trades': len(trades) // 2,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': trades,
            'equity_curve': equity_series.tolist()
        }
        
        return self.results[symbol]
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown.min()
    
    def calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate dari trades"""
        if len(trades) < 2:
            return 0.0
        
        # Group trades in pairs (buy-sell)
        winning_trades = 0
        total_pairs = len(trades) // 2
        
        for i in range(0, len(trades) - 1, 2):
            buy_trade = trades[i]
            sell_trade = trades[i + 1]
            
            buy_price = buy_trade['price']
            sell_price = sell_trade['price']
            
            if sell_price > buy_price:
                winning_trades += 1
        
        return winning_trades / total_pairs if total_pairs > 0 else 0.0
    
    def generate_report(self, symbol: str = None):
        """Generate backtest report"""
        if symbol and symbol in self.results:
            result = self.results[symbol]
        elif self.results:
            # Use first symbol if none specified
            result = list(self.results.values())[0]
        else:
            print("No backtest results available")
            return
        
        print("="*70)
        print("BACKTEST REPORT")
        print("="*70)
        print(f"Initial Capital: {result['initial_capital']:,.0f} IDR")
        print(f"Final Capital: {result['final_capital']:,.0f} IDR")
        print(f"Total Return: {result['total_return']*100:.2f}%")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Win Rate: {result['win_rate']*100:.2f}%")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print("="*70)
        
        # Plot equity curve
        self.plot_equity_curve(result['equity_curve'])
        
        # Plot trades
        if result['trades']:
            self.plot_trades(result['trades'])
    
    def plot_equity_curve(self, equity_curve: List[float]):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label='Equity', linewidth=2)
        plt.title('Equity Curve', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('Equity (IDR)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('backtest_equity_curve.png', dpi=100)
        plt.show()
    
    def plot_trades(self, trades: List[Dict]):
        """Visualize trades"""
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        buy_times = [t['timestamp'] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]
        
        sell_times = [t['timestamp'] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(buy_times, buy_prices, color='green', s=100, 
                   label='Buy', marker='^', alpha=0.8)
        plt.scatter(sell_times, sell_prices, color='red', s=100,
                   label='Sell', marker='v', alpha=0.8)
        
        # Connect buy-sell pairs
        for i in range(min(len(buy_trades), len(sell_trades))):
            plt.plot([buy_times[i], sell_times[i]], 
                    [buy_prices[i], sell_prices[i]], 
                    'k--', alpha=0.3)
        
        plt.title('Trade Signals', fontsize=14)
        plt.xlabel('Time')
        plt.ylabel('Price (IDR)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('backtest_trades.png', dpi=100)
        plt.show()
    
    def compare_strategies(self, symbol: str, strategies: List[str]):
        """Bandingkan beberapa strategi"""
        comparison = {}
        
        for strategy in strategies:
            print(f"\nRunning backtest for {strategy} strategy...")
            result = self.run_backtest(symbol, strategy)
            comparison[strategy] = result
        
        # Create comparison table
        df_comparison = pd.DataFrame({
            'Strategy': strategies,
            'Total Return': [comparison[s]['total_return']*100 for s in strategies],
            'Sharpe Ratio': [comparison[s]['sharpe_ratio'] for s in strategies],
            'Max Drawdown': [comparison[s]['max_drawdown']*100 for s in strategies],
            'Win Rate': [comparison[s]['win_rate']*100 for s in strategies],
            'Total Trades': [comparison[s]['total_trades'] for s in strategies]
        })
        
        print("\n" + "="*70)
        print("STRATEGY COMPARISON")
        print("="*70)
        print(df_comparison.to_string(index=False))
        
        # Plot comparison
        self.plot_strategy_comparison(comparison)
        
        return comparison
    
    def plot_strategy_comparison(self, comparison: Dict):
        """Plot perbandingan strategi"""
        strategies = list(comparison.keys())
        returns = [comparison[s]['total_return']*100 for s in strategies]
        sharpe = [comparison[s]['sharpe_ratio'] for s in strategies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Total Return
        bars1 = ax1.bar(strategies, returns, color=['green', 'blue', 'orange'])
        ax1.set_title('Total Return (%)', fontsize=14)
        ax1.set_ylabel('Return %')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Sharpe Ratio
        bars2 = ax2.bar(strategies, sharpe, color=['green', 'blue', 'orange'])
        ax2.set_title('Sharpe Ratio', fontsize=14)
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('strategy_comparison.png', dpi=100)
        plt.show()


def run_backtest_example():
    """Contoh penggunaan backtester"""
    backtester = Backtester()
    
    # Run single strategy backtest
    print("Running single strategy backtest...")
    result = backtester.run_backtest('BTC/IDR', strategy='ml')
    backtester.generate_report('BTC/IDR')
    
    # Compare multiple strategies
    print("\n" + "="*70)
    print("Comparing multiple strategies...")
    strategies = ['ml', 'momentum', 'mean_reversion', 'breakout']
    comparison = backtester.compare_strategies('BTC/IDR', strategies)
    
    return backtester


if __name__ == "__main__":
    run_backtest_example()