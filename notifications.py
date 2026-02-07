#!/usr/bin/env python3
"""
NOTIFICATION SYSTEM - Telegram, Discord, Email
"""

import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import logging
from datetime import datetime
from typing import Dict, Optional

class NotificationSystem:
    """Sistem notifikasi multi-channel"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Inisialisasi sistem notifikasi"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.notifications_config = self.config.get('notifications', {})
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def send_telegram(self, message: str, parse_mode: str = 'HTML'):
        """Kirim notifikasi ke Telegram"""
        try:
            telegram_config = self.notifications_config.get('telegram', {})
            if not telegram_config.get('enabled', False):
                return
            
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                self.logger.warning("Telegram credentials not configured")
                return
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                self.logger.info("Telegram notification sent")
            else:
                self.logger.error(f"Telegram error: {response.text}")
        
        except Exception as e:
            self.logger.error(f"Telegram send error: {e}")
    
    def send_discord(self, message: str, embed: Optional[Dict] = None):
        """Kirim notifikasi ke Discord"""
        try:
            discord_config = self.notifications_config.get('discord', {})
            if not discord_config.get('enabled', False):
                return
            
            webhook_url = discord_config.get('webhook_url')
            if not webhook_url:
                self.logger.warning("Discord webhook not configured")
                return
            
            payload = {'content': message}
            
            if embed:
                payload['embeds'] = [embed]
            
            response = requests.post(webhook_url, json=payload)
            if response.status_code in [200, 204]:
                self.logger.info("Discord notification sent")
            else:
                self.logger.error(f"Discord error: {response.text}")
        
        except Exception as e:
            self.logger.error(f"Discord send error: {e}")
    
    def send_email(self, subject: str, body: str, html_body: Optional[str] = None):
        """Kirim notifikasi email"""
        try:
            email_config = self.notifications_config.get('email', {})
            if not email_config.get('enabled', False):
                return
            
            # Setup email parameters
            smtp_server = email_config.get('smtp_server')
            smtp_port = email_config.get('smtp_port')
            email = email_config.get('email')
            password = email_config.get('password')
            recipient = email_config.get('recipient')
            
            if not all([smtp_server, smtp_port, email, password, recipient]):
                self.logger.warning("Email configuration incomplete")
                return
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email
            msg['To'] = recipient
            
            # Attach plain text
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach HTML if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email, password)
                server.send_message(msg)
            
            self.logger.info("Email notification sent")
        
        except Exception as e:
            self.logger.error(f"Email send error: {e}")
    
    def send_trade_signal(self, symbol: str, signal: str, price: float, confidence: float):
        """Kirim notifikasi sinyal trading"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Telegram message
        telegram_msg = f"""
<b>📈 TRADE SIGNAL</b>
──────────────
<b>Pair:</b> {symbol}
<b>Signal:</b> <code>{signal}</code>
<b>Price:</b> {price:,.0f} IDR
<b>Confidence:</b> {confidence:.1%}
<b>Time:</b> {timestamp}
──────────────
<i>AI Trading Bot v1.0</i>
"""
        self.send_telegram(telegram_msg)
        
        # Discord embed
        discord_embed = {
            'title': '📈 Trade Signal',
            'description': f'**Pair:** {symbol}\n**Signal:** {signal}\n**Price:** {price:,.0f} IDR',
            'color': 0x00ff00 if signal == 'BUY' else 0xff0000,
            'fields': [
                {'name': 'Confidence', 'value': f'{confidence:.1%}', 'inline': True},
                {'name': 'Time', 'value': timestamp, 'inline': True}
            ],
            'footer': {'text': 'AI Trading Bot v1.0'}
        }
        self.send_discord(f"Trade Signal: {symbol} {signal}", discord_embed)
        
        # Email
        email_subject = f"Trade Signal: {symbol} {signal}"
        email_body = f"""
Trade Signal Generated:
────────────────────
Pair: {symbol}
Signal: {signal}
Price: {price:,.0f} IDR
Confidence: {confidence:.1%}
Time: {timestamp}
────────────────────
AI Trading Bot v1.0
"""
        self.send_email(email_subject, email_body)
    
    def send_trade_result(self, symbol: str, side: str, pnl_percent: float, 
                         pnl_idr: float, duration: float, reason: str):
        """Kirim notifikasi hasil trade"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration_str = f"{duration:.1f} min"
        
        # Determine color/emoji based on P&L
        if pnl_percent > 0:
            result_emoji = "🟢"
            result_text = "PROFIT"
            color = 0x00ff00
        elif pnl_percent < 0:
            result_emoji = "🔴"
            result_text = "LOSS"
            color = 0xff0000
        else:
            result_emoji = "⚪"
            result_text = "BREAK EVEN"
            color = 0x808080
        
        # Telegram message
        telegram_msg = f"""
{result_emoji} <b>TRADE CLOSED</b>
──────────────
<b>Pair:</b> {symbol}
<b>Side:</b> <code>{side}</code>
<b>Result:</b> {result_text}
<b>P&L:</b> {pnl_idr:+,.0f} IDR ({pnl_percent:+.2f}%)
<b>Duration:</b> {duration_str}
<b>Reason:</b> {reason}
<b>Time:</b> {timestamp}
──────────────
<i>AI Trading Bot v1.0</i>
"""
        self.send_telegram(telegram_msg)
        
        # Discord embed
        discord_embed = {
            'title': f'{result_emoji} Trade Closed',
            'description': f'**Pair:** {symbol}\n**Side:** {side}\n**Result:** {result_text}',
            'color': color,
            'fields': [
                {'name': 'P&L', 'value': f'{pnl_idr:+,.0f} IDR\n({pnl_percent:+.2f}%)', 'inline': True},
                {'name': 'Duration', 'value': duration_str, 'inline': True},
                {'name': 'Reason', 'value': reason, 'inline': False}
            ],
            'footer': {'text': f'AI Trading Bot v1.0 | {timestamp}'}
        }
        self.send_discord(f"Trade Closed: {symbol} {result_text}", discord_embed)
    
    def send_daily_report(self, daily_profit: float, total_trades: int, 
                         win_rate: float, best_trade: Dict, worst_trade: Dict):
        """Kirim laporan harian"""
        timestamp = datetime.now().strftime('%Y-%m-%d')
        
        # Telegram message
        telegram_msg = f"""
<b>📊 DAILY REPORT</b>
──────────────
<b>Date:</b> {timestamp}
<b>Daily Profit:</b> {daily_profit*100:+.2f}%
<b>Total Trades:</b> {total_trades}
<b>Win Rate:</b> {win_rate*100:.1f}%
──────────────
<b>🏆 Best Trade:</b>
{symbol} | {pnl_percent:+.2f}% | {duration} min
──────────────
<b>📉 Worst Trade:</b>
{symbol} | {pnl_percent:+.2f}% | {duration} min
──────────────
<i>AI Trading Bot v1.0</i>
"""
        self.send_telegram(telegram_msg)
        
        # Email report
        email_subject = f"Daily Trading Report - {timestamp}"
        email_body = f"""
DAILY TRADING REPORT
────────────────────
Date: {timestamp}
Daily Profit: {daily_profit*100:+.2f}%
Total Trades: {total_trades}
Win Rate: {win_rate*100:.1f}%

BEST TRADE:
- Symbol: {best_trade.get('symbol', 'N/A')}
- P&L: {best_trade.get('pnl_percent', 0)*100:+.2f}%
- Duration: {best_trade.get('duration', 0):.1f} min

WORST TRADE:
- Symbol: {worst_trade.get('symbol', 'N/A')}
- P&L: {worst_trade.get('pnl_percent', 0)*100:+.2f}%
- Duration: {worst_trade.get('duration', 0):.1f} min
────────────────────
AI Trading Bot v1.0
"""
        self.send_email(email_subject, email_body)
    
    def send_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Kirim alert umum"""
        severity_emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
            "success": "✅"
        }
        
        emoji = severity_emojis.get(severity, "ℹ️")
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Telegram message
        telegram_msg = f"""
<b>{emoji} ALERT: {alert_type}</b>
──────────────
{message}
──────────────
<b>Time:</b> {timestamp}
<i>AI Trading Bot v1.0</i>
"""
        self.send_telegram(telegram_msg)
        
        # Discord embed
        color_map = {
            "info": 0x3498db,
            "warning": 0xf39c12,
            "error": 0xe74c3c,
            "success": 0x2ecc71
        }
        
        discord_embed = {
            'title': f'{emoji} {alert_type}',
            'description': message,
            'color': color_map.get(severity, 0x3498db),
            'footer': {'text': f'AI Trading Bot v1.0 | {timestamp}'}
        }
        self.send_discord(f"Alert: {alert_type}", discord_embed)
    
    def send_emergency_stop(self, reason: str, details: str = ""):
        """Kirim notifikasi emergency stop"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Telegram message
        telegram_msg = f"""
<b>🚨 EMERGENCY STOP</b>
──────────────
<b>Reason:</b> {reason}
──────────────
{details}
──────────────
<b>Time:</b> {timestamp}
<i>AI Trading Bot has been stopped for safety</i>
"""
        self.send_telegram(telegram_msg)
        
        # Make a phone call if available (Twilio integration bisa ditambahkan)
        self.logger.critical(f"EMERGENCY STOP: {reason}")
    
    def send_health_check(self, status: Dict):
        """Kirim health check report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format status message
        status_msg = ""
        for key, value in status.items():
            status_msg += f"<b>{key}:</b> {value}\n"
        
        telegram_msg = f"""
<b>🩺 HEALTH CHECK</b>
──────────────
{status_msg}
──────────────
<b>Time:</b> {timestamp}
<i>AI Trading Bot v1.0</i>
"""
        self.send_telegram(telegram_msg)