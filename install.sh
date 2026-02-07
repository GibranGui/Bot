#!/bin/bash

echo "🤖 Installing AI Trading Bot..."
echo "================================"

# Create virtual environment
echo "1. Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "2. Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "3. Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "4. Creating directories..."
mkdir -p ml_models logs data backups

# Create config file if not exists
if [ ! -f "config.json" ]; then
    echo "5. Creating config template..."
    cat > config.json << EOF
{
    "api_key": "YOUR_API_KEY_HERE",
    "secret_key": "YOUR_SECRET_KEY_HERE",
    "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID"
}
EOF
    echo "⚠ Please edit config.json with your credentials!"
fi

# Set permissions
echo "6. Setting permissions..."
chmod +x ai_trading_bot.py
chmod +x dashboard.py

echo "✅ Installation complete!"
echo ""
echo "🚀 To start the bot:"
echo "   source venv/bin/activate"
echo "   python ai_trading_bot.py"
echo ""
echo "🌐 To start dashboard:"
echo "   python dashboard.py"
echo ""
echo "📊 Dashboard will be available at: http://localhost:8050"