#!/usr/bin/env python3
"""
TRADING BOT DASHBOARD - Real-time Monitoring
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time

class TradingDashboard:
    """Real-time dashboard untuk monitoring bot"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🤖 AI Trading Bot Dashboard", 
                       style={'textAlign': 'center', 'color': '#2E86AB'}),
                html.Div(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                        id='update-time',
                        style={'textAlign': 'center', 'color': '#666'})
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
            
            # Performance Metrics
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("💰 Daily Profit", style={'color': '#28a745'}),
                        html.H2(id='daily-profit', children='0.00%')
                    ], className='metric-card'),
                    
                    html.Div([
                        html.H3("📊 Win Rate", style={'color': '#007bff'}),
                        html.H2(id='win-rate', children='0.00%')
                    ], className='metric-card'),
                    
                    html.Div([
                        html.H3("🎯 Active Trades", style={'color': '#ffc107'}),
                        html.H2(id='active-trades', children='0')
                    ], className='metric-card'),
                    
                    html.Div([
                        html.H3("📈 Total Trades", style={'color': '#6f42c1'}),
                        html.H2(id='total-trades', children='0')
                    ], className='metric-card')
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(4, 1fr)',
                    'gap': '20px',
                    'marginBottom': '30px'
                }),
                
                # Charts Row 1
                html.Div([
                    dcc.Graph(id='profit-chart', style={'height': '400px'}),
                    dcc.Graph(id='trades-chart', style={'height': '400px'})
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '20px',
                    'marginBottom': '30px'
                }),
                
                # Charts Row 2
                html.Div([
                    dcc.Graph(id='performance-chart', style={'height': '400px'}),
                    dcc.Graph(id='risk-chart', style={'height': '400px'})
                ], style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr',
                    'gap': '20px',
                    'marginBottom': '30px'
                }),
                
                # Data Tables
                html.Div([
                    html.H3("📋 Active Trades"),
                    html.Div(id='active-trades-table'),
                    
                    html.H3("📜 Recent Trade History"),
                    html.Div(id='trade-history-table')
                ]),
                
                # Controls
                html.Div([
                    html.Button('🔄 Refresh Data', id='refresh-btn', n_clicks=0),
                    dcc.Interval(
                        id='interval-component',
                        interval=10*1000,  # 10 detik
                        n_intervals=0
                    )
                ], style={'marginTop': '30px', 'textAlign': 'center'})
            ], style={'padding': '20px'})
        ])
        
        # Custom CSS
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>AI Trading Bot Dashboard</title>
                {%favicon%}
                {%css%}
                <style>
                    .metric-card {
                        background: white;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        text-align: center;
                    }
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                        background-color: #f5f5f5;
                        margin: 0;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('daily-profit', 'children'),
             Output('win-rate', 'children'),
             Output('active-trades', 'children'),
             Output('total-trades', 'children'),
             Output('update-time', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-btn', 'n_clicks')]
        )
        def update_metrics(n_intervals, n_clicks):
            """Update performance metrics"""
            daily_profit = f"{self.bot.daily_profit*100:.2f}%"
            win_rate = f"{self.bot.performance_metrics['win_rate']*100:.2f}%"
            active_trades = len(self.bot.active_trades)
            total_trades = self.bot.performance_metrics['total_trades']
            update_time = f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return daily_profit, win_rate, str(active_trades), str(total_trades), update_time
        
        @self.app.callback(
            Output('profit-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_profit_chart(n_intervals):
            """Update profit chart"""
            # Data contoh (dalam implementasi real, ambil dari database)
            dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
            profits = np.random.randn(len(dates)).cumsum() * 0.01 + 1.0
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=profits * 100,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#2E86AB', width=2)
            ))
            
            fig.update_layout(
                title='Portfolio Growth (%)',
                xaxis_title='Date',
                yaxis_title='Return (%)',
                template='plotly_white',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('active-trades-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_active_trades_table(n_intervals):
            """Update active trades table"""
            if not self.bot.active_trades:
                return html.Div("No active trades", style={'color': '#666'})
            
            table_header = [
                html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Side"),
                    html.Th("Entry Price"),
                    html.Th("Current Price"),
                    html.Th("P&L %"),
                    html.Th("Time")
                ]))
            ]
            
            table_rows = []
            for trade_id, trade in self.bot.active_trades.items():
                # Ini contoh - di implementasi real, ambil current price dari market
                current_price = trade['entry_price'] * (1 + np.random.uniform(-0.02, 0.02))
                pnl = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                pnl_color = 'green' if pnl > 0 else 'red'
                
                row = html.Tr([
                    html.Td(trade['symbol']),
                    html.Td(trade['side'].upper(), 
                           style={'color': 'green' if trade['side'] == 'buy' else 'red'}),
                    html.Td(f"{trade['entry_price']:,.0f}"),
                    html.Td(f"{current_price:,.0f}"),
                    html.Td(f"{pnl:+.2f}%", style={'color': pnl_color}),
                    html.Td(str(trade['entry_time'].strftime('%H:%M:%S')))
                ])
                table_rows.append(row)
            
            table_body = [html.Tbody(table_rows)]
            
            return html.Table(table_header + table_body, 
                            style={'width': '100%', 'borderCollapse': 'collapse'})
    
    def run(self, port=8050):
        """Run dashboard"""
        print(f"🌐 Dashboard running on http://localhost:{port}")
        self.app.run_server(debug=False, port=port)