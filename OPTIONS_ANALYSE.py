# Options-Analyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize_scalar, brentq
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedOptionsStrategyGenerator:
    def __init__(self, symbol: str, risk_free_rate: float = 0.05):
        """
        Advanced Options Strategy Generator with hedge fund level features
        
        Args:
            symbol: Stock ticker symbol
            risk_free_rate: Risk-free interest rate (default 5%)
        """
        self.symbol = symbol.upper()
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.current_price = None
        self.volatility = None
        self.implied_vol_surface = None
        self.earnings_dates = []
        
    def fetch_stock_data(self, period: str = "2y") -> None:
        """Fetch comprehensive stock data and calculate advanced metrics"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.stock_data = ticker.history(period=period)
            info = ticker.info
            
            self.current_price = self.stock_data['Close'].iloc[-1]
            
            # Calculate multiple volatility measures
            returns = np.log(self.stock_data['Close'] / self.stock_data['Close'].shift(1))
            self.volatility = returns.std() * np.sqrt(252)
            
            # Calculate GARCH volatility (simplified)
            self.garch_vol = self._calculate_garch_volatility(returns)
            
            # Calculate implied volatility term structure
            self.vol_term_structure = self._calculate_vol_term_structure(returns)
            
            # Get dividend yield and other fundamentals
            self.dividend_yield = info.get('dividendYield', 0) or 0
            self.beta = info.get('beta', 1.0) or 1.0
            
            print(f"Current price of {self.symbol}: ${self.current_price:.2f}")
            print(f"Historical volatility: {self.volatility:.2%}")
            print(f"GARCH volatility: {self.garch_vol:.2%}")
            print(f"Beta: {self.beta:.2f}")
            print(f"Dividend yield: {self.dividend_yield:.2%}")
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            self.current_price = 100.0
            self.volatility = 0.25
            self.garch_vol = 0.25
            self.vol_term_structure = {30: 0.25, 60: 0.26, 90: 0.27}
            self.dividend_yield = 0.02
            self.beta = 1.0
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """Calculate GARCH(1,1) volatility forecast"""
        # Simplified GARCH calculation
        returns = returns.dropna()
        long_run_var = returns.var()
        
        # GARCH parameters (typical values)
        alpha = 0.1
        beta_garch = 0.85
        omega = long_run_var * (1 - alpha - beta_garch)
        
        # Calculate conditional variance
        conditional_var = long_run_var
        for r in returns.tail(10):
            conditional_var = omega + alpha * r**2 + beta_garch * conditional_var
        
        return np.sqrt(conditional_var * 252)
    
    def _calculate_vol_term_structure(self, returns: pd.Series) -> Dict[int, float]:
        """Calculate volatility term structure"""
        base_vol = returns.std() * np.sqrt(252)
        return {
            7: base_vol * 1.2,    # Weekly options higher vol
            30: base_vol,         # Monthly baseline
            60: base_vol * 1.05,  # Slight increase
            90: base_vol * 1.1,   # Quarterly higher vol
            180: base_vol * 1.15, # Semi-annual
            365: base_vol * 1.2   # Annual
        }
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call', dividend_yield: float = 0) -> float:
        """Enhanced Black-Scholes with dividend yield"""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp(-dividend_yield * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-dividend_yield * T) * norm.cdf(-d1)
        
        return max(price, 0)
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str = 'call', dividend_yield: float = 0) -> Dict[str, float]:
        """Calculate all option Greeks"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0, 'charm': 0, 'vanna': 0}
        
        d1 = (np.log(S / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # First-order Greeks
        if option_type.lower() == 'call':
            delta = np.exp(-dividend_yield * T) * norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = -np.exp(-dividend_yield * T) * norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        # Common Greeks
        gamma = np.exp(-dividend_yield * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta calculation
        if option_type.lower() == 'call':
            theta = (-(S * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)
                    + dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(d1)) / 365
        else:
            theta = (-(S * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)
                    - dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(-d1)) / 365
        
        # Second-order Greeks
        charm = -np.exp(-dividend_yield * T) * norm.pdf(d1) * (2 * (r - dividend_yield) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        vanna = vega * d2 / sigma
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'charm': charm,
            'vanna': vanna
        }
    
    def calculate_implied_volatility(self, option_price: float, S: float, K: float, T: float, 
                                   r: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Brent's method"""
        def objective(sigma):
            return self.black_scholes_price(S, K, T, r, sigma, option_type, self.dividend_yield) - option_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except:
            return self.volatility
    
    def portfolio_greeks(self, strategies: List[Dict]) -> Dict[str, float]:
        """Calculate portfolio-level Greeks"""
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for strategy in strategies:
            if 'greeks' in strategy:
                for greek in portfolio_greeks:
                    if greek in strategy['greeks']:
                        portfolio_greeks[greek] += strategy['greeks'][greek] * strategy.get('quantity', 1)
        
        return portfolio_greeks
    
    # Enhanced Strategy Methods
    def long_call(self, strike: float, expiry_days: int, premium: Optional[float] = None, quantity: int = 1) -> Dict:
        """Enhanced Long Call Strategy"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        if premium is None:
            premium = self.black_scholes_price(self.current_price, strike, T, 
                                             self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        greeks = self.calculate_all_greeks(self.current_price, strike, T, 
                                         self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Risk metrics
        probability_profit = norm.cdf((np.log((strike + premium) / self.current_price)) / (vol * np.sqrt(T)))
        max_risk = premium * quantity
        
        return {
            'strategy': 'Long Call',
            'strike': strike,
            'premium': premium,
            'expiry_days': expiry_days,
            'quantity': quantity,
            'max_profit': 'Unlimited',
            'max_loss': max_risk,
            'breakeven': strike + premium,
            'probability_profit': 1 - probability_profit,
            'risk_reward_ratio': np.inf,
            'greeks': greeks,
            'margin_required': 0,  # No margin for long options
            'volatility_used': vol
        }
    
    def covered_call(self, strike: float, expiry_days: int, shares_owned: int = 100) -> Dict:
        """Covered Call Strategy"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        call_premium = self.black_scholes_price(self.current_price, strike, T, 
                                              self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Calculate combined Greeks (long stock + short call)
        call_greeks = self.calculate_all_greeks(self.current_price, strike, T, 
                                              self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        combined_greeks = {
            'delta': 1 - call_greeks['delta'],  # Long stock delta = 1
            'gamma': -call_greeks['gamma'],
            'theta': -call_greeks['theta'],
            'vega': -call_greeks['vega'],
            'rho': -call_greeks['rho']
        }
        
        max_profit = (strike - self.current_price + call_premium) * shares_owned
        max_loss_per_share = self.current_price - call_premium
        
        return {
            'strategy': 'Covered Call',
            'strike': strike,
            'premium_received': call_premium,
            'expiry_days': expiry_days,
            'shares_owned': shares_owned,
            'max_profit': max_profit,
            'max_loss': max_loss_per_share * shares_owned,
            'breakeven': self.current_price - call_premium,
            'greeks': combined_greeks,
            'annualized_return': (call_premium / self.current_price) * (365 / expiry_days),
            'volatility_used': vol
        }
    
    def iron_butterfly(self, center_strike: float, wing_width: float, expiry_days: int) -> Dict:
        """Iron Butterfly Strategy"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        lower_strike = center_strike - wing_width
        upper_strike = center_strike + wing_width
        
        # Long put at lower strike
        long_put_premium = self.black_scholes_price(self.current_price, lower_strike, T, 
                                                  self.risk_free_rate, vol, 'put', self.dividend_yield)
        # Short put at center
        short_put_premium = self.black_scholes_price(self.current_price, center_strike, T, 
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        # Short call at center
        short_call_premium = self.black_scholes_price(self.current_price, center_strike, T, 
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        # Long call at upper strike
        long_call_premium = self.black_scholes_price(self.current_price, upper_strike, T, 
                                                   self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        net_credit = short_put_premium + short_call_premium - long_put_premium - long_call_premium
        max_loss = wing_width - net_credit
        
        # Calculate combined Greeks
        put_greeks_lower = self.calculate_all_greeks(self.current_price, lower_strike, T, 
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        put_greeks_center = self.calculate_all_greeks(self.current_price, center_strike, T, 
                                                    self.risk_free_rate, vol, 'put', self.dividend_yield)
        call_greeks_center = self.calculate_all_greeks(self.current_price, center_strike, T, 
                                                     self.risk_free_rate, vol, 'call', self.dividend_yield)
        call_greeks_upper = self.calculate_all_greeks(self.current_price, upper_strike, T, 
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        combined_greeks = {
            'delta': (put_greeks_lower['delta'] - put_greeks_center['delta'] - 
                     call_greeks_center['delta'] + call_greeks_upper['delta']),
            'gamma': (put_greeks_lower['gamma'] - put_greeks_center['gamma'] - 
                     call_greeks_center['gamma'] + call_greeks_upper['gamma']),
            'theta': (put_greeks_lower['theta'] - put_greeks_center['theta'] - 
                     call_greeks_center['theta'] + call_greeks_upper['theta']),
            'vega': (put_greeks_lower['vega'] - put_greeks_center['vega'] - 
                    call_greeks_center['vega'] + call_greeks_upper['vega']),
            'rho': (put_greeks_lower['rho'] - put_greeks_center['rho'] - 
                   call_greeks_center['rho'] + call_greeks_upper['rho'])
        }
        
        return {
            'strategy': 'Iron Butterfly',
            'center_strike': center_strike,
            'wing_width': wing_width,
            'strikes': [lower_strike, center_strike, center_strike, upper_strike],
            'net_credit': net_credit,
            'expiry_days': expiry_days,
            'max_profit': net_credit,
            'max_loss': max_loss,
            'breakeven_low': center_strike - net_credit,
            'breakeven_high': center_strike + net_credit,
            'probability_profit': self._calculate_butterfly_prob_profit(center_strike, net_credit, vol, T),
            'volatility_used': vol,
            'greeks': combined_greeks
        }
    
    def _calculate_butterfly_prob_profit(self, center: float, credit: float, vol: float, T: float) -> float:
        """Calculate probability of profit for butterfly"""
        lower_be = center - credit
        upper_be = center + credit
        
        prob_below_lower = norm.cdf(np.log(lower_be / self.current_price) / (vol * np.sqrt(T)))
        prob_above_upper = 1 - norm.cdf(np.log(upper_be / self.current_price) / (vol * np.sqrt(T)))
        
        return prob_below_lower + prob_above_upper
    
    def jade_lizard(self, short_put_strike: float, short_call_strike: float, 
                   long_call_strike: float, expiry_days: int) -> Dict:
        """Jade Lizard Strategy (Advanced)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        short_put_premium = self.black_scholes_price(self.current_price, short_put_strike, T, 
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        short_call_premium = self.black_scholes_price(self.current_price, short_call_strike, T, 
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        long_call_premium = self.black_scholes_price(self.current_price, long_call_strike, T, 
                                                   self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        net_credit = short_put_premium + short_call_premium - long_call_premium
        max_loss_upside = long_call_strike - short_call_strike - net_credit
        max_loss_downside = short_put_strike - net_credit
        
        return {
            'strategy': 'Jade Lizard',
            'short_put_strike': short_put_strike,
            'short_call_strike': short_call_strike,
            'long_call_strike': long_call_strike,
            'net_credit': net_credit,
            'expiry_days': expiry_days,
            'max_profit': net_credit,
            'max_loss_upside': max_loss_upside,
            'max_loss_downside': max_loss_downside,
            'breakeven': short_put_strike - net_credit,
            'volatility_used': vol
        }
    
    def calculate_pnl_advanced(self, strategy_info: Dict, stock_prices: np.ndarray) -> np.ndarray:
        """Advanced P&L calculation with comprehensive strategy support"""
        strategy = strategy_info['strategy']
        
        if strategy == 'Long Call':
            return (np.maximum(stock_prices - strategy_info['strike'], 0) - strategy_info['premium']) * strategy_info.get('quantity', 1)
        
        elif strategy == 'Covered Call':
            stock_pnl = (stock_prices - self.current_price) * strategy_info['shares_owned']
            call_pnl = -np.maximum(stock_prices - strategy_info['strike'], 0) * strategy_info['shares_owned'] / 100
            return stock_pnl + call_pnl + strategy_info['premium_received'] * strategy_info['shares_owned'] / 100
        
        elif strategy == 'Iron Butterfly':
            center = strategy_info['center_strike']
            width = strategy_info['wing_width']
            lower = center - width
            upper = center + width
            
            long_put_pnl = np.maximum(lower - stock_prices, 0)
            short_put_pnl = -np.maximum(center - stock_prices, 0)
            short_call_pnl = -np.maximum(stock_prices - center, 0)
            long_call_pnl = np.maximum(stock_prices - upper, 0)
            
            return long_put_pnl + short_put_pnl + short_call_pnl + long_call_pnl + strategy_info['net_credit']
        
        elif strategy == 'Jade Lizard':
            short_put_pnl = -np.maximum(strategy_info['short_put_strike'] - stock_prices, 0)
            short_call_pnl = -np.maximum(stock_prices - strategy_info['short_call_strike'], 0)
            long_call_pnl = np.maximum(stock_prices - strategy_info['long_call_strike'], 0)
            
            return short_put_pnl + short_call_pnl + long_call_pnl + strategy_info['net_credit']
        
        else:
            return np.zeros_like(stock_prices)
    
    def calculate_volatility_impact(self, strategy_info: Dict, vol_range: np.ndarray) -> np.ndarray:
        """Calculate P&L impact across different volatility levels"""
        strategy = strategy_info['strategy']
        T = strategy_info.get('expiry_days', 30) / 365
        pnl_vol_impact = []
        
        for vol in vol_range:
            if strategy == 'Long Call':
                temp_premium = self.black_scholes_price(
                    self.current_price, strategy_info['strike'], T, 
                    self.risk_free_rate, vol, 'call', self.dividend_yield
                )
                pnl_vol_impact.append(temp_premium - strategy_info['premium'])
                
            elif strategy == 'Iron Butterfly':
                center = strategy_info['center_strike']
                width = strategy_info['wing_width']
                lower = center - width
                upper = center + width
                
                # Recalculate all premiums with new volatility
                long_put_new = self.black_scholes_price(self.current_price, lower, T, 
                                                      self.risk_free_rate, vol, 'put', self.dividend_yield)
                short_put_new = self.black_scholes_price(self.current_price, center, T, 
                                                       self.risk_free_rate, vol, 'put', self.dividend_yield)
                short_call_new = self.black_scholes_price(self.current_price, center, T, 
                                                        self.risk_free_rate, vol, 'call', self.dividend_yield)
                long_call_new = self.black_scholes_price(self.current_price, upper, T, 
                                                       self.risk_free_rate, vol, 'call', self.dividend_yield)
                
                new_net_credit = short_put_new + short_call_new - long_put_new - long_call_new
                pnl_vol_impact.append(new_net_credit - strategy_info['net_credit'])
                
            elif strategy == 'Covered Call':
                temp_premium = self.black_scholes_price(
                    self.current_price, strategy_info['strike'], T, 
                    self.risk_free_rate, vol, 'call', self.dividend_yield
                )
                pnl_vol_impact.append(temp_premium - strategy_info['premium_received'])
                
            else:
                # For other strategies, assume neutral impact
                pnl_vol_impact.append(0)
        
        return np.array(pnl_vol_impact)
    
    def plot_advanced_strategy(self, strategy_info: Dict, price_range: float = 0.4):
        """Advanced strategy plotting with multiple visualizations"""
        current_price = self.current_price
        price_min = current_price * (1 - price_range)
        price_max = current_price * (1 + price_range)
        stock_prices = np.linspace(price_min, price_max, 200)
        
        pnl = self.calculate_pnl_advanced(strategy_info, stock_prices)
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f"Advanced Analysis: {strategy_info['strategy']} for {self.symbol}", fontsize=16, fontweight='bold')
        
        # 1. P&L Diagram
        ax1.plot(stock_prices, pnl, 'b-', linewidth=3, label='P&L at Expiration')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.axvline(x=current_price, color='r', linestyle='--', alpha=0.7, 
                   label=f'Current Price: ${current_price:.2f}')
        
        # Add breakeven lines
        if 'breakeven' in strategy_info:
            ax1.axvline(x=strategy_info['breakeven'], color='g', linestyle=':', 
                       label=f"Breakeven: ${strategy_info['breakeven']:.2f}")
        elif 'breakeven_low' in strategy_info and 'breakeven_high' in strategy_info:
            ax1.axvline(x=strategy_info['breakeven_low'], color='g', linestyle=':', 
                       label=f"BE Low: ${strategy_info['breakeven_low']:.2f}")
            ax1.axvline(x=strategy_info['breakeven_high'], color='g', linestyle=':', 
                       label=f"BE High: ${strategy_info['breakeven_high']:.2f}")
        
        # Profit/Loss zones
        profit_mask = pnl > 0
        loss_mask = pnl < 0
        ax1.fill_between(stock_prices, 0, pnl, where=profit_mask, alpha=0.3, color='green', label='Profit Zone')
        ax1.fill_between(stock_prices, 0, pnl, where=loss_mask, alpha=0.3, color='red', label='Loss Zone')
        
        ax1.set_xlabel('Stock Price at Expiration ($)')
        ax1.set_ylabel('Profit/Loss ($)')
        ax1.set_title('P&L Diagram')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Greeks Evolution
        if 'greeks' in strategy_info:
            days_to_expiry = np.linspace(strategy_info.get('expiry_days', 30), 1, 50)
            greeks_evolution = {'delta': [], 'gamma': [], 'theta': [], 'vega': []}
            
            for days in days_to_expiry:
                T = days / 365
                if strategy_info['strategy'] == 'Long Call':
                    temp_greeks = self.calculate_all_greeks(
                        current_price, strategy_info.get('strike', current_price), T,
                        self.risk_free_rate, strategy_info.get('volatility_used', self.volatility),
                        'call', self.dividend_yield
                    )
                else:
                    # For complex strategies, use current Greeks values
                    temp_greeks = strategy_info['greeks']
                
                for greek in greeks_evolution:
                    greeks_evolution[greek].append(temp_greeks.get(greek, 0))
            
            for greek, values in greeks_evolution.items():
                ax2.plot(days_to_expiry, values, label=f'{greek.capitalize()}', linewidth=2)
            
            ax2.set_xlabel('Days to Expiration')
            ax2.set_ylabel('Greek Values')
            ax2.set_title('Greeks Evolution Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Volatility Surface Impact (Fixed)
        vol_range = np.linspace(0.1, 0.6, 50)
        pnl_vol_impact = self.calculate_volatility_impact(strategy_info, vol_range)
        
        ax3.plot(vol_range * 100, pnl_vol_impact, 'purple', linewidth=3)
        ax3.axvline(x=strategy_info.get('volatility_used', self.volatility) * 100, 
                   color='r', linestyle='--', label='Current Vol')
        ax3.set_xlabel('Implied Volatility (%)')
        ax3.set_ylabel('P&L Impact ($)')
        ax3.set_title('Volatility Sensitivity Analysis')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Risk Metrics Dashboard
        ax4.axis('off')
        
        # Create risk metrics text
        risk_text = f"""
        RISK METRICS DASHBOARD
        {'='*30}
        
        Strategy: {strategy_info['strategy']}
        Current Stock Price: ${current_price:.2f}
        
        PROFIT/LOSS ANALYSIS:
        Max Profit: {strategy_info.get('max_profit', 'N/A')}
        Max Loss: ${strategy_info.get('max_loss', 'N/A'):.2f}
        
        PROBABILITY ANALYSIS:
        Prob. of Profit: {strategy_info.get('probability_profit', 0)*100:.1f}%
        
        GREEKS (Current):
        Delta: {strategy_info.get('greeks', {}).get('delta', 0):.4f}
        Gamma: {strategy_info.get('greeks', {}).get('gamma', 0):.4f}
        Theta: {strategy_info.get('greeks', {}).get('theta', 0):.4f}
        Vega: {strategy_info.get('greeks', {}).get('vega', 0):.4f}
        
        VOLATILITY METRICS:
        Implied Vol: {strategy_info.get('volatility_used', self.volatility)*100:.1f}%
        Historical Vol: {self.volatility*100:.1f}%
        
        TIME METRICS:
        Days to Expiry: {strategy_info.get('expiry_days', 0)}
        """
        
        ax4.text(0.05, 0.95, risk_text, fontsize=12, va='top', family='monospace')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
    def diagonal_spread(self, short_strike: float, long_strike: float, 
                        short_expiry_days: int, long_expiry_days: int, 
                        option_type: str = 'call') -> Dict:
        """Diagonal Spread Strategy (Time and Strike Spread)"""
        short_T = short_expiry_days / 365
        long_T = long_expiry_days / 365
        
        short_vol = self.vol_term_structure.get(short_expiry_days, self.volatility)
        long_vol = self.vol_term_structure.get(long_expiry_days, self.volatility)
        
        short_premium = self.black_scholes_price(self.current_price, short_strike, short_T,
                                               self.risk_free_rate, short_vol, option_type, self.dividend_yield)
        
        long_premium = self.black_scholes_price(self.current_price, long_strike, long_T,
                                              self.risk_free_rate, long_vol, option_type, self.dividend_yield)
        
        net_debit = long_premium - short_premium
        
        # Calculate Greeks
        short_greeks = self.calculate_all_greeks(self.current_price, short_strike, short_T,
                                               self.risk_free_rate, short_vol, option_type, self.dividend_yield)
        
        long_greeks = self.calculate_all_greeks(self.current_price, long_strike, long_T,
                                              self.risk_free_rate, long_vol, option_type, self.dividend_yield)
        
        combined_greeks = {
            'delta': long_greeks['delta'] - short_greeks['delta'],
            'gamma': long_greeks['gamma'] - short_greeks['gamma'],
            'theta': long_greeks['theta'] - short_greeks['theta'],
            'vega': long_greeks['vega'] - short_greeks['vega'],
            'rho': long_greeks['rho'] - short_greeks['rho']
        }
        
        # Estimate max profit/loss (simplified)
        if option_type.lower() == 'call':
            max_profit_estimate = (short_strike - long_strike) + net_debit if short_strike > long_strike else 'Variable'
            max_loss_estimate = net_debit
        else:  # put
            max_profit_estimate = (long_strike - short_strike) + net_debit if long_strike > short_strike else 'Variable'
            max_loss_estimate = net_debit
            
        return {
            'strategy': f'Diagonal {option_type.capitalize()} Spread',
            'short_strike': short_strike,
            'long_strike': long_strike,
            'short_expiry_days': short_expiry_days,
            'long_expiry_days': long_expiry_days,
            'short_premium': short_premium,
            'long_premium': long_premium,
            'net_debit': net_debit,
            'max_profit': max_profit_estimate,
            'max_loss': max_loss_estimate,
            'greeks': combined_greeks,
            'volatility_used_short': short_vol,
            'volatility_used_long': long_vol
        }
    
    def double_calendar_spread(self, center_strike: float, expiry_days_near: int, 
                              expiry_days_far: int) -> Dict:
        """Double Calendar Spread Strategy (Both Calls and Puts)"""
        near_T = expiry_days_near / 365
        far_T = expiry_days_far / 365
        
        near_vol = self.vol_term_structure.get(expiry_days_near, self.volatility)
        far_vol = self.vol_term_structure.get(expiry_days_far, self.volatility)
        
        # Near-term options (short)
        near_call_premium = self.black_scholes_price(self.current_price, center_strike, near_T,
                                                  self.risk_free_rate, near_vol, 'call', self.dividend_yield)
        
        near_put_premium = self.black_scholes_price(self.current_price, center_strike, near_T,
                                                 self.risk_free_rate, near_vol, 'put', self.dividend_yield)
        
        # Far-term options (long)
        far_call_premium = self.black_scholes_price(self.current_price, center_strike, far_T,
                                                 self.risk_free_rate, far_vol, 'call', self.dividend_yield)
        
        far_put_premium = self.black_scholes_price(self.current_price, center_strike, far_T,
                                                self.risk_free_rate, far_vol, 'put', self.dividend_yield)
        
        net_debit = (far_call_premium + far_put_premium) - (near_call_premium + near_put_premium)
        
        # Calculate Greeks
        near_call_greeks = self.calculate_all_greeks(self.current_price, center_strike, near_T,
                                                  self.risk_free_rate, near_vol, 'call', self.dividend_yield)
        
        near_put_greeks = self.calculate_all_greeks(self.current_price, center_strike, near_T,
                                                 self.risk_free_rate, near_vol, 'put', self.dividend_yield)
        
        far_call_greeks = self.calculate_all_greeks(self.current_price, center_strike, far_T,
                                                 self.risk_free_rate, far_vol, 'call', self.dividend_yield)
        
        far_put_greeks = self.calculate_all_greeks(self.current_price, center_strike, far_T,
                                                self.risk_free_rate, far_vol, 'put', self.dividend_yield)
        
        combined_greeks = {
            'delta': (far_call_greeks['delta'] + far_put_greeks['delta']) - 
                    (near_call_greeks['delta'] + near_put_greeks['delta']),
            'gamma': (far_call_greeks['gamma'] + far_put_greeks['gamma']) - 
                    (near_call_greeks['gamma'] + near_put_greeks['gamma']),
            'theta': (far_call_greeks['theta'] + far_put_greeks['theta']) - 
                    (near_call_greeks['theta'] + near_put_greeks['theta']),
            'vega': (far_call_greeks['vega'] + far_put_greeks['vega']) - 
                   (near_call_greeks['vega'] + near_put_greeks['vega']),
            'rho': (far_call_greeks['rho'] + far_put_greeks['rho']) - 
                  (near_call_greeks['rho'] + near_put_greeks['rho'])
        }
        
        return {
            'strategy': 'Double Calendar Spread',
            'center_strike': center_strike,
            'near_expiry_days': expiry_days_near,
            'far_expiry_days': expiry_days_far,
            'near_call_premium': near_call_premium,
            'near_put_premium': near_put_premium,
            'far_call_premium': far_call_premium,
            'far_put_premium': far_put_premium,
            'net_debit': net_debit,
            'max_profit': 'Variable (Depends on volatility expansion)',
            'max_loss': net_debit,
            'greeks': combined_greeks,
            'volatility_used_near': near_vol,
            'volatility_used_far': far_vol
        }
    
    def risk_reversal(self, put_strike: float, call_strike: float, expiry_days: int) -> Dict:
        """Risk Reversal Strategy (Sell OTM Put, Buy OTM Call)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        put_premium = self.black_scholes_price(self.current_price, put_strike, T,
                                             self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        call_premium = self.black_scholes_price(self.current_price, call_strike, T,
                                              self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        net_cost = call_premium - put_premium
        
        # Calculate Greeks
        put_greeks = self.calculate_all_greeks(self.current_price, put_strike, T,
                                             self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        call_greeks = self.calculate_all_greeks(self.current_price, call_strike, T,
                                              self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        combined_greeks = {
            'delta': call_greeks['delta'] + put_greeks['delta'],  # Short put has positive delta
            'gamma': call_greeks['gamma'] - put_greeks['gamma'],
            'theta': call_greeks['theta'] - put_greeks['theta'],
            'vega': call_greeks['vega'] - put_greeks['vega'],
            'rho': call_greeks['rho'] - put_greeks['rho']
        }
        
        return {
            'strategy': 'Risk Reversal',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expiry_days': expiry_days,
            'put_premium': put_premium,
            'call_premium': call_premium,
            'net_cost': net_cost,
            'max_profit': 'Unlimited',
            'max_loss': put_strike - net_cost,
            'breakeven_upside': call_strike + net_cost,
            'breakeven_downside': put_strike - net_cost,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def ratio_spread(self, long_strike: float, short_strike: float, ratio: float, 
                    expiry_days: int, option_type: str = 'call') -> Dict:
        """Ratio Spread Strategy (Buy 1, Sell Multiple)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        long_premium = self.black_scholes_price(self.current_price, long_strike, T,
                                              self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        short_premium = self.black_scholes_price(self.current_price, short_strike, T,
                                               self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        net_cost = long_premium - (short_premium * ratio)
        
        # Calculate Greeks
        long_greeks = self.calculate_all_greeks(self.current_price, long_strike, T,
                                              self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        short_greeks = self.calculate_all_greeks(self.current_price, short_strike, T,
                                               self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        combined_greeks = {
            'delta': long_greeks['delta'] - (short_greeks['delta'] * ratio),
            'gamma': long_greeks['gamma'] - (short_greeks['gamma'] * ratio),
            'theta': long_greeks['theta'] - (short_greeks['theta'] * ratio),
            'vega': long_greeks['vega'] - (short_greeks['vega'] * ratio),
            'rho': long_greeks['rho'] - (short_greeks['rho'] * ratio)
        }
        
        # Calculate max profit/loss based on option type
        if option_type.lower() == 'call':
            max_profit = (short_strike - long_strike) - net_cost if net_cost > 0 else (short_strike - long_strike) + abs(net_cost)
            max_loss = net_cost if net_cost > 0 else 'Unlimited'
        else:  # put
            max_profit = (long_strike - short_strike) - net_cost if net_cost > 0 else (long_strike - short_strike) + abs(net_cost)
            max_loss = net_cost if net_cost > 0 else 'Unlimited'
        
        return {
            'strategy': f'Ratio {option_type.capitalize()} Spread ({ratio}:1)',
            'long_strike': long_strike,
            'short_strike': short_strike,
            'ratio': ratio,
            'expiry_days': expiry_days,
            'long_premium': long_premium,
            'short_premium': short_premium,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def condor(self, lowest_strike: float, lower_middle_strike: float, 
              upper_middle_strike: float, highest_strike: float, 
              expiry_days: int) -> Dict:
        """Condor Strategy (Iron Condor with equal wing widths)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        # Long put at lowest strike
        long_put_premium = self.black_scholes_price(self.current_price, lowest_strike, T,
                                                  self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        # Short put at lower middle strike
        short_put_premium = self.black_scholes_price(self.current_price, lower_middle_strike, T,
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        # Short call at upper middle strike
        short_call_premium = self.black_scholes_price(self.current_price, upper_middle_strike, T,
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Long call at highest strike
        long_call_premium = self.black_scholes_price(self.current_price, highest_strike, T,
                                                   self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        net_credit = (short_put_premium + short_call_premium) - (long_put_premium + long_call_premium)
        
        # Calculate max profit/loss
        max_profit = net_credit
        max_loss = min(highest_strike - upper_middle_strike, lower_middle_strike - lowest_strike) - net_credit
        
        # Calculate Greeks
        long_put_greeks = self.calculate_all_greeks(self.current_price, lowest_strike, T,
                                                  self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        short_put_greeks = self.calculate_all_greeks(self.current_price, lower_middle_strike, T,
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        short_call_greeks = self.calculate_all_greeks(self.current_price, upper_middle_strike, T,
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        long_call_greeks = self.calculate_all_greeks(self.current_price, highest_strike, T,
                                                   self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        combined_greeks = {
            'delta': long_put_greeks['delta'] - short_put_greeks['delta'] - 
                    short_call_greeks['delta'] + long_call_greeks['delta'],
            'gamma': long_put_greeks['gamma'] - short_put_greeks['gamma'] - 
                    short_call_greeks['gamma'] + long_call_greeks['gamma'],
            'theta': long_put_greeks['theta'] - short_put_greeks['theta'] - 
                    short_call_greeks['theta'] + long_call_greeks['theta'],
            'vega': long_put_greeks['vega'] - short_put_greeks['vega'] - 
                   short_call_greeks['vega'] + long_call_greeks['vega'],
            'rho': long_put_greeks['rho'] - short_put_greeks['rho'] - 
                  short_call_greeks['rho'] + long_call_greeks['rho']
        }
        
        return {
            'strategy': 'Iron Condor',
            'lowest_strike': lowest_strike,
            'lower_middle_strike': lower_middle_strike,
            'upper_middle_strike': upper_middle_strike,
            'highest_strike': highest_strike,
            'expiry_days': expiry_days,
            'net_credit': net_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_low': lower_middle_strike - net_credit,
            'breakeven_high': upper_middle_strike + net_credit,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def backratio_spread(self, front_strike: float, back_strike: float, ratio: float, 
                        expiry_days: int, option_type: str = 'call') -> Dict:
        """Back-Ratio Spread Strategy (Sell 1, Buy Multiple)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        front_premium = self.black_scholes_price(self.current_price, front_strike, T,
                                               self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        back_premium = self.black_scholes_price(self.current_price, back_strike, T,
                                              self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        net_cost = (back_premium * ratio) - front_premium
        
        # Calculate Greeks
        front_greeks = self.calculate_all_greeks(self.current_price, front_strike, T,
                                               self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        back_greeks = self.calculate_all_greeks(self.current_price, back_strike, T,
                                              self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        combined_greeks = {
            'delta': (back_greeks['delta'] * ratio) - front_greeks['delta'],
            'gamma': (back_greeks['gamma'] * ratio) - front_greeks['gamma'],
            'theta': (back_greeks['theta'] * ratio) - front_greeks['theta'],
            'vega': (back_greeks['vega'] * ratio) - front_greeks['vega'],
            'rho': (back_greeks['rho'] * ratio) - front_greeks['rho']
        }
        
        # Calculate max profit/loss based on option type
        if option_type.lower() == 'call':
            if back_strike > front_strike:
                max_profit = 'Unlimited'
                max_loss = (back_strike - front_strike) + net_cost
            else:
                max_profit = (front_strike - back_strike) - net_cost
                max_loss = net_cost
        else:  # put
            if back_strike < front_strike:
                max_profit = 'Unlimited'
                max_loss = (front_strike - back_strike) + net_cost
            else:
                max_profit = (back_strike - front_strike) - net_cost
                max_loss = net_cost
        
        return {
            'strategy': f'Back-Ratio {option_type.capitalize()} Spread (1:{ratio})',
            'front_strike': front_strike,
            'back_strike': back_strike,
            'ratio': ratio,
            'expiry_days': expiry_days,
            'front_premium': front_premium,
            'back_premium': back_premium,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def optimize_strategy(self, strategy_type: str, target_delta: float = 0, 
                         max_loss_limit: float = None, expiry_days: int = 30) -> Dict:
        """Optimize strategy parameters based on constraints"""
        current_price = self.current_price
        
        if strategy_type == 'iron_butterfly':
            # Optimize center strike and wing width
            def objective(params):
                center, width = params
                strategy = self.iron_butterfly(center, width, expiry_days)
                
                # Penalize strategies that exceed max loss limit
                if max_loss_limit and strategy['max_loss'] > max_loss_limit:
                    return 1000  # Large penalty
                
                # Optimize for delta-neutral and maximum theta/vega ratio
                delta_penalty = abs(strategy['greeks']['delta'] - target_delta) * 100
                theta_vega_ratio = -strategy['greeks']['theta'] / (strategy['greeks']['vega'] + 0.0001)
                
                return delta_penalty - theta_vega_ratio
            
            # Initial guess: ATM strike and 5% wing width
            initial_guess = [current_price, current_price * 0.05]
            
            # Bounds for optimization
            bounds = [(current_price * 0.8, current_price * 1.2),  # Center strike bounds
                     (current_price * 0.02, current_price * 0.2)]  # Wing width bounds
            
            from scipy.optimize import minimize
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimized_center, optimized_width = result.x
                return self.iron_butterfly(optimized_center, optimized_width, expiry_days)
            else:
                # Fallback to default parameters
                return self.iron_butterfly(current_price, current_price * 0.05, expiry_days)
        
        elif strategy_type == 'covered_call':
            # Optimize strike price for maximum yield
            def objective(strike):
                strategy = self.covered_call(strike, expiry_days)
                
                # Maximize annualized return while respecting delta target
                delta_penalty = abs(strategy['greeks']['delta'] - target_delta) * 100
                return -strategy['annualized_return'] + delta_penalty
            
            # Search range: 100% to 120% of current price
            result = minimize_scalar(objective, bounds=(current_price, current_price * 1.2), method='bounded')
            
            if result.success:
                return self.covered_call(result.x, expiry_days)
            else:
                # Fallback to 5% OTM
                return self.covered_call(current_price * 1.05, expiry_days)
        
        elif strategy_type == 'risk_reversal':
            # Optimize put and call strikes for delta-neutral position
            def objective(params):
                put_strike, call_strike = params
                strategy = self.risk_reversal(put_strike, call_strike, expiry_days)
                
                # Penalize strategies that exceed max loss limit
                if max_loss_limit and strategy['max_loss'] > max_loss_limit:
                    return 1000  # Large penalty
                
                # Optimize for target delta and maximum premium received
                delta_penalty = abs(strategy['greeks']['delta'] - target_delta) * 100
                premium_factor = -strategy['net_cost']  # Negative because we want to maximize premium
                
                return delta_penalty + premium_factor
            
            # Initial guess: 5% OTM put and call
            initial_guess = [current_price * 0.95, current_price * 1.05]
            
            # Bounds for optimization
            bounds = [(current_price * 0.8, current_price * 0.99),  # Put strike bounds
                     (current_price * 1.01, current_price * 1.2)]  # Call strike bounds
            
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                optimized_put, optimized_call = result.x
                return self.risk_reversal(optimized_put, optimized_call, expiry_days)
            else:
                # Fallback to default parameters
                return self.risk_reversal(current_price * 0.95, current_price * 1.05, expiry_days)
        
        else:
            raise ValueError(f"Optimization not implemented for strategy type: {strategy_type}")
    
    def generate_earnings_strategies(self, days_to_earnings: int = 30) -> List[Dict]:
        """Generate strategies optimized for earnings announcements"""
        strategies = []
        
        # Volatility typically increases before earnings
        earnings_vol = self.volatility * 1.3
        
        # 1. Long Straddle (volatility expansion play)
        straddle_expiry = days_to_earnings + 5  # Capture post-earnings move
        
        # Calculate premiums with elevated IV
        call_premium = self.black_scholes_price(self.current_price, self.current_price, straddle_expiry/365,
                                              self.risk_free_rate, earnings_vol, 'call', self.dividend_yield)
        
        put_premium = self.black_scholes_price(self.current_price, self.current_price, straddle_expiry/365,
                                             self.risk_free_rate, earnings_vol, 'put', self.dividend_yield)
        
        total_premium = call_premium + put_premium
        
        # Calculate Greeks
        call_greeks = self.calculate_all_greeks(self.current_price, self.current_price, straddle_expiry/365,
                                              self.risk_free_rate, earnings_vol, 'call', self.dividend_yield)
        
        put_greeks = self.calculate_all_greeks(self.current_price, self.current_price, straddle_expiry/365,
                                             self.risk_free_rate, earnings_vol, 'put', self.dividend_yield)
        
        combined_greeks = {
            'delta': call_greeks['delta'] + put_greeks['delta'],
            'gamma': call_greeks['gamma'] + put_greeks['gamma'],
            'theta': call_greeks['theta'] + put_greeks['theta'],
            'vega': call_greeks['vega'] + put_greeks['vega'],
            'rho': call_greeks['rho'] + put_greeks['rho']
        }
        
        # Expected move based on options pricing
        expected_move = total_premium / self.current_price
        
        straddle = {
            'strategy': 'Earnings Straddle',
            'strike': self.current_price,
            'expiry_days': straddle_expiry,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'total_premium': total_premium,
            'breakeven_up': self.current_price + total_premium,
            'breakeven_down': self.current_price - total_premium,
            'expected_move': expected_move * 100,  # As percentage
            'greeks': combined_greeks,
            'max_loss': total_premium,
            'max_profit': 'Unlimited',
            'volatility_used': earnings_vol
        }
        
        strategies.append(straddle)
        
        # 2. Iron Condor (IV crush play)
        # Use wider wings to account for potential large move
        expected_move_dollars = self.current_price * expected_move
        
        lower_middle = self.current_price - (expected_move_dollars * 0.8)
        upper_middle = self.current_price + (expected_move_dollars * 0.8)
        lowest = lower_middle - (expected_move_dollars * 0.5)
        highest = upper_middle + (expected_move_dollars * 0.5)
        
        condor = self.condor(lowest, lower_middle, upper_middle, highest, straddle_expiry)
        condor['strategy'] = 'Earnings Iron Condor'
        condor['expected_move'] = expected_move * 100  # As percentage
        
        strategies.append(condor)
        
        # 3. Calendar Spread (IV crush play with directionality)
        # Short near-term, long further-term
        near_term = days_to_earnings + 1
        far_term = days_to_earnings + 30
        
        calendar = self.double_calendar_spread(self.current_price, near_term, far_term)
        calendar['strategy'] = 'Earnings Calendar Spread'
        calendar['expected_move'] = expected_move * 100  # As percentage
        
        strategies.append(calendar)
        
        return strategies
    
    def generate_portfolio_hedge(self, portfolio_delta: float, portfolio_value: float) -> Dict:
        """Generate optimal hedge for an existing portfolio"""
        # Calculate number of contracts needed to neutralize delta
        hedge_delta_needed = -portfolio_delta
        
        # Determine if we need positive or negative delta
        if hedge_delta_needed > 0:
            # Need positive delta - use calls or put spreads
            expiry_days = 45  # Standard 45 DTE
            
            # Try a long call
            atm_call = self.long_call(self.current_price, expiry_days)
            contracts_needed = hedge_delta_needed / atm_call['greeks']['delta']
            
            hedge_cost = atm_call['premium'] * contracts_needed
            hedge_cost_pct = hedge_cost / portfolio_value * 100
            
            call_hedge = {
                'strategy': 'Portfolio Hedge - Long Calls',
                'contracts': contracts_needed,
                'strike': self.current_price,
                'expiry_days': expiry_days,
                'premium': atm_call['premium'],
                'total_cost': hedge_cost,
                'cost_percentage': hedge_cost_pct,
                'delta_hedge': hedge_delta_needed,
                'greeks': {k: v * contracts_needed for k, v in atm_call['greeks'].items()}
            }
            
            return call_hedge
        else:
            # Need negative delta - use puts or call spreads
            expiry_days = 45  # Standard 45 DTE
            
            # Try a long put
            atm_put_premium = self.black_scholes_price(self.current_price, self.current_price, expiry_days/365,
                                                     self.risk_free_rate, self.volatility, 'put', self.dividend_yield)
            
            put_greeks = self.calculate_all_greeks(self.current_price, self.current_price, expiry_days/365,
                                                 self.risk_free_rate, self.volatility, 'put', self.dividend_yield)
            
            contracts_needed = abs(hedge_delta_needed / put_greeks['delta'])
            
            hedge_cost = atm_put_premium * contracts_needed
            hedge_cost_pct = hedge_cost / portfolio_value * 100
            
            put_hedge = {
                'strategy': 'Portfolio Hedge - Long Puts',
                'contracts': contracts_needed,
                'strike': self.current_price,
                'expiry_days': expiry_days,
                'premium': atm_put_premium,
                'total_cost': hedge_cost,
                'cost_percentage': hedge_cost_pct,
                'delta_hedge': hedge_delta_needed,
                'greeks': {k: v * contracts_needed for k, v in put_greeks.items()}
            }
            
            return put_hedge
    
    def generate_volatility_strategies(self, vol_view: str = 'neutral') -> List[Dict]:
        """Generate strategies based on volatility outlook"""
        strategies = []
        
        if vol_view == 'increase':
            # Strategies that benefit from volatility increase
            
            # 1. Long Straddle
            expiry_days = 45
            straddle_call_premium = self.black_scholes_price(self.current_price, self.current_price, expiry_days/365,
                                                          self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            straddle_put_premium = self.black_scholes_price(self.current_price, self.current_price, expiry_days/365,
                                                         self.risk_free_rate, self.volatility, 'put', self.dividend_yield)
            
            call_greeks = self.calculate_all_greeks(self.current_price, self.current_price, expiry_days/365,
                                                  self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            put_greeks = self.calculate_all_greeks(self.current_price, self.current_price, expiry_days/365,
                                                 self.risk_free_rate, self.volatility, 'put', self.dividend_yield)
            
            combined_greeks = {
                'delta': call_greeks['delta'] + put_greeks['delta'],
                'gamma': call_greeks['gamma'] + put_greeks['gamma'],
                'theta': call_greeks['theta'] + put_greeks['theta'],
                'vega': call_greeks['vega'] + put_greeks['vega'],
                'rho': call_greeks['rho'] + put_greeks['rho']
            }
            
            total_premium = straddle_call_premium + straddle_put_premium
            
            straddle = {
                'strategy': 'Long Straddle (Vol Increase)',
                'strike': self.current_price,
                'expiry_days': expiry_days,
                'call_premium': straddle_call_premium,
                'put_premium': straddle_put_premium,
                'total_premium': total_premium,
                'breakeven_up': self.current_price + total_premium,
                'breakeven_down': self.current_price - total_premium,
                'max_loss': total_premium,
                'max_profit': 'Unlimited',
                'greeks': combined_greeks,
                'volatility_used': self.volatility
            }
            
            strategies.append(straddle)
            
            # 2. Call Backspread
            ratio = 2.0
            front_strike = self.current_price
            back_strike = self.current_price * 1.05
            
            backspread = self.backratio_spread(front_strike, back_strike, ratio, expiry_days, 'call')
            backspread['strategy'] = 'Call Backspread (Vol Increase)'
            
            strategies.append(backspread)
            
        elif vol_view == 'decrease':
            # Strategies that benefit from volatility decrease
            
            # 1. Iron Condor
            expiry_days = 45
            width = self.current_price * 0.05
            
            lower_middle = self.current_price - width
            upper_middle = self.current_price + width
            lowest = lower_middle - width
            highest = upper_middle + width
            
            condor = self.condor(lowest, lower_middle, upper_middle, highest, expiry_days)
            condor['strategy'] = 'Iron Condor (Vol Decrease)'
            
            strategies.append(condor)
            
            # 2. Short Straddle(with defined risk)
            expiry_days = 30  # Shorter timeframe for short vol strategies
            
            short_call_premium = self.black_scholes_price(self.current_price, self.current_price, expiry_days/365,
                                                       self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            short_put_premium = self.black_scholes_price(self.current_price, self.current_price, expiry_days/365,
                                                      self.risk_free_rate, self.volatility, 'put', self.dividend_yield)
            
            # Add long wings for protection
            wing_width = self.current_price * 0.1
            
            long_put_premium = self.black_scholes_price(self.current_price, self.current_price - wing_width, expiry_days/365,
                                                     self.risk_free_rate, self.volatility, 'put', self.dividend_yield)
            
            long_call_premium = self.black_scholes_price(self.current_price, self.current_price + wing_width, expiry_days/365,
                                                      self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            net_credit = (short_call_premium + short_put_premium) - (long_call_premium + long_put_premium)
            
            iron_butterfly = {
                'strategy': 'Iron Butterfly (Vol Decrease)',
                'center_strike': self.current_price,
                'wing_width': wing_width,
                'expiry_days': expiry_days,
                'net_credit': net_credit,
                'max_profit': net_credit,
                'max_loss': wing_width - net_credit,
                'breakeven_low': self.current_price - net_credit,
                'breakeven_high': self.current_price + net_credit,
                'volatility_used': self.volatility
            }
            
            strategies.append(iron_butterfly)
            
        else:  # neutral
            # Strategies for stable volatility
            
            # 1. Calendar Spread
            near_term = 30
            far_term = 60
            
            calendar = self.double_calendar_spread(self.current_price, near_term, far_term)
            calendar['strategy'] = 'Double Calendar (Vol Neutral)'
            
            strategies.append(calendar)
            
            # 2. Butterfly
            expiry_days = 45
            width = self.current_price * 0.03
            
            butterfly = {
                'strategy': 'Butterfly (Vol Neutral)',
                'center_strike': self.current_price,
                'wing_width': width,
                'expiry_days': expiry_days,
                'strikes': [self.current_price - width, self.current_price, self.current_price + width],
                'volatility_used': self.volatility
            }
            
            # Calculate premiums
            lower_call_premium = self.black_scholes_price(self.current_price, self.current_price - width, expiry_days/365,
                                                       self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            center_call_premium = self.black_scholes_price(self.current_price, self.current_price, expiry_days/365,
                                                        self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            upper_call_premium = self.black_scholes_price(self.current_price, self.current_price + width, expiry_days/365,
                                                       self.risk_free_rate, self.volatility, 'call', self.dividend_yield)
            
            net_debit = lower_call_premium - (2 * center_call_premium) + upper_call_premium
            
            butterfly['net_debit'] = net_debit
            butterfly['max_profit'] = width - net_debit
            butterfly['max_loss'] = net_debit
            butterfly['breakeven_low'] = self.current_price - width + net_debit
            butterfly['breakeven_high'] = self.current_price + width - net_debit
            
            strategies.append(butterfly)
        
        return strategies
    
    def double_calendar_spread(self, center_strike: float, near_term_days: int, far_term_days: int) -> Dict:
        """Double Calendar Spread (Put and Call calendars at same strike)"""
        near_T = near_term_days / 365
        far_T = far_term_days / 365
        
        near_vol = self.vol_term_structure.get(near_term_days, self.volatility)
        far_vol = self.vol_term_structure.get(far_term_days, self.volatility)
        
        # Call calendar
        near_call_premium = self.black_scholes_price(self.current_price, center_strike, near_T,
                                                  self.risk_free_rate, near_vol, 'call', self.dividend_yield)
        
        far_call_premium = self.black_scholes_price(self.current_price, center_strike, far_T,
                                                 self.risk_free_rate, far_vol, 'call', self.dividend_yield)
        
        # Put calendar
        near_put_premium = self.black_scholes_price(self.current_price, center_strike, near_T,
                                                 self.risk_free_rate, near_vol, 'put', self.dividend_yield)
        
        far_put_premium = self.black_scholes_price(self.current_price, center_strike, far_T,
                                                self.risk_free_rate, far_vol, 'put', self.dividend_yield)
        
        # Net debit
        net_debit = (far_call_premium - near_call_premium) + (far_put_premium - near_put_premium)
        
        # Calculate Greeks
        near_call_greeks = self.calculate_all_greeks(self.current_price, center_strike, near_T,
                                                  self.risk_free_rate, near_vol, 'call', self.dividend_yield)
        
        far_call_greeks = self.calculate_all_greeks(self.current_price, center_strike, far_T,
                                                 self.risk_free_rate, far_vol, 'call', self.dividend_yield)
        
        near_put_greeks = self.calculate_all_greeks(self.current_price, center_strike, near_T,
                                                 self.risk_free_rate, near_vol, 'put', self.dividend_yield)
        
        far_put_greeks = self.calculate_all_greeks(self.current_price, center_strike, far_T,
                                                self.risk_free_rate, far_vol, 'put', self.dividend_yield)
        
        # Combined Greeks
        combined_greeks = {
            'delta': (-near_call_greeks['delta'] + far_call_greeks['delta'] - 
                     near_put_greeks['delta'] + far_put_greeks['delta']),
            'gamma': (-near_call_greeks['gamma'] + far_call_greeks['gamma'] - 
                     near_put_greeks['gamma'] + far_put_greeks['gamma']),
            'theta': (-near_call_greeks['theta'] + far_call_greeks['theta'] - 
                     near_put_greeks['theta'] + far_put_greeks['theta']),
            'vega': (-near_call_greeks['vega'] + far_call_greeks['vega'] - 
                    near_put_greeks['vega'] + far_put_greeks['vega']),
            'rho': (-near_call_greeks['rho'] + far_call_greeks['rho'] - 
                   near_put_greeks['rho'] + far_put_greeks['rho'])
        }
        
        return {
            'strategy': 'Double Calendar Spread',
            'center_strike': center_strike,
            'near_term_days': near_term_days,
            'far_term_days': far_term_days,
            'near_call_premium': near_call_premium,
            'far_call_premium': far_call_premium,
            'near_put_premium': near_put_premium,
            'far_put_premium': far_put_premium,
            'net_debit': net_debit,
            'max_loss': net_debit,
            'max_profit': 'Limited but variable',  # Depends on volatility and price movement
            'greeks': combined_greeks,
            'near_volatility': near_vol,
            'far_volatility': far_vol
        }
    
    def plot_strategy_comparison(self, strategies: List[Dict], price_range: float = 0.3):
        """Compare multiple strategies in a single plot"""
        current_price = self.current_price
        price_min = current_price * (1 - price_range)
        price_max = current_price * (1 + price_range)
        stock_prices = np.linspace(price_min, price_max, 200)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle(f"Strategy Comparison for {self.symbol}", fontsize=16, fontweight='bold')
        
        # Plot P&L for each strategy
        for strategy_info in strategies:
            pnl = self.calculate_pnl_advanced(strategy_info, stock_prices)
            ax1.plot(stock_prices, pnl, linewidth=2, label=strategy_info['strategy'])
        
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.axvline(x=current_price, color='r', linestyle='--', alpha=0.7,
                   label=f'Current Price: ${current_price:.2f}')
        
        ax1.set_xlabel('Stock Price at Expiration ($)')
        ax1.set_ylabel('Profit/Loss ($)')
        ax1.set_title('P&L Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot key metrics comparison
        strategy_names = [s['strategy'] for s in strategies]
        metrics = {
            'Max Profit': [],
            'Max Loss': [],
            'Vega': [],
            'Theta': []
        }
        
        for strategy_info in strategies:
            # Max profit
            if isinstance(strategy_info.get('max_profit', 'Unlimited'), str):
                metrics['Max Profit'].append(float('nan'))  # Can't plot "Unlimited"
            else:
                metrics['Max Profit'].append(float(strategy_info.get('max_profit', 0)))
            
            # Max loss
            if isinstance(strategy_info.get('max_loss', 'Unlimited'), str):
                metrics['Max Loss'].append(float('nan'))
            else:
                metrics['Max Loss'].append(float(strategy_info.get('max_loss', 0)))
            
            # Greeks
            if 'greeks' in strategy_info:
                metrics['Vega'].append(strategy_info['greeks'].get('vega', 0))
                metrics['Theta'].append(strategy_info['greeks'].get('theta', 0))
            else:
                metrics['Vega'].append(0)
                metrics['Theta'].append(0)
        
        # Create bar chart for metrics
        x = np.arange(len(strategy_names))
        width = 0.2
        
        # Filter out metrics with NaN values
        valid_metrics = {k: v for k, v in metrics.items() if not all(np.isnan(v))}
        
        for i, (metric_name, values) in enumerate(valid_metrics.items()):
            ax2.bar(x + (i - len(valid_metrics)/2 + 0.5) * width, values, width, label=metric_name)
        
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Value')
        ax2.set_title('Strategy Metrics Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def plot_greeks_surface(self, option_type: str = 'call', strike_pct_range: float = 0.2, 
                           days_range: List[int] = [7, 30, 60, 90, 180]):
        """Plot 3D surface of option Greeks across strikes and expirations"""
        current_price = self.current_price
        
        # Create strike price and days to expiry grid
        strike_min = current_price * (1 - strike_pct_range)
        strike_max = current_price * (1 + strike_pct_range)
        strikes = np.linspace(strike_min, strike_max, 20)
        
        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(strikes, days_range)
        Z_delta = np.zeros_like(X, dtype=float)
        Z_gamma = np.zeros_like(X, dtype=float)
        Z_theta = np.zeros_like(X, dtype=float)
        Z_vega = np.zeros_like(X, dtype=float)
        
        # Calculate Greeks for each point
        for i, days in enumerate(days_range):
            for j, strike in enumerate(strikes):
                T = days / 365
                vol = self.vol_term_structure.get(days, self.volatility)
                
                greeks = self.calculate_all_greeks(current_price, strike, T,
                                                self.risk_free_rate, vol, option_type, self.dividend_yield)
                
                Z_delta[i, j] = greeks['delta']
                Z_gamma[i, j] = greeks['gamma']
                Z_theta[i, j] = greeks['theta']
                Z_vega[i, j] = greeks['vega']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f"{option_type.capitalize()} Option Greeks Surface for {self.symbol}", 
                    fontsize=16, fontweight='bold')
        
        # Delta surface
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_delta, cmap='viridis', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Days to Expiry')
        ax1.set_zlabel('Delta')
        ax1.set_title('Delta Surface')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # Gamma surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_gamma, cmap='plasma', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax2.set_xlabel('Strike Price')
        ax2.set_ylabel('Days to Expiry')
        ax2.set_zlabel('Gamma')
        ax2.set_title('Gamma Surface')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        # Theta surface
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        surf3 = ax3.plot_surface(X, Y, Z_theta, cmap='coolwarm', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax3.set_xlabel('Strike Price')
        ax3.set_ylabel('Days to Expiry')
        ax3.set_zlabel('Theta')
        ax3.set_title('Theta Surface')
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)
        
        # Vega surface
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        surf4 = ax4.plot_surface(X, Y, Z_vega, cmap='magma', alpha=0.8, 
                               linewidth=0, antialiased=True)
        ax4.set_xlabel('Strike Price')
        ax4.set_ylabel('Days to Expiry')
        ax4.set_zlabel('Vega')
        ax4.set_title('Vega Surface')
        fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def risk_reversal(self, put_strike: float, call_strike: float, expiry_days: int) -> Dict:
        """Risk Reversal Strategy (Short Put, Long Call)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        # Calculate premiums
        put_premium = self.black_scholes_price(self.current_price, put_strike, T,
                                             self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        call_premium = self.black_scholes_price(self.current_price, call_strike, T,
                                              self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Calculate Greeks
        put_greeks = self.calculate_all_greeks(self.current_price, put_strike, T,
                                             self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        call_greeks = self.calculate_all_greeks(self.current_price, call_strike, T,
                                              self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Combined Greeks (short put, long call)
        combined_greeks = {
            'delta': -put_greeks['delta'] + call_greeks['delta'],
            'gamma': -put_greeks['gamma'] + call_greeks['gamma'],
            'theta': -put_greeks['theta'] + call_greeks['theta'],
            'vega': -put_greeks['vega'] + call_greeks['vega'],
            'rho': -put_greeks['rho'] + call_greeks['rho']
        }
        
        # Net cost (negative if credit)
        net_cost = call_premium - put_premium
        
        return {
            'strategy': 'Risk Reversal',
            'put_strike': put_strike,
            'call_strike': call_strike,
            'expiry_days': expiry_days,
            'put_premium': put_premium,
            'call_premium': call_premium,
            'net_cost': net_cost,
            'max_profit': 'Unlimited',
            'max_loss': 'Limited to put strike - net cost',
            'breakeven_down': put_strike - net_cost,
            'breakeven_up': call_strike + net_cost,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def backratio_spread(self, front_strike: float, back_strike: float, ratio: float, 
                        expiry_days: int, option_type: str = 'call') -> Dict:
        """Back Ratio Spread (1 short front strike, ratio long back strikes)"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        # Calculate premiums
        front_premium = self.black_scholes_price(self.current_price, front_strike, T,
                                               self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        back_premium = self.black_scholes_price(self.current_price, back_strike, T,
                                              self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        # Calculate Greeks
        front_greeks = self.calculate_all_greeks(self.current_price, front_strike, T,
                                               self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        back_greeks = self.calculate_all_greeks(self.current_price, back_strike, T,
                                              self.risk_free_rate, vol, option_type, self.dividend_yield)
        
        # Combined Greeks (short front, long back * ratio)
        combined_greeks = {
            'delta': -front_greeks['delta'] + (back_greeks['delta'] * ratio),
            'gamma': -front_greeks['gamma'] + (back_greeks['gamma'] * ratio),
            'theta': -front_greeks['theta'] + (back_greeks['theta'] * ratio),
            'vega': -front_greeks['vega'] + (back_greeks['vega'] * ratio),
            'rho': -front_greeks['rho'] + (back_greeks['rho'] * ratio)
        }
        
        # Net cost (negative if credit)
        net_cost = (back_premium * ratio) - front_premium
        
        # Max profit/loss calculations
        if option_type.lower() == 'call':
            # For call backratio
            max_profit = 'Unlimited'
            max_loss = (back_strike - front_strike) + net_cost
            breakeven_low = front_strike + net_cost
            breakeven_high = back_strike + (net_cost / (ratio - 1)) if ratio > 1 else None
        else:
            # For put backratio
            max_profit = (front_strike - back_strike) * (ratio - 1) - net_cost
            max_loss = back_strike + net_cost
            breakeven_low = back_strike - (net_cost / (ratio - 1)) if ratio > 1 else None
            breakeven_high = front_strike - net_cost
        
        return {
            'strategy': f'{option_type.capitalize()} Back Ratio {1}:{ratio}',
            'front_strike': front_strike,
            'back_strike': back_strike,
            'ratio': ratio,
            'option_type': option_type,
            'expiry_days': expiry_days,
            'front_premium': front_premium,
            'back_premium': back_premium,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven_low': breakeven_low,
            'breakeven_high': breakeven_high,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def condor(self, lowest_strike: float, lower_middle_strike: float, 
              upper_middle_strike: float, highest_strike: float, expiry_days: int) -> Dict:
        """Iron Condor Strategy"""
        T = expiry_days / 365
        vol = self.vol_term_structure.get(expiry_days, self.volatility)
        
        # Calculate premiums
        long_put_premium = self.black_scholes_price(self.current_price, lowest_strike, T,
                                                  self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        short_put_premium = self.black_scholes_price(self.current_price, lower_middle_strike, T,
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        short_call_premium = self.black_scholes_price(self.current_price, upper_middle_strike, T,
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        long_call_premium = self.black_scholes_price(self.current_price, highest_strike, T,
                                                   self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Net credit
        net_credit = (short_put_premium + short_call_premium) - (long_put_premium + long_call_premium)
        
        # Calculate max profit/loss
        put_wing_width = lower_middle_strike - lowest_strike
        call_wing_width = highest_strike - upper_middle_strike
        max_loss = min(put_wing_width, call_wing_width) - net_credit
        
        # Calculate Greeks
        long_put_greeks = self.calculate_all_greeks(self.current_price, lowest_strike, T,
                                                  self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        short_put_greeks = self.calculate_all_greeks(self.current_price, lower_middle_strike, T,
                                                   self.risk_free_rate, vol, 'put', self.dividend_yield)
        
        short_call_greeks = self.calculate_all_greeks(self.current_price, upper_middle_strike, T,
                                                    self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        long_call_greeks = self.calculate_all_greeks(self.current_price, highest_strike, T,
                                                   self.risk_free_rate, vol, 'call', self.dividend_yield)
        
        # Combined Greeks
        combined_greeks = {
            'delta': (long_put_greeks['delta'] - short_put_greeks['delta'] - 
                     short_call_greeks['delta'] + long_call_greeks['delta']),
            'gamma': (long_put_greeks['gamma'] - short_put_greeks['gamma'] - 
                     short_call_greeks['gamma'] + long_call_greeks['gamma']),
            'theta': (long_put_greeks['theta'] - short_put_greeks['theta'] - 
                     short_call_greeks['theta'] + long_call_greeks['theta']),
            'vega': (long_put_greeks['vega'] - short_put_greeks['vega'] - 
                    short_call_greeks['vega'] + long_call_greeks['vega']),
            'rho': (long_put_greeks['rho'] - short_put_greeks['rho'] - 
                   short_call_greeks['rho'] + long_call_greeks['rho'])
        }
        
        # Calculate probability of profit
        prob_below_lower = norm.cdf(np.log(lower_middle_strike / self.current_price) / (vol * np.sqrt(T)))
        prob_above_upper = 1 - norm.cdf(np.log(upper_middle_strike / self.current_price) / (vol * np.sqrt(T)))
        prob_profit = prob_below_lower + prob_above_upper
        
        return {
            'strategy': 'Iron Condor',
            'lowest_strike': lowest_strike,
            'lower_middle_strike': lower_middle_strike,
            'upper_middle_strike': upper_middle_strike,
            'highest_strike': highest_strike,
            'expiry_days': expiry_days,
            'net_credit': net_credit,
            'max_profit': net_credit,
            'max_loss': max_loss,
            'breakeven_low': lower_middle_strike - net_credit,
            'breakeven_high': upper_middle_strike + net_credit,
            'probability_profit': prob_profit,
            'greeks': combined_greeks,
            'volatility_used': vol
        }
    
    def plot_advanced_strategy(self, strategy_info: Dict, price_range: float = 0.4):
        """Advanced strategy plotting with multiple visualizations"""
        current_price = self.current_price
        price_min = current_price * (1 - price_range)
        price_max = current_price * (1 + price_range)
        stock_prices = np.linspace(price_min, price_max, 200)
        
        pnl = self.calculate_pnl_advanced(strategy_info, stock_prices)
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f"Advanced Analysis: {strategy_info['strategy']} for {self.symbol}", fontsize=16, fontweight='bold')
        
        # 1. P&L Diagram
        ax1.plot(stock_prices, pnl, 'b-', linewidth=3, label='P&L at Expiration')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.axvline(x=current_price, color='r', linestyle='--', alpha=0.7,
                   label=f'Current Price: ${current_price:.2f}')
        
        # Add breakeven lines
        if 'breakeven' in strategy_info:
            ax1.axvline(x=strategy_info['breakeven'], color='g', linestyle=':',
                       label=f"Breakeven: ${strategy_info['breakeven']:.2f}")
        elif 'breakeven_low' in strategy_info and 'breakeven_high' in strategy_info:
            ax1.axvline(x=strategy_info['breakeven_low'], color='g', linestyle=':',
                       label=f"BE Low: ${strategy_info['breakeven_low']:.2f}")
            ax1.axvline(x=strategy_info['breakeven_high'], color='g', linestyle=':',
                       label=f"BE High: ${strategy_info['breakeven_high']:.2f}")
        
        # Profit/Loss zones
        profit_mask = pnl > 0
        loss_mask = pnl < 0
        ax1.fill_between(stock_prices, 0, pnl, where=profit_mask, alpha=0.3, color='green', label='Profit Zone')
        ax1.fill_between(stock_prices, 0, pnl, where=loss_mask, alpha=0.3, color='red', label='Loss Zone')
        
        ax1.set_xlabel('Stock Price at Expiration ($)')
        ax1.set_ylabel('Profit/Loss ($)')
        ax1.set_title('P&L Diagram')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Greeks Evolution
        if 'greeks' in strategy_info:
            days_to_expiry = np.linspace(strategy_info.get('expiry_days', 30), 1, 50)
            greeks_evolution = {'delta': [], 'gamma': [], 'theta': [], 'vega': []}
            
            for days in days_to_expiry:
                T = days / 365
                if strategy_info['strategy'] == 'Long Call':
                    temp_greeks = self.calculate_all_greeks(
                        current_price, strategy_info.get('strike', current_price), T,
                        self.risk_free_rate, strategy_info.get('volatility_used', self.volatility),
                        'call', self.dividend_yield
                    )
                else:
                    # For complex strategies, use current Greeks values
                    temp_greeks = strategy_info['greeks']
                
                for greek in greeks_evolution:
                    greeks_evolution[greek].append(temp_greeks.get(greek, 0))
            
            for greek, values in greeks_evolution.items():
                ax2.plot(days_to_expiry, values, label=f'{greek.capitalize()}', linewidth=2)
            
            ax2.set_xlabel('Days to Expiration')
            ax2.set_ylabel('Greek Values')
            ax2.set_title('Greeks Evolution Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Volatility Surface Impact (Fixed)
        vol_range = np.linspace(0.1, 0.6, 50)
        pnl_vol_impact = self.calculate_volatility_impact(strategy_info, vol_range)
        
        ax3.plot(vol_range * 100, pnl_vol_impact, 'purple', linewidth=3)
        ax3.axvline(x=strategy_info.get('volatility_used', self.volatility) * 100,
                   color='r', linestyle='--', label='Current Vol')
        ax3.set_xlabel('Implied Volatility (%)')
        ax3.set_ylabel('P&L Impact ($)')
        ax3.set_title('Volatility Sensitivity Analysis')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Risk Metrics Dashboard
        ax4.axis('off')
        
        # Create risk metrics text
        risk_text = f"""
        RISK METRICS DASHBOARD
        {'='*30}
        
        Strategy: {strategy_info['strategy']}
        Current Stock Price: ${current_price:.2f}
        
        PROFIT/LOSS ANALYSIS:
        Max Profit: {strategy_info.get('max_profit', 'N/A')}
        Max Loss: ${strategy_info.get('max_loss', 'N/A'):.2f}
        
        PROBABILITY ANALYSIS:
        Prob. of Profit: {strategy_info.get('probability_profit', 0)*100:.1f}%
        
        GREEKS (Current):
        Delta: {strategy_info.get('greeks', {}).get('delta', 0):.4f}
        Gamma: {strategy_info.get('greeks', {}).get('gamma', 0):.4f}
        Theta: {strategy_info.get('greeks', {}).get('theta', 0):.4f}
        Vega: {strategy_info.get('greeks', {}).get('vega', 0):.4f}
        
        VOLATILITY:
        Current Vol: {strategy_info.get('volatility_used', self.volatility)*100:.1f}%
        """
        
        ax4.text(0.05, 0.95, risk_text, fontsize=12, va='top', family='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def monte_carlo_simulation(self, strategy_info: Dict, days: int = 30, 
                              num_simulations: int = 1000, confidence_level: float = 0.95):
        """Run Monte Carlo simulation to estimate strategy P&L distribution"""
        # Parameters
        S0 = self.current_price
        vol = strategy_info.get('volatility_used', self.volatility)
        r = self.risk_free_rate
        T = days / 365
        dt = 1/252  # Daily steps
        steps = int(days * 1)  # Number of steps in simulation
        
        # Initialize price paths
        price_paths = np.zeros((num_simulations, steps + 1))
        price_paths[:, 0] = S0
        
        # Generate random paths
        for t in range(1, steps + 1):
            Z = np.random.standard_normal(num_simulations)
            price_paths[:, t] = price_paths[:, t-1] * np.exp((r - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z)
        
        # Calculate P&L for each path
        final_prices = price_paths[:, -1]
        pnl_results = self.calculate_pnl_advanced(strategy_info, final_prices)
        
        # Calculate statistics
        mean_pnl = np.mean(pnl_results)
        std_pnl = np.std(pnl_results)
        min_pnl = np.min(pnl_results)
        max_pnl = np.max(pnl_results)
        
        # Calculate Value at Risk (VaR)
        var = np.percentile(pnl_results, (1 - confidence_level) * 100)
        
        # Calculate Expected Shortfall (ES) / Conditional VaR
        es = np.mean(pnl_results[pnl_results <= var])
        
        # Probability of profit
        prob_profit = np.mean(pnl_results > 0)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Monte Carlo Simulation: {strategy_info['strategy']} for {self.symbol}", 
                    fontsize=16, fontweight='bold')
        
        # Plot price paths (sample)
        sample_paths = price_paths[np.random.choice(num_simulations, 100, replace=False)]
        for path in sample_paths:
            ax1.plot(path, linewidth=0.5, alpha=0.3, color='blue')
        
        ax1.plot(np.mean(price_paths, axis=0), linewidth=2, color='red', label='Mean Path')
        ax1.axhline(y=S0, color='k', linestyle='--', alpha=0.5, label=f'Current Price: ${S0:.2f}')
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Stock Price ($)')
        ax1.set_title(f'Sample of {len(sample_paths)} Price Paths')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot P&L distribution
        ax2.hist(pnl_results, bins=50, alpha=0.7, color='green', density=True)
        
        # Add VaR line
        ax2.axvline(x=var, color='r', linestyle='--', 
                   label=f'VaR ({confidence_level*100}%): ${var:.2f}')
        
        # Add mean P&L line
        ax2.axvline(x=mean_pnl, color='blue', linestyle='-', 
                   label=f'Mean P&L: ${mean_pnl:.2f}')
        
        ax2.set_xlabel('Profit/Loss ($)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('P&L Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add text box with statistics
        stats_text = f"""
        SIMULATION STATISTICS
        {'='*25}
        Simulations: {num_simulations}
        Time Horizon: {days} days
        
        P&L METRICS:
        Mean P&L: ${mean_pnl:.2f}
        Std Dev: ${std_pnl:.2f}
        Min P&L: ${min_pnl:.2f}
        Max P&L: ${max_pnl:.2f}
        
        RISK METRICS:
        VaR ({confidence_level*100}%): ${-var:.2f}
        Expected Shortfall: ${-es:.2f}
        Prob. of Profit: {prob_profit*100:.1f}%
        """
        
        # Place text box in the center of the figure
        fig.text(0.5, 0.02, stats_text, fontsize=12, ha='center', 
                va='bottom', bbox=dict(facecolor='white', alpha=0.8), family='monospace')
        
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.show()
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'min_pnl': min_pnl,
            'max_pnl': max_pnl,
            'var': -var,
            'expected_shortfall': -es,
            'probability_profit': prob_profit
        }
    
    def optimize_strategy_parameters(self, strategy_type: str, objective: str = 'max_profit', 
                                   constraints: Dict = None):
        """Optimize strategy parameters based on objective function"""
        if constraints is None:
            constraints = {}
        
        results = []
        
        if strategy_type == 'iron_butterfly':
            # Define parameter ranges
            center_strikes = np.linspace(self.current_price * 0.9, self.current_price * 1.1, 10)
            wing_widths = np.linspace(self.current_price * 0.05, self.current_price * 0.2, 10)
            expiry_days_options = [30, 45, 60, 90]
            
            for center in center_strikes:
                for width in wing_widths:
                    for days in expiry_days_options:
                        strategy = self.iron_butterfly(center, width, days)
                        
                        # Apply constraints
                        if 'max_loss' in constraints and strategy.get('max_loss', float('inf')) > constraints['max_loss']:
                            continue
                        
                        if 'min_profit' in constraints and strategy.get('max_profit', 0) < constraints['min_profit']:
                            continue
                        
                        # Calculate objective value
                        if objective == 'max_profit':
                            obj_value = strategy.get('max_profit', 0)
                        elif objective == 'risk_reward':
                            max_profit = strategy.get('max_profit', 0)
                            max_loss = strategy.get('max_loss', float('inf'))
                            obj_value = max_profit / max_loss if max_loss > 0 else 0
                        elif objective == 'probability_profit':
                            obj_value = strategy.get('probability_profit', 0)
                        else:
                            obj_value = 0
                        
                        results.append({
                            'parameters': {
                                'center_strike': center,
                                'wing_width': width,
                                'expiry_days': days
                            },
                            'strategy': strategy,
                            'objective_value': obj_value
                        })
        
        elif strategy_type == 'condor':
            # Define parameter ranges
            width_pcts = np.linspace(0.05, 0.2, 8)  # Width as percentage of stock price
            spread_pcts = np.linspace(0.02, 0.1, 5)  # Spread between short strikes
            expiry_days_options = [30, 45, 60, 90]
            
            for width_pct in width_pcts:
                for spread_pct in spread_pcts:
                    for days in expiry_days_options:
                        # Calculate strikes
                        middle_spread = self.current_price * spread_pct
                        wing_width = self.current_price * width_pct
                        
                        lower_middle = self.current_price - (middle_spread / 2)
                        upper_middle = self.current_price + (middle_spread / 2)
                        lowest = lower_middle - wing_width
                        highest = upper_middle + wing_width
                        
                        strategy = self.condor(lowest, lower_middle, upper_middle, highest, days)
                        
                        # Apply constraints
                        if 'max_loss' in constraints and strategy.get('max_loss', float('inf')) > constraints['max_loss']:
                            continue
                        
                        if 'min_credit' in constraints and strategy.get('net_credit', 0) < constraints['min_credit']:
                            continue
                        
                        # Calculate objective value
                        if objective == 'max_credit':
                            obj_value = strategy.get('net_credit', 0)
                        elif objective == 'probability_profit':
                            obj_value = strategy.get('probability_profit', 0)
                        elif objective == 'risk_reward':
                            net_credit = strategy.get('net_credit', 0)
                            max_loss = strategy.get('max_loss', float('inf'))
                            obj_value = net_credit / max_loss if max_loss > 0 else 0
                        else:
                            obj_value = 0
                        
                        results.append({
                            'parameters': {
                                'lowest_strike': lowest,
                                'lower_middle_strike': lower_middle,
                                'upper_middle_strike': upper_middle,
                                'highest_strike': highest,
                                'expiry_days': days
                            },
                            'strategy': strategy,
                            'objective_value': obj_value
                        })
        
        # Sort results by objective value (descending)
        results.sort(key=lambda x: x['objective_value'], reverse=True)
        
        # Return top 3 strategies
        return results[:3] if results else []

# Create an instance of the strategy generator
strategy_generator = AdvancedOptionsStrategyGenerator(symbol='AAPL')

# Fetch stock data
strategy_generator.fetch_stock_data(period="1y")

# Example 1: Create and analyze a Long Call strategy
print("\n=== Long Call Strategy Analysis ===")
long_call = strategy_generator.long_call(
    strike=strategy_generator.current_price * 1.05,  # 5% OTM call
    expiry_days=45,
    quantity=1
)
print(f"Strategy: {long_call['strategy']}")
print(f"Strike Price: ${long_call['strike']:.2f}")
print(f"Premium: ${long_call['premium']:.2f}")
print(f"Max Loss: ${long_call['max_loss']:.2f}")
print(f"Breakeven: ${long_call['breakeven']:.2f}")
print(f"Probability of Profit: {long_call['probability_profit']*100:.1f}%")

# Print Greeks
print("\nGreeks:")
for greek, value in long_call['greeks'].items():
    print(f"{greek.capitalize()}: {value:.4f}")

# Visualize the Long Call strategy
strategy_generator.plot_advanced_strategy(long_call)

# Example 2: Create and analyze a Covered Call strategy
print("\n=== Covered Call Strategy Analysis ===")
covered_call = strategy_generator.covered_call(
    strike=strategy_generator.current_price * 1.1,  # 10% OTM call
    expiry_days=30,
    shares_owned=100
)
print(f"Strategy: {covered_call['strategy']}")
print(f"Strike Price: ${covered_call['strike']:.2f}")
print(f"Premium Received: ${covered_call['premium_received']:.2f}")
print(f"Max Profit: ${covered_call['max_profit']:.2f}")
print(f"Max Loss: ${covered_call['max_loss']:.2f}")
print(f"Breakeven: ${covered_call['breakeven']:.2f}")
print(f"Annualized Return: {covered_call['annualized_return']*100:.2f}%")

# Print Greeks
print("\nGreeks:")
for greek, value in covered_call['greeks'].items():
    print(f"{greek.capitalize()}: {value:.4f}")

# Visualize the Covered Call strategy
strategy_generator.plot_advanced_strategy(covered_call)

# Example 3: Create and analyze an Iron Butterfly strategy
print("\n=== Iron Butterfly Strategy Analysis ===")
iron_butterfly = strategy_generator.iron_butterfly(
    center_strike=strategy_generator.current_price,  # ATM
    wing_width=strategy_generator.current_price * 0.1,  # 10% wings
    expiry_days=45
)
print(f"Strategy: {iron_butterfly['strategy']}")
print(f"Center Strike: ${iron_butterfly['center_strike']:.2f}")
print(f"Wing Width: ${iron_butterfly['wing_width']:.2f}")
print(f"Net Credit: ${iron_butterfly['net_credit']:.2f}")
print(f"Max Profit: ${iron_butterfly['max_profit']:.2f}")
print(f"Max Loss: ${iron_butterfly['max_loss']:.2f}")
print(f"Breakeven Low: ${iron_butterfly['breakeven_low']:.2f}")
print(f"Breakeven High: ${iron_butterfly['breakeven_high']:.2f}")
print(f"Probability of Profit: {iron_butterfly['probability_profit']*100:.1f}%")

# Print Greeks
print("\nGreeks:")
for greek, value in iron_butterfly['greeks'].items():
    print(f"{greek.capitalize()}: {value:.4f}")

# Visualize the Iron Butterfly strategy
strategy_generator.plot_advanced_strategy(iron_butterfly)

# Example 4: Create and analyze a Jade Lizard strategy
print("\n=== Jade Lizard Strategy Analysis ===")
jade_lizard = strategy_generator.jade_lizard(
    short_put_strike=strategy_generator.current_price * 0.9,  # 10% OTM put
    short_call_strike=strategy_generator.current_price * 1.05,  # 5% OTM call
    long_call_strike=strategy_generator.current_price * 1.15,  # 15% OTM call
    expiry_days=45
)
print(f"Strategy: {jade_lizard['strategy']}")
print(f"Short Put Strike: ${jade_lizard['short_put_strike']:.2f}")
print(f"Short Call Strike: ${jade_lizard['short_call_strike']:.2f}")
print(f"Long Call Strike: ${jade_lizard['long_call_strike']:.2f}")
print(f"Net Credit: ${jade_lizard['net_credit']:.2f}")
print(f"Max Profit: ${jade_lizard['max_profit']:.2f}")
print(f"Max Loss Upside: ${jade_lizard['max_loss_upside']:.2f}")
print(f"Max Loss Downside: ${jade_lizard['max_loss_downside']:.2f}")
print(f"Breakeven: ${float(jade_lizard['breakeven']):.2f}")

# Visualize the Jade Lizard strategy
strategy_generator.plot_advanced_strategy(jade_lizard)

# Example 5: Run Monte Carlo simulation for the Iron Butterfly strategy
print("=== Monte Carlo Simulation for Iron Butterfly ===")
simulation_results = strategy_generator.monte_carlo_simulation(
    strategy_info=iron_butterfly,
    days=45,
    num_simulations=10000,
    confidence_level=0.95
)
print(f"Mean P&L: ${simulation_results['mean_pnl']:.2f}")
print(f"Standard Deviation: ${simulation_results['std_pnl']:.2f}")
print(f"Value at Risk (95%): ${simulation_results['var']:.2f}")
print(f"Probability of Profit: {simulation_results['probability_profit']*100:.1f}%")

# Visualize Monte Carlo simulation results
strategy_generator.plot_monte_carlo_results(simulation_results, "Iron Butterfly")

# Example 6: Optimize Iron Butterfly parameters
print("\n=== Optimizing Iron Butterfly Strategy ===")
optimized_strategies = strategy_generator.optimize_strategy_parameters(
    strategy_type='iron_butterfly',
    objective='probability_profit',
    constraints={'max_loss': 500}
)

if optimized_strategies:
    best_strategy = optimized_strategies[0]['strategy']
    print("Best Iron Butterfly Strategy:")
    print(f"Center Strike: ${best_strategy['center_strike']:.2f}")
    print(f"Wing Width: ${best_strategy['wing_width']:.2f}")
    print(f"Expiry Days: {best_strategy['expiry_days']}")
    print(f"Net Credit: ${best_strategy['net_credit']:.2f}")
    print(f"Max Profit: ${best_strategy['max_profit']:.2f}")
    print(f"Max Loss: ${best_strategy['max_loss']:.2f}")
    print(f"Probability of Profit: {best_strategy['probability_profit']*100:.1f}%")
    
    # Visualize the optimized strategy
    strategy_generator.plot_advanced_strategy(best_strategy)
else:
    print("No strategies found that meet the constraints.")

# Example 7: Portfolio Analysis with Multiple Strategies
print("\n=== Portfolio Analysis with Multiple Strategies ===")

# Create a portfolio of strategies
portfolio = [
    long_call,
    covered_call,
    iron_butterfly
]

# Calculate portfolio-level Greeks
portfolio_greeks = strategy_generator.portfolio_greeks(portfolio)

print("Portfolio Greeks:")
for greek, value in portfolio_greeks.items():
    print(f"{greek.capitalize()}: {value:.4f}")

# Calculate portfolio-level risk metrics
total_max_loss = sum(strategy.get('max_loss', 0) for strategy in portfolio if isinstance(strategy.get('max_loss'), (int, float)))
print(f"Total Maximum Loss: ${total_max_loss:.2f}")

# Example 8: Risk Management - Delta Hedging
print("\n=== Risk Management - Delta Hedging ===")

# Calculate number of shares needed to delta hedge the iron butterfly
delta_to_hedge = iron_butterfly['greeks']['delta']
shares_needed = int(delta_to_hedge * 100)  # Each option contract represents 100 shares

print(f"Iron Butterfly Delta: {delta_to_hedge:.4f}")
if delta_to_hedge > 0:
    print(f"To delta hedge, short {abs(shares_needed)} shares of {strategy_generator.symbol}")
else:
    print(f"To delta hedge, buy {abs(shares_needed)} shares of {strategy_generator.symbol}")

# Example 9: Volatility Analysis
print("\n=== Volatility Analysis ===")

# Compare historical vs. implied volatility
print(f"Historical Volatility: {strategy_generator.volatility:.2%}")
print(f"GARCH Volatility: {strategy_generator.garch_vol:.2%}")

# Print volatility term structure
print("\nVolatility Term Structure:")
for days, vol in strategy_generator.vol_term_structure.items():
    print(f"{days} days: {vol:.2%}")

# Example 10: Earnings Analysis
print("\n=== Pre-Earnings Strategy Analysis ===")

# Simulate higher implied volatility before earnings
earnings_vol = strategy_generator.volatility * 1.5

# Create a strangle strategy for earnings
strangle_call_strike = strategy_generator.current_price * 1.1
strangle_put_strike = strategy_generator.current_price * 0.9
expiry_days = 30

# Calculate call premium with elevated IV
call_premium = strategy_generator.black_scholes_price(
    strategy_generator.current_price, 
    strangle_call_strike, 
    expiry_days/365, 
    strategy_generator.risk_free_rate, 
    earnings_vol, 
    'call', 
    strategy_generator.dividend_yield
)

# Calculate put premium with elevated IV
put_premium = strategy_generator.black_scholes_price(
    strategy_generator.current_price, 
    strangle_put_strike, 
    expiry_days/365, 
    strategy_generator.risk_free_rate, 
    earnings_vol, 
    'put', 
    strategy_generator.dividend_yield
)

total_premium = call_premium + put_premium

print(f"Pre-Earnings Long Strangle Strategy:")
print(f"Call Strike: ${strangle_call_strike:.2f}")
print(f"Put Strike: ${strangle_put_strike:.2f}")
print(f"Call Premium: ${call_premium:.2f}")
print(f"Put Premium: ${put_premium:.2f}")
print(f"Total Premium (Cost): ${total_premium:.2f}")
print(f"Breakeven Points: ${strangle_put_strike - total_premium:.2f} and ${strangle_call_strike + total_premium:.2f}")
print(f"Expected IV Crush After Earnings: {earnings_vol:.2%} → {strategy_generator.volatility:.2%}")

# Calculate expected value after earnings (assuming IV crush but no price movement)
post_earnings_call_value = strategy_generator.black_scholes_price(
    strategy_generator.current_price, 
    strangle_call_strike, 
    expiry_days/365, 
    strategy_generator.risk_free_rate, 
    strategy_generator.volatility, 
    'call', 
    strategy_generator.dividend_yield
)

post_earnings_put_value = strategy_generator.black_scholes_price(
    strategy_generator.current_price, 
    strangle_put_strike, 
    expiry_days/365, 
    strategy_generator.risk_free_rate, 
    strategy_generator.volatility, 
    'put', 
    strategy_generator.dividend_yield
)

post_earnings_total = post_earnings_call_value + post_earnings_put_value
expected_iv_crush_loss = total_premium - post_earnings_total

print(f"Expected Value After IV Crush: ${post_earnings_total:.2f}")
print(f"Expected Loss from IV Crush: ${expected_iv_crush_loss:.2f} ({expected_iv_crush_loss/total_premium:.1%} of premium)")

print("\n=== Analysis Complete ===")
