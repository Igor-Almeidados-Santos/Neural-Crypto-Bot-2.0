"""
Orderbook Features Module

Contains classes and functions for calculating various orderbook-based features
that provide insights into market microstructure and liquidity.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from feature_engineering.domain.entities.feature import (
    Feature, FeatureType, FeatureCategory, FeatureScope, FeatureTimeframe, FeatureMetadata
)

logger = logging.getLogger(__name__)


class OrderbookFeatureCalculator:
    """Calculator for orderbook-based features."""
    
    def __init__(self, symbol: str, timeframe: FeatureTimeframe):
        self.symbol = symbol
        self.timeframe = timeframe
    
    def create_feature_metadata(
        self, 
        name: str,
        description: str, 
        category: FeatureCategory = FeatureCategory.LIQUIDITY,
        scope: FeatureScope = FeatureScope.SINGLE_ASSET,
        is_experimental: bool = False,
        tags: List[str] = None,
        properties: Dict[str, any] = None
    ) -> FeatureMetadata:
        """Create standardized metadata for orderbook features."""
        if tags is None:
            tags = []
        if properties is None:
            properties = {}
            
        # Add standard tags for orderbook features
        tags.extend(["orderbook", "market_microstructure", category.name.lower()])
        
        return FeatureMetadata(
            description=description,
            category=category,
            scope=scope,
            timeframe=self.timeframe,
            is_experimental=is_experimental,
            tags=tags,
            properties=properties
        )
    
    def calculate_bid_ask_spread(self, orderbook: Dict) -> Feature:
        """
        Calculate the bid-ask spread
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
                       Example: {'bids': [[price, qty], ...], 'asks': [[price, qty], ...]}
            
        Returns:
            Feature object containing the bid-ask spread value
        """
        try:
            # Extract best bid and ask
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            
            # Calculate spread
            spread = best_ask - best_bid
            
            # Calculate spread percentage
            spread_pct = (spread / best_bid) * 100
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name="BID_ASK_SPREAD",
                description="Bid-Ask Spread percentage",
                category=FeatureCategory.LIQUIDITY,
                tags=["spread", "liquidity"],
                properties={"absolute_spread": float(spread)}
            )
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create and return feature
            return Feature.create(
                name="BID_ASK_SPREAD",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(spread_pct),
                metadata=metadata,
                lookback_periods=0,  # Only current orderbook needed
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating bid-ask spread: {str(e)}")
            raise
    
    def calculate_order_book_imbalance(self, orderbook: Dict, depth: int = 10) -> Feature:
        """
        Calculate the order book imbalance (ratio of bid to ask volume)
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
            depth: Number of price levels to consider (default: 10)
            
        Returns:
            Feature object containing the imbalance value
        """
        try:
            # Ensure we don't exceed available depth
            depth = min(depth, len(orderbook['bids']), len(orderbook['asks']))
            
            # Calculate total volume on bid and ask sides to specified depth
            bid_volume = sum(qty for price, qty in orderbook['bids'][:depth])
            ask_volume = sum(qty for price, qty in orderbook['asks'][:depth])
            
            # Calculate imbalance ratio (>1 means more bids than asks)
            if ask_volume == 0:
                imbalance = 100.0  # Avoid division by zero, indicate extreme imbalance
            else:
                imbalance = bid_volume / ask_volume
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"OB_IMBALANCE-{depth}",
                description=f"Order Book Imbalance (bid/ask ratio) to depth {depth}",
                category=FeatureCategory.LIQUIDITY,
                tags=["imbalance", "liquidity", "orderbook_depth"],
                properties={
                    "depth": depth,
                    "bid_volume": float(bid_volume),
                    "ask_volume": float(ask_volume)
                }
            )
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create and return feature
            return Feature.create(
                name=f"OB_IMBALANCE-{depth}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(imbalance),
                metadata=metadata,
                lookback_periods=0,  # Only current orderbook needed
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating order book imbalance: {str(e)}")
            raise
    
    def calculate_liquidity_at_price(self, orderbook: Dict, price_level_pct: float = 1.0) -> Feature:
        """
        Calculate available liquidity at specified price level
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
            price_level_pct: Percentage from mid price to calculate liquidity (default: 1.0%)
            
        Returns:
            Feature object containing the liquidity value
        """
        try:
            # Extract best bid and ask
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            
            # Calculate mid price
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate price levels
            bid_price_level = mid_price * (1 - price_level_pct / 100)
            ask_price_level = mid_price * (1 + price_level_pct / 100)
            
            # Sum liquidity available at or better than specified price levels
            bid_liquidity = sum(qty for price, qty in orderbook['bids'] if price >= bid_price_level)
            ask_liquidity = sum(qty for price, qty in orderbook['asks'] if price <= ask_price_level)
            
            # Total liquidity at specified price range
            total_liquidity = bid_liquidity + ask_liquidity
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"LIQUIDITY_AT_{price_level_pct}PCT",
                description=f"Liquidity available within {price_level_pct}% of mid price",
                category=FeatureCategory.LIQUIDITY,
                tags=["liquidity", "depth", "slippage"],
                properties={
                    "price_level_pct": price_level_pct,
                    "bid_liquidity": float(bid_liquidity),
                    "ask_liquidity": float(ask_liquidity),
                    "mid_price": float(mid_price)
                }
            )
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create and return feature
            return Feature.create(
                name=f"LIQUIDITY_AT_{price_level_pct}PCT",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(total_liquidity),
                metadata=metadata,
                lookback_periods=0,  # Only current orderbook needed
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating liquidity at price: {str(e)}")
            raise
    
    def calculate_market_impact(self, orderbook: Dict, trade_size: float) -> Feature:
        """
        Calculate estimated market impact for a given trade size
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
            trade_size: Size of the trade in base currency units
            
        Returns:
            Feature object containing the estimated price impact percentage
        """
        try:
            # Determine if buying or selling based on trade_size sign
            is_buying = trade_size > 0
            
            # Get the relevant side of the book
            book_side = orderbook['asks'] if is_buying else orderbook['bids']
            trade_size = abs(trade_size)
            
            # Get reference price (best ask for buys, best bid for sells)
            reference_price = book_side[0][0]
            
            # Calculate cumulative notional and executed quantity
            remaining_qty = trade_size
            notional_value = 0
            
            for price, qty in book_side:
                executable_qty = min(remaining_qty, qty)
                notional_value += executable_qty * price
                remaining_qty -= executable_qty
                
                if remaining_qty <= 0:
                    break
            
            # Calculate effective price if we can fill the entire order
            if remaining_qty <= 0:
                effective_price = notional_value / trade_size
                
                # Calculate price impact as percentage
                if is_buying:
                    impact_pct = ((effective_price - reference_price) / reference_price) * 100
                else:
                    impact_pct = ((reference_price - effective_price) / reference_price) * 100
            else:
                # If we can't fill the entire order, report maximum impact
                impact_pct = 100.0
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"MARKET_IMPACT-{trade_size}",
                description=f"Estimated market impact for trade size {trade_size}",
                category=FeatureCategory.LIQUIDITY,
                tags=["market_impact", "slippage", "execution"],
                properties={
                    "trade_size": float(trade_size),
                    "direction": "buy" if is_buying else "sell",
                    "reference_price": float(reference_price),
                    "unfilled_quantity": float(max(0, remaining_qty))
                }
            )
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create and return feature
            return Feature.create(
                name=f"MARKET_IMPACT-{trade_size}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(impact_pct),
                metadata=metadata,
                lookback_periods=0,  # Only current orderbook needed
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating market impact: {str(e)}")
            raise
    
    def calculate_order_book_pressure(self, orderbook: Dict, levels: int = 5) -> Feature:
        """
        Calculate order book pressure indicator
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
            levels: Number of price levels to consider (default: 5)
            
        Returns:
            Feature object containing the pressure indicator value
        """
        try:
            # Ensure we don't exceed available depth
            levels = min(levels, len(orderbook['bids']), len(orderbook['asks']))
            
            # Extract price and quantity at specified levels
            bids = orderbook['bids'][:levels]
            asks = orderbook['asks'][:levels]
            
            # Calculate price-weighted volume
            bid_pressure = sum(price * qty for price, qty in bids)
            ask_pressure = sum(price * qty for price, qty in asks)
            
            # Calculate pressure ratio (-1 to +1, where positive means more buying pressure)
            total_pressure = bid_pressure + ask_pressure
            if total_pressure == 0:
                pressure = 0
            else:
                pressure = (bid_pressure - ask_pressure) / total_pressure
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"OB_PRESSURE-{levels}",
                description=f"Order Book Pressure Indicator to depth {levels}",
                category=FeatureCategory.LIQUIDITY,
                tags=["pressure", "orderbook_depth", "buying_selling_power"],
                properties={
                    "levels": levels,
                    "bid_pressure": float(bid_pressure),
                    "ask_pressure": float(ask_pressure)
                }
            )
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create and return feature
            return Feature.create(
                name=f"OB_PRESSURE-{levels}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(pressure),
                metadata=metadata,
                lookback_periods=0,  # Only current orderbook needed
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating order book pressure: {str(e)}")
            raise
    
    def calculate_depth_curve_slope(self, orderbook: Dict, side: str = 'both', max_levels: int = 20) -> Feature:
        """
        Calculate the slope of the depth curve, indicating liquidity distribution
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
            side: Which side of the book to analyze ('bid', 'ask', or 'both')
            max_levels: Maximum number of price levels to consider
            
        Returns:
            Feature object containing the slope value
        """
        try:
            # Ensure we don't exceed available depth
            max_levels = min(max_levels, len(orderbook['bids']), len(orderbook['asks']))
            
            slopes = []
            
            # Calculate for bid side if requested
            if side in ['bid', 'both']:
                # Extract levels
                bids = orderbook['bids'][:max_levels]
                
                # Convert to arrays for linear regression
                bid_prices = np.array([price for price, _ in bids])
                bid_quantities = np.array([qty for _, qty in bids])
                
                # Cumulative quantities
                cum_bid_qty = np.cumsum(bid_quantities)
                
                # Calculate percentage change from best bid
                best_bid = bid_prices[0]
                price_pct_change = (bid_prices - best_bid) / best_bid * 100
                
                # Linear regression to find slope
                if len(bid_prices) > 1:
                    bid_slope = np.polyfit(price_pct_change, cum_bid_qty, 1)[0]
                    slopes.append(bid_slope)
            
            # Calculate for ask side if requested
            if side in ['ask', 'both']:
                # Extract levels
                asks = orderbook['asks'][:max_levels]
                
                # Convert to arrays for linear regression
                ask_prices = np.array([price for price, _ in asks])
                ask_quantities = np.array([qty for _, qty in asks])
                
                # Cumulative quantities
                cum_ask_qty = np.cumsum(ask_quantities)
                
                # Calculate percentage change from best ask
                best_ask = ask_prices[0]
                price_pct_change = (ask_prices - best_ask) / best_ask * 100
                
                # Linear regression to find slope
                if len(ask_prices) > 1:
                    ask_slope = np.polyfit(price_pct_change, cum_ask_qty, 1)[0]
                    slopes.append(ask_slope)
            
            # Calculate overall slope
            if slopes:
                avg_slope = np.mean(slopes)
            else:
                avg_slope = 0
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"DEPTH_SLOPE-{side}-{max_levels}",
                description=f"Order Book Depth Curve Slope for {side} side to depth {max_levels}",
                category=FeatureCategory.LIQUIDITY,
                tags=["depth_curve", "liquidity_distribution", "orderbook_analysis"],
                properties={
                    "side": side,
                    "max_levels": max_levels
                }
            )
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create and return feature
            return Feature.create(
                name=f"DEPTH_SLOPE-{side}-{max_levels}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(avg_slope),
                metadata=metadata,
                lookback_periods=0,  # Only current orderbook needed
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating depth curve slope: {str(e)}")
            raise
    
    def calculate_orderbook_volume_profile(self, orderbook: Dict, num_bins: int = 10, price_range_pct: float = 5.0) -> Dict[str, Feature]:
        """
        Calculate order book volume profile features
        
        Args:
            orderbook: Dictionary containing bids and asks with price and quantity
            num_bins: Number of price bins to divide the range into
            price_range_pct: Price range percentage from mid price to analyze
            
        Returns:
            Dictionary of volume profile Features
        """
        try:
            # Extract best bid and ask
            best_bid = orderbook['bids'][0][0]
            best_ask = orderbook['asks'][0][0]
            
            # Calculate mid price
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate price range
            min_price = mid_price * (1 - price_range_pct / 100)
            max_price = mid_price * (1 + price_range_pct / 100)
            
            # Create price bins
            price_bins = np.linspace(min_price, max_price, num_bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            
            # Initialize bin volumes
            bin_volumes = np.zeros(num_bins)
            
            # Aggregate bids into bins
            for price, qty in orderbook['bids']:
                if min_price <= price <= max_price:
                    bin_idx = np.digitize(price, price_bins) - 1
                    if 0 <= bin_idx < num_bins:
                        bin_volumes[bin_idx] += qty
            
            # Aggregate asks into bins
            for price, qty in orderbook['asks']:
                if min_price <= price <= max_price:
                    bin_idx = np.digitize(price, price_bins) - 1
                    if 0 <= bin_idx < num_bins:
                        bin_volumes[bin_idx] += qty
            
            # Calculate total volume and volume concentration metrics
            total_volume = np.sum(bin_volumes)
            if total_volume > 0:
                volume_concentration = np.max(bin_volumes) / total_volume
            else:
                volume_concentration = 0
            
            # Find price level with maximum volume
            max_volume_bin_idx = np.argmax(bin_volumes)
            max_volume_price = bin_centers[max_volume_bin_idx]
            
            # Calculate volume-weighted average price (VWAP) from orderbook
            if total_volume > 0:
                vwap = np.sum(bin_centers * bin_volumes) / total_volume
            else:
                vwap = mid_price
            
            # Create features
            features = {}
            
            # Get current timestamp
            timestamp = datetime.utcnow()
            
            # Create volume concentration feature
            concentration_metadata = self.create_feature_metadata(
                name=f"OB_VOL_CONCENTRATION-{num_bins}-{price_range_pct}",
                description=f"Order Book Volume Concentration over {price_range_pct}% range",
                category=FeatureCategory.LIQUIDITY,
                tags=["volume_profile", "concentration", "orderbook_analysis"],
                properties={
                    "num_bins": num_bins,
                    "price_range_pct": price_range_pct,
                    "total_volume": float(total_volume)
                }
            )
            
            features["volume_concentration"] = Feature.create(
                name=f"OB_VOL_CONCENTRATION-{num_bins}-{price_range_pct}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(volume_concentration),
                metadata=concentration_metadata,
                lookback_periods=0,
                timestamp=timestamp
            )
            
            # Create max volume price feature
            max_vol_price_metadata = self.create_feature_metadata(
                name=f"OB_MAX_VOL_PRICE-{num_bins}-{price_range_pct}",
                description=f"Order Book Price Level with Maximum Volume over {price_range_pct}% range",
                category=FeatureCategory.LIQUIDITY,
                tags=["volume_profile", "max_volume", "orderbook_analysis"],
                properties={
                    "num_bins": num_bins,
                    "price_range_pct": price_range_pct,
                    "max_volume": float(np.max(bin_volumes)) if len(bin_volumes) > 0 else 0.0,
                    "mid_price": float(mid_price)
                }
            )
            
            features["max_volume_price"] = Feature.create(
                name=f"OB_MAX_VOL_PRICE-{num_bins}-{price_range_pct}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(max_volume_price),
                metadata=max_vol_price_metadata,
                lookback_periods=0,
                timestamp=timestamp
            )
            
            # Create VWAP feature
            vwap_metadata = self.create_feature_metadata(
                name=f"OB_VWAP-{num_bins}-{price_range_pct}",
                description=f"Order Book Volume-Weighted Average Price over {price_range_pct}% range",
                category=FeatureCategory.LIQUIDITY,
                tags=["volume_profile", "vwap", "orderbook_analysis"],
                properties={
                    "num_bins": num_bins,
                    "price_range_pct": price_range_pct,
                    "mid_price": float(mid_price)
                }
            )
            
            features["vwap"] = Feature.create(
                name=f"OB_VWAP-{num_bins}-{price_range_pct}",
                symbol=self.symbol,
                type=FeatureType.ORDERBOOK,
                value=float(vwap),
                metadata=vwap_metadata,
                lookback_periods=0,
                timestamp=timestamp
            )
            
            return features
        except Exception as e:
            logger.error(f"Error calculating orderbook volume profile: {str(e)}")
            raise
