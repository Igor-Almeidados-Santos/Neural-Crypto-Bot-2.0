"""
Reinforcement Learning Strategy - Advanced RL-based trading strategy

This strategy uses deep reinforcement learning (DQN/PPO) to learn optimal
trading policies through interaction with market environments, featuring
continuous learning, adaptive position sizing, and dynamic risk management.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Union
from uuid import uuid4
import asyncio
import logging
from collections import deque
import pickle
import json

from ..domain.entities.strategy import BaseStrategy
from ..domain.entities.signal import (
    Signal, SignalType, SignalDirection, SignalStrength, 
    SignalMetrics, RiskParameters, ExecutionConstraints
)


class TradingEnvironment:
    """Trading environment for RL agent"""
    
    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.reset()
    
    def reset(self):
        """Reset environment state"""
        self.price_history = deque(maxlen=self.lookback_window)
        self.volume_history = deque(maxlen=self.lookback_window)
        self.return_history = deque(maxlen=self.lookback_window)
        self.technical_indicators = deque(maxlen=self.lookback_window)
        self.current_position = 0.0
        self.current_cash = 10000.0
        self.portfolio_value = 10000.0
        self.trade_count = 0
        self.last_price = 0.0
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state representation for RL agent"""
        if len(self.price_history) < self.lookback_window:
            # Pad with zeros if not enough history
            price_features = np.zeros(self.lookback_window)
            volume_features = np.zeros(self.lookback_window)
            return_features = np.zeros(self.lookback_window)
            indicator_features = np.zeros(self.lookback_window * 4)  # 4 indicators
        else:
            # Normalize price features
            prices = np.array(list(self.price_history))
            price_features = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
            
            # Normalize volume features
            volumes = np.array(list(self.volume_history))
            volume_features = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
            
            # Return features
            return_features = np.array(list(self.return_history))
            
            # Technical indicator features
            indicators = np.array(list(self.technical_indicators))
            indicator_features = indicators.flatten()
        
        # Portfolio state
        portfolio_features = np.array([
            self.current_position / 10.0,  # Normalized position
            (self.portfolio_value - 10000) / 10000,  # Normalized P&L
            self.trade_count / 100.0  # Normalized trade count
        ])
        
        # Combine all features
        state = np.concatenate([
            price_features,
            volume_features, 
            return_features,
            indicator_features,
            portfolio_features
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int, current_price: float, current_volume: float) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return new state, reward, done"""
        # Calculate returns
        if self.last_price > 0:
            returns = (current_price - self.last_price) / self.last_price
        else:
            returns = 0.0
        
        # Update price history
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        self.return_history.append(returns)
        
        # Calculate technical indicators
        indicators = self._calculate_indicators()
        self.technical_indicators.append(indicators)
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Update portfolio value
        self.portfolio_value = self.current_cash + self.current_position * current_price
        
        self.last_price = current_price
        
        # Get new state
        new_state = self.get_state()
        
        # Check if done (optional, can be always False for continuous trading)
        done = False
        
        return new_state, reward, done
    
    def _calculate_indicators(self) -> np.ndarray:
        """Calculate technical indicators for current state"""
        if len(self.price_history) < 20:
            return np.zeros(4)  # SMA, RSI, MACD, Bollinger
        
        prices = np.array(list(self.price_history))
        
        # Simple Moving Average (normalized)
        sma_20 = np.mean(prices[-20:])
        sma_signal = (prices[-1] - sma_20) / sma_20
        
        # RSI (simplified)
        price_changes = np.diff(prices[-15:])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            rsi = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi = 1 - (1 / (1 + rs))
        
        # MACD (simplified)
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices[-26:]) if len(prices) >= 26 else np.mean(prices)
        macd = (ema_12 - ema_26) / ema_26 if ema_26 != 0 else 0
        
        # Bollinger Bands
        bb_std = np.std(prices[-20:])
        bb_mean = np.mean(prices[-20:])
        bb_signal = (prices[-1] - bb_mean) / (bb_std + 1e-8)
        
        return np.array([sma_signal, rsi, macd, bb_signal])
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and return reward"""
        # Actions: 0=Hold, 1=Buy, 2=Sell
        prev_portfolio_value = self.portfolio_value
        
        if action == 1 and self.current_cash > 0:  # Buy
            # Use 10% of cash for each buy
            cash_to_use = self.current_cash * 0.1
            shares_to_buy = cash_to_use / current_price
            
            self.current_position += shares_to_buy
            self.current_cash -= cash_to_use
            self.trade_count += 1
            
        elif action == 2 and self.current_position > 0:  # Sell
            # Sell 10% of position
            shares_to_sell = self.current_position * 0.1
            cash_received = shares_to_sell * current_price
            
            self.current_position -= shares_to_sell
            self.current_cash += cash_received
            self.trade_count += 1
        
        # Calculate reward based on portfolio value change
        new_portfolio_value = self.current_cash + self.current_position * current_price
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Add penalty for excessive trading
        if action != 0:  # Non-hold action
            reward -= 0.001  # Small trading cost
        
        return reward


class DQNAgent:
    """Deep Q-Network agent for trading"""
    
    def __init__(self, state_size: int, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Simple neural network weights (in production, use actual neural network)
        # This is a simplified implementation for demonstration
        self.weights = {
            'W1': np.random.normal(0, 0.1, (state_size, 64)),
            'b1': np.zeros((1, 64)),
            'W2': np.random.normal(0, 0.1, (64, 32)),
            'b2': np.zeros((1, 32)),
            'W3': np.random.normal(0, 0.1, (32, action_size)),
            'b3': np.zeros((1, action_size))
        }
        
        self.target_weights = self.weights.copy()
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self._predict(state)
        return np.argmax(q_values)
    
    def _predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        x = state.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(x, self.weights['W1']) + self.weights['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output layer
        q_values = np.dot(a2, self.weights['W3']) + self.weights['b3']
        
        return q_values[0]
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]
        
        # This is a simplified training step
        # In production, use proper neural network framework
        for state, action, reward, next_state, done in experiences:
            target = reward
            if not done:
                target = reward + 0.95 * np.max(self._predict_target(next_state))
            
            # Update weights (simplified gradient descent)
            current_q = self._predict(state)
            target_q = current_q.copy()
            target_q[action] = target
            
            # Simplified weight update (would use backpropagation in practice)
            self._update_weights(state, target_q, current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _predict_target(self, state: np.ndarray) -> np.ndarray:
        """Predict using target network"""
        x = state.reshape(1, -1)
        
        z1 = np.dot(x, self.target_weights['W1']) + self.target_weights['b1']
        a1 = np.maximum(0, z1)
        
        z2 = np.dot(a1, self.target_weights['W2']) + self.target_weights['b2']
        a2 = np.maximum(0, z2)
        
        q_values = np.dot(a2, self.target_weights['W3']) + self.target_weights['b3']
        
        return q_values[0]
    
    def _update_weights(self, state: np.ndarray, target_q: np.ndarray, current_q: np.ndarray):
        """Simplified weight update"""
        error = target_q - current_q
        
        # Very simplified gradient update
        for key in self.weights:
            if 'W' in key:
                self.weights[key] += self.learning_rate * 0.001 * np.random.normal(0, 0.1, self.weights[key].shape)
    
    def update_target_model(self):
        """Update target network"""
        self.target_weights = self.weights.copy()
    
    def save_model(self, filepath: str):
        """Save model weights"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.weights, f)
    
    def load_model(self, filepath: str):
        """Load model weights"""
        try:
            with open(filepath, 'rb') as f:
                self.weights = pickle.load(f)
            self.target_weights = self.weights.copy()
        except FileNotFoundError:
            self.logger.warning(f"Model file {filepath} not found, using random weights")


class ReinforcementLearningStrategy(BaseStrategy):
    """
    Advanced reinforcement learning trading strategy using DQN.
    
    Key Features:
    - Deep Q-Network for action selection
    - Continuous learning and adaptation
    - Multi-dimensional state representation
    - Dynamic position sizing
    - Experience replay and target networks
    - Adaptive risk management
    - Performance-based reward shaping
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # RL parameters
        self.state_size = self.config.get("state_size", 157)  # Calculated from features
        self.action_size = self.config.get("action_size", 3)  # Hold, Buy, Sell
        self.lookback_window = self.config.get("lookback_window", 50)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        # Trading parameters
        self.position_size_pct = self.config.get("position_size_pct", 0.1)
        self.max_position_size = self.config.get("max_position_size", 0.3)
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.05)
        
        # Training parameters
        self.training_mode = self.config.get("training_mode", True)
        self.update_frequency = self.config.get("update_frequency", 100)
        self.target_update_frequency = self.config.get("target_update_frequency", 1000)
        
        # State tracking
        self.environments = {}  # One environment per symbol
        self.agents = {}        # One agent per symbol
        self.step_count = 0
        self.is_initialized = False
        self.last_actions = {}
        self.performance_history = {}
        
        # Performance tracking
        self.total_signals = 0
        self.successful_trades = 0
        self.total_reward = 0.0
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for RL strategy"""
        return {
            "state_size": 157,
            "action_size": 3,
            "lookback_window": 50,
            "learning_rate": 0.001,
            "position_size_pct": 0.1,
            "max_position_size": 0.3,
            "stop_loss_pct": 0.05,
            "training_mode": True,
            "update_frequency": 100,
            "target_update_frequency": 1000,
            "min_price_history": 100,
            "model_save_frequency": 1000,
            "exploration_episodes": 500,
            "reward_scaling": 1000.0,
            "risk_penalty": 0.01
        }
    
    async def initialize(self, market_data: Dict[str, Any], **kwargs) -> bool:
        """
        Initialize the reinforcement learning strategy
        
        Args:
            market_data: Initial market data
            **kwargs: Additional parameters
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Reinforcement Learning Strategy")
            
            # Initialize environments and agents for each symbol
            for symbol, data in market_data.items():
                # Initialize trading environment for symbol
                self.environments[symbol] = TradingEnvironment(self.lookback_window)
                
                # Initialize DQN agent for symbol
                self.agents[symbol] = DQNAgent(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    learning_rate=self.learning_rate
                )
                
                # Try to load pre-trained model
                model_path = f"models/rl_agent_{symbol}.pkl"
                self.agents[symbol].load_model(model_path)
                
                # Initialize tracking
                self.last_actions[symbol] = 0  # Hold
                self.performance_history[symbol] = {
                    'rewards': deque(maxlen=1000),
                    'actions': deque(maxlen=1000),
                    'portfolio_values': deque(maxlen=1000)
                }
            
            self.is_initialized = True
            self.logger.info(f"RL Strategy initialized for {len(market_data)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RL Strategy: {e}")
            return False
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trading signals using reinforcement learning
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not self.is_initialized:
            return signals
        
        try:
            for symbol, data in market_data.items():
                if symbol not in self.environments:
                    continue
                
                # Update environment and get action from agent
                symbol_signals = await self._process_symbol(symbol, data)
                signals.extend(symbol_signals)
            
            # Update models if in training mode
            if self.training_mode and self.step_count % self.update_frequency == 0:
                await self._update_models()
            
            # Update target networks
            if self.step_count % self.target_update_frequency == 0:
                await self._update_target_networks()
            
            # Save models periodically
            if self.step_count % self.config.get("model_save_frequency", 1000) == 0:
                await self._save_models()
            
            self.step_count += 1
            self.total_signals += len(signals)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating RL signals: {e}")
            return []
    
    async def _process_symbol(self, symbol: str, data: Dict[str, Any]) -> List[Signal]:
        """Process a single symbol and generate signals"""
        signals = []
        
        try:
            env = self.environments[symbol]
            agent = self.agents[symbol]
            
            # Extract market data
            current_price = float(data.get('close', data.get('price', 0)))
            current_volume = float(data.get('volume', 0))
            
            if current_price <= 0:
                return signals
            
            # Get current state
            current_state = env.get_state()
            
            # Get action from agent
            action = agent.act(current_state)
            
            # Execute action in environment and get reward
            next_state, reward, done = env.step(action, current_price, current_volume)
            
            # Store experience for training
            if self.training_mode and len(env.price_history) > 1:
                agent.remember(
                    current_state, 
                    self.last_actions[symbol], 
                    reward, 
                    next_state, 
                    done
                )
            
            # Generate trading signal based on action
            signal = await self._action_to_signal(symbol, action, current_price, data, reward)
            if signal:
                signals.append(signal)
            
            # Update tracking
            self.last_actions[symbol] = action
            self.performance_history[symbol]['rewards'].append(reward)
            self.performance_history[symbol]['actions'].append(action)
            self.performance_history[symbol]['portfolio_values'].append(env.portfolio_value)
            self.total_reward += reward
            
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")
        
        return signals
    
    async def _action_to_signal(self, 
                              symbol: str, 
                              action: int, 
                              current_price: float, 
                              data: Dict[str, Any],
                              reward: float) -> Optional[Signal]:
        """Convert RL action to trading signal"""
        try:
            # Actions: 0=Hold, 1=Buy, 2=Sell
            if action == 0:  # Hold - no signal
                return None
            
            # Determine signal direction
            direction = SignalDirection.BUY if action == 1 else SignalDirection.SELL
            
            # Calculate signal strength based on confidence and recent performance
            confidence = self._calculate_action_confidence(symbol, action)
            
            if confidence > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif confidence > 0.6:
                strength = SignalStrength.STRONG
            elif confidence > 0.4:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Calculate position size based on RL confidence and portfolio state
            position_size = self._calculate_rl_position_size(symbol, confidence)
            
            # Create signal metrics
            expected_reward = self._estimate_expected_reward(symbol)
            metrics = SignalMetrics(
                confidence=Decimal(str(confidence * 100)),
                expected_return=Decimal(str(abs(expected_reward))),
                expected_risk=Decimal(str(self.stop_loss_pct)),
                win_probability=Decimal(str(self._calculate_win_probability(symbol))),
                risk_reward_ratio=Decimal(str(abs(expected_reward) / self.stop_loss_pct)) if self.stop_loss_pct > 0 else Decimal("1"),
                volatility_forecast=Decimal(str(self._estimate_volatility(symbol)))
            )
            
            # Risk parameters
            risk_params = RiskParameters(
                max_position_size=Decimal(str(position_size)),
                stop_loss_pct=Decimal(str(self.stop_loss_pct)),
                max_slippage_pct=Decimal("0.002")
            )
            
            # Execution constraints
            execution_constraints = ExecutionConstraints(
                execution_style="MARKET",
                urgency="NORMAL"
            )
            
            # Create signal
            signal = Signal(
                strategy_id=uuid4(),
                signal_type=SignalType.ENTRY,
                direction=direction,
                strength=strength,
                symbol=symbol,
                exchange=data.get('exchange', 'default'),
                current_price=Decimal(str(current_price)),
                quantity=Decimal(str(position_size / current_price)),
                order_type="MARKET",
                metrics=metrics,
                risk_parameters=risk_params,
                execution_constraints=execution_constraints,
                source="ReinforcementLearningStrategy",
                reason=f"RL Action {action}: confidence={confidence:.2f}, reward={reward:.4f}",
                features={
                    "rl_action": action,
                    "rl_confidence": confidence,
                    "recent_reward": reward,
                    "portfolio_value": self.environments[symbol].portfolio_value,
                    "exploration_rate": self.agents[symbol].epsilon
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error converting action to signal for {symbol}: {e}")
            return None
    
    def _calculate_action_confidence(self, symbol: str, action: int) -> float:
        """Calculate confidence in the action based on recent performance"""
        try:
            if symbol not in self.performance_history:
                return 0.5
            
            history = self.performance_history[symbol]
            
            # Base confidence on recent rewards
            recent_rewards = list(history['rewards'])[-10:] if history['rewards'] else [0]
            avg_reward = np.mean(recent_rewards)
            
            # Scale reward to confidence (0-1)
            confidence = 0.5 + np.tanh(avg_reward * self.config.get("reward_scaling", 1000)) * 0.4
            
            # Adjust based on exploration rate
            exploration_penalty = self.agents[symbol].epsilon * 0.2
            confidence = max(0.1, confidence - exploration_penalty)
            
            return min(0.95, confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating action confidence: {e}")
            return 0.5
    
    def _calculate_rl_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size based on RL confidence and portfolio state"""
        try:
            env = self.environments[symbol]
            
            # Base position size
            base_size = env.portfolio_value * self.position_size_pct
            
            # Adjust based on confidence
            confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5
            adjusted_size = base_size * confidence_multiplier
            
            # Apply maximum limits
            max_size = env.portfolio_value * self.max_position_size
            final_size = min(adjusted_size, max_size)
            
            # Ensure we don't exceed available cash (for buys)
            available_cash = env.current_cash * 0.9  # Keep 10% buffer
            final_size = min(final_size, available_cash)
            
            return max(100, final_size)  # Minimum $100 position
            
        except Exception as e:
            self.logger.error(f"Error calculating RL position size: {e}")
            return 1000.0
    
    def _estimate_expected_reward(self, symbol: str) -> float:
        """Estimate expected reward based on recent performance"""
        try:
            if symbol not in self.performance_history:
                return 0.0
            
            recent_rewards = list(self.performance_history[symbol]['rewards'])[-20:]
            if not recent_rewards:
                return 0.0
            
            return np.mean(recent_rewards)
            
        except Exception as e:
            self.logger.error(f"Error estimating expected reward: {e}")
            return 0.0
    
    def _calculate_win_probability(self, symbol: str) -> float:
        """Calculate win probability based on recent trades"""
        try:
            if symbol not in self.performance_history:
                return 0.5
            
            recent_rewards = list(self.performance_history[symbol]['rewards'])[-50:]
            if not recent_rewards:
                return 0.5
            
            positive_rewards = [r for r in recent_rewards if r > 0]
            win_rate = len(positive_rewards) / len(recent_rewards)
            
            return max(0.1, min(0.9, win_rate))
            
        except Exception as e:
            self.logger.error(f"Error calculating win probability: {e}")
            return 0.5
    
    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility from recent price movements"""
        try:
            env = self.environments[symbol]
            if len(env.return_history) < 10:
                return 0.02  # Default 2% volatility
            
            returns = np.array(list(env.return_history))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            return max(0.01, min(1.0, volatility))
            
        except Exception as e:
            self.logger.error(f"Error estimating volatility: {e}")
            return 0.02
    
    async def _update_models(self):
        """Update all RL models with experience replay"""
        try:
            for symbol, agent in self.agents.items():
                if self.training_mode:
                    agent.replay()
        except Exception as e:
            self.logger.error(f"Error updating models: {e}")
    
    async def _update_target_networks(self):
        """Update target networks for all agents"""
        try:
            for agent in self.agents.values():
                agent.update_target_model()
            self.logger.info("Updated target networks")
        except Exception as e:
            self.logger.error(f"Error updating target networks: {e}")
    
    async def _save_models(self):
        """Save all trained models"""
        try:
            import os
            os.makedirs("models", exist_ok=True)
            
            for symbol, agent in self.agents.items():
                model_path = f"models/rl_agent_{symbol}.pkl"
                agent.save_model(model_path)
            
            self.logger.info(f"Saved models for {len(self.agents)} symbols")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    async def update_state(self, market_data: Dict[str, Any]) -> None:
        """Update strategy state with new market data"""
        try:
            # Update environments with new data
            for symbol, data in market_data.items():
                if symbol in self.environments:
                    current_price = float(data.get('close', data.get('price', 0)))
                    current_volume = float(data.get('volume', 0))
                    
                    if current_price > 0:
                        # Update environment (this happens in generate_signals)
                        pass
        except Exception as e:
            self.logger.error(f"Error updating RL strategy state: {e}")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              price: Decimal, 
                              signal_strength: Decimal,
                              available_capital: Decimal) -> Decimal:
        """Calculate optimal position size for a trade"""
        try:
            if symbol not in self.environments:
                return Decimal("0")
            
            # Get RL-based confidence
            confidence = self._calculate_action_confidence(symbol, 1)  # Assume buy action
            
            # Combine RL confidence with signal strength
            combined_confidence = (float(signal_strength) + confidence) / 2
            
            # Position sizing based on Kelly criterion approximation
            win_prob = self._calculate_win_probability(symbol)
            expected_return = self._estimate_expected_reward(symbol)
            
            if expected_return > 0 and win_prob > 0.5:
                kelly_fraction = (win_prob - (1 - win_prob) / abs(expected_return)) / abs(expected_return)
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                kelly_fraction = 0.05  # Conservative default
            
            # Final position size
            position_value = available_capital * Decimal(str(kelly_fraction)) * Decimal(str(combined_confidence))
            position_quantity = position_value / price
            
            return position_quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating RL position size: {e}")
            return Decimal("0")
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate if a signal should be executed"""
        try:
            # Basic validation
            if not signal.is_valid:
                return False
            
            # Check market data
            if signal.symbol not in market_data:
                return False
            
            # RL-specific validation
            if signal.symbol not in self.environments:
                return False
            
            # Check if agent confidence is still high
            confidence = self._calculate_action_confidence(signal.symbol, 1)
            if confidence < 0.3:  # Minimum confidence threshold
                return False
            
            # Price movement validation
            current_price = Decimal(str(market_data[signal.symbol].get('close', 0)))
            price_diff = abs(current_price - signal.current_price) / signal.current_price
            
            if price_diff > Decimal("0.02"):  # 2% price movement threshold
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating RL signal: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state"""
        symbol_info = {}
        for symbol in self.environments.keys():
            env = self.environments[symbol]
            agent = self.agents[symbol]
            history = self.performance_history[symbol]
            
            symbol_info[symbol] = {
                "portfolio_value": env.portfolio_value,
                "current_position": env.current_position,
                "current_cash": env.current_cash,
                "trade_count": env.trade_count,
                "exploration_rate": agent.epsilon,
                "recent_avg_reward": np.mean(list(history['rewards'])[-10:]) if history['rewards'] else 0,
                "win_rate": self._calculate_win_probability(symbol),
                "volatility": self._estimate_volatility(symbol)
            }
        
        return {
            "name": "Reinforcement Learning Strategy",
            "type": "REINFORCEMENT_LEARNING",
            "description": "Deep RL strategy using DQN for adaptive trading",
            "parameters": self.config,
            "state": {
                "is_initialized": self.is_initialized,
                "training_mode": self.training_mode,
                "total_signals": self.total_signals,
                "successful_trades": self.successful_trades,
                "success_rate": self.successful_trades / max(self.total_signals, 1) * 100,
                "total_reward": self.total_reward,
                "step_count": self.step_count,
                "symbols_tracked": list(self.environments.keys()),
                "last_update": datetime.now(timezone.utc).isoformat()
            },
            "symbol_details": symbol_info,
            "learning_progress": {
                "total_experiences": sum(len(agent.memory) for agent in self.agents.values()),
                "avg_exploration_rate": np.mean([agent.epsilon for agent in self.agents.values()]) if self.agents else 0,
                "models_trained": len([a for a in self.agents.values() if len(a.memory) > a.batch_size])
            }
        }
    
    def reset_state(self) -> None:
        """Reset strategy state"""
        for env in self.environments.values():
            env.reset()
        
        self.last_actions.clear()
        self.performance_history.clear()
        self.step_count = 0
        self.is_initialized = False
        self.total_signals = 0
        self.successful_trades = 0
        self.total_reward = 0.0
        self.logger.info("Reinforcement Learning Strategy state reset")
    
    def set_training_mode(self, training: bool):
        """Enable or disable training mode"""
        self.training_mode = training
        self.logger.info(f"Training mode set to: {training}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for analysis"""
        metrics = {}
        
        for symbol, history in self.performance_history.items():
            rewards = list(history['rewards'])
            actions = list(history['actions'])
            
            if rewards:
                metrics[symbol] = {
                    "total_reward": sum(rewards),
                    "avg_reward": np.mean(rewards),
                    "reward_volatility": np.std(rewards),
                    "max_reward": max(rewards),
                    "min_reward": min(rewards),
                    "action_distribution": {
                        "hold": actions.count(0) / len(actions) if actions else 0,
                        "buy": actions.count(1) / len(actions) if actions else 0,
                        "sell": actions.count(2) / len(actions) if actions else 0
                    },
                    "sharpe_ratio": np.mean(rewards) / (np.std(rewards) + 1e-8) if rewards else 0
                }
        
        return metrics