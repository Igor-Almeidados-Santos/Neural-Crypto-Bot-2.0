# Remove a subscrição
        if 'trades' in self._subscriptions:
            self._subscriptions['trades'] = [
                sub for sub in self._subscriptions['trades'] 
                if sub['trading_pair'] != trading_pair
            ]
    
    async def unsubscribe_candles(self, trading_pair: str, timeframe: TimeFrame) -> None:
        """
        Cancela a subscrição para atualizações de velas.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
        """
        if not self._initialized or not self._public_ws:
            return
            
        # Verifica se o timeframe é suportado
        bybit_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not bybit_timeframe:
            logger.warning(f"Timeframe não suportado para cancelamento: {timeframe}")
            return
        
        # Remove o callback
        if trading_pair in self._candle_callbacks and timeframe in self._candle_callbacks[trading_pair]:
            del self._candle_callbacks[trading_pair][timeframe]
            
            if not self._candle_callbacks[trading_pair]:
                del self._candle_callbacks[trading_pair]
        
        # Obtém o símbolo e o intervalo para o par
        symbol = self._exchange.market_id(trading_pair)
        interval = f"{bybit_timeframe}"
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "op": "unsubscribe",
            "args": [f"kline.{interval}.{symbol}"]
        }
        
        # Envia o cancelamento
        await self._public_ws.send(json.dumps(subscription))
        
        # Remove a subscrição
        if 'candles' in self._subscriptions:
            self._subscriptions['candles'] = [
                sub for sub in self._subscriptions['candles'] 
                if not (sub['trading_pair'] == trading_pair and sub['timeframe'] == timeframe)
            ]
    
    async def unsubscribe_funding_rates(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        if not self._initialized or not self._public_ws:
            return
            
        # Remove o callback
        if trading_pair in self._funding_rate_callbacks:
            del self._funding_rate_callbacks[trading_pair]
        
        # Obtém o símbolo para o par
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "op": "unsubscribe",
            "args": [f"funding.{symbol}"]
        }
        
        # Envia o cancelamento
        await self._public_ws.send(json.dumps(subscription))
        
        # Remove a subscrição
        if 'funding' in self._subscriptions:
            self._subscriptions['funding'] = [
                sub for sub in self._subscriptions['funding'] 
                if sub['trading_pair'] != trading_pair
            ]
    
    async def unsubscribe_liquidations(self, trading_pair: Optional[str] = None) -> None:
        """
        Cancela a subscrição para eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação (opcional)
        """
        if not self._initialized or not self._public_ws:
            return
            
        if trading_pair:
            # Remove o callback para o par específico
            if trading_pair in self._liquidation_callbacks:
                del self._liquidation_callbacks[trading_pair]
            
            # Obtém o símbolo para o par
            symbol = self._exchange.market_id(trading_pair)
            
            # Prepara a mensagem de cancelamento
            subscription = {
                "op": "unsubscribe",
                "args": [f"liquidation.{symbol}"]
            }
            
            # Envia o cancelamento
            await self._public_ws.send(json.dumps(subscription))
        else:
            # Remove o callback para todos os pares
            if "*" in self._liquidation_callbacks:
                del self._liquidation_callbacks["*"]
            
            # Prepara a mensagem de cancelamento para todas as categorias
            subscription = {
                "op": "unsubscribe",
                "args": ["liquidation"]
            }
            
            # Envia o cancelamento
            await self._public_ws.send(json.dumps(subscription))
        
        # Remove a subscrição
        if 'liquidation' in self._subscriptions:
            if trading_pair:
                self._subscriptions['liquidation'] = [
                    sub for sub in self._subscriptions['liquidation'] 
                    if sub['trading_pair'] != trading_pair
                ]
            else:
                self._subscriptions['liquidation'] = [
                    sub for sub in self._subscriptions['liquidation'] 
                    if sub['trading_pair'] != "*"
                ]
    
    def validate_trading_pair(self, trading_pair: str) -> bool:
        """
        Valida se um par de negociação é suportado pela exchange.
        
        Args:
            trading_pair: Par de negociação a ser validado
            
        Returns:
            bool: True se o par é suportado, False caso contrário
        """
        return trading_pair in self._trading_pairs
    
    def get_supported_timeframes(self) -> Dict[str, TimeFrame]:
        """
        Obtém os timeframes suportados pela exchange.
        
        Returns:
            Dict[str, TimeFrame]: Dicionário de timeframes suportados,
            mapeando o código da exchange para o TimeFrame padronizado
        """
        return self.TIMEFRAME_MAP
    
    async def _initialize_public_websocket(self) -> None:
        """
        Inicializa o WebSocket público para a Bybit.
        """
        if self._public_ws:
            return
            
        try:
            # Cria um cliente WebSocket
            self._public_ws = WebSocketClient(self._ws_public_url)
            self._public_ws.on_message(self._handle_public_message)
            self._public_ws.on_connect(self._handle_public_connect)
            self._public_ws.on_close(self._handle_public_close)
            
            # Conecta ao WebSocket
            await self._public_ws.connect()
            
            # Inicia a tarefa de ping periódico
            self._ping_task = asyncio.create_task(self._ping_loop())
            
        except Exception as e:
            logger.error(f"Erro ao inicializar WebSocket público para Bybit: {str(e)}", exc_info=True)
            raise
    
    async def _initialize_private_websocket(self) -> None:
        """
        Inicializa o WebSocket privado para a Bybit.
        
        Requer API key e secret.
        """
        if not self._api_key or not self._api_secret:
            logger.warning("API key e secret são necessários para WebSocket privado")
            return
            
        if self._private_ws:
            return
            
        try:
            # Cria um cliente WebSocket
            self._private_ws = WebSocketClient(self._ws_private_url)
            self._private_ws.on_message(self._handle_private_message)
            self._private_ws.on_connect(self._handle_private_connect)
            self._private_ws.on_close(self._handle_private_close)
            
            # Conecta ao WebSocket
            await self._private_ws.connect()
            
        except Exception as e:
            logger.error(f"Erro ao inicializar WebSocket privado para Bybit: {str(e)}", exc_info=True)
            raise
    
    async def _ping_loop(self) -> None:
        """
        Envia pings periódicos para manter a conexão WebSocket ativa.
        """
        while self._public_ws and self._ws_connected:
            try:
                # Bybit requer um ping a cada 20 segundos
                await asyncio.sleep(20)
                
                if self._public_ws:
                    ping_message = {
                        "op": "ping"
                    }
                    await self._public_ws.send(json.dumps(ping_message))
                
                if self._private_ws:
                    ping_message = {
                        "op": "ping"
                    }
                    await self._private_ws.send(json.dumps(ping_message))
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Erro no loop de ping: {str(e)}", exc_info=True)
    
    async def _authenticate_websocket(self) -> None:
        """
        Autentica o WebSocket privado.
        """
        if not self._private_ws or not self._api_key or not self._api_secret:
            return
            
        try:
            # Gera os parâmetros para autenticação
            expires = int((time.time() + 10) * 1000)  # 10 segundos no futuro
            
            # Constrói a string para assinatura
            signature_payload = f"GET/realtime{expires}"
            
            # Gera a assinatura
            signature = hmac.new(
                self._api_secret.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Envia a mensagem de autenticação
            auth_message = {
                "op": "auth",
                "args": [self._api_key, expires, signature]
            }
            
            await self._private_ws.send(json.dumps(auth_message))
            
        except Exception as e:
            logger.error(f"Erro ao autenticar WebSocket privado: {str(e)}", exc_info=True)
            raise
    
    async def _handle_public_connect(self) -> None:
        """
        Callback chamado quando o WebSocket público se conecta.
        """
        logger.info("WebSocket público da Bybit conectado")
        self._ws_connected = True
    
    async def _handle_public_close(self, code: int, reason: str) -> None:
        """
        Callback chamado quando o WebSocket público se desconecta.
        """
        logger.info(f"WebSocket público da Bybit desconectado: {code}, {reason}")
        self._ws_connected = False
        
        # Tenta reconectar após um tempo
        if self._initialized:
            await asyncio.sleep(5)
            try:
                logger.info("Tentando reconectar o WebSocket público")
                await self._initialize_public_websocket()
            except Exception as e:
                logger.error(f"Erro ao reconectar WebSocket público: {str(e)}", exc_info=True)
    
    async def _handle_private_connect(self) -> None:
        """
        Callback chamado quando o WebSocket privado se conecta.
        """
        logger.info("WebSocket privado da Bybit conectado")
        
        # Autentica a conexão
        await self._authenticate_websocket()
    
    async def _handle_private_close(self, code: int, reason: str) -> None:
        """
        Callback chamado quando o WebSocket privado se desconecta.
        """
        logger.info(f"WebSocket privado da Bybit desconectado: {code}, {reason}")
        
        # Tenta reconectar após um tempo
        if self._initialized and self._api_key and self._api_secret:
            await asyncio.sleep(5)
            try:
                logger.info("Tentando reconectar o WebSocket privado")
                await self._initialize_private_websocket()
            except Exception as e:
                logger.error(f"Erro ao reconectar WebSocket privado: {str(e)}", exc_info=True)
    
    async def _handle_public_message(self, message: str) -> None:
        """
        Processa mensagens recebidas via WebSocket público.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = json.loads(message)
            
            # Processa diferentes tipos de mensagens
            if "topic" in data:
                # Mensagem de dados
                topic = data["topic"]
                
                if topic.startswith("orderbook"):
                    # Atualização de orderbook
                    # Formato: "orderbook.{depth}.{symbol}"
                    parts = topic.split(".")
                    if len(parts) >= 3:
                        symbol = parts[2]
                        await self._process_orderbook_update(data, symbol)
                
                elif topic.startswith("publicTrade"):
                    # Atualização de trades
                    # Formato: "publicTrade.{symbol}"
                    parts = topic.split(".")
                    if len(parts) >= 2:
                        symbol = parts[1]
                        await self._process_trades_update(data, symbol)
                
                elif topic.startswith("kline"):
                    # Atualização de candles
                    # Formato: "kline.{interval}.{symbol}"
                    parts = topic.split(".")
                    if len(parts) >= 3:
                        interval = parts[1]
                        symbol = parts[2]
                        await self._process_candle_update(data, interval, symbol)
                
                elif topic.startswith("funding"):
                    # Atualização de taxa de financiamento
                    # Formato: "funding.{symbol}"
                    parts = topic.split(".")
                    if len(parts) >= 2:
                        symbol = parts[1]
                        await self._process_funding_rate_update(data, symbol)
                
                elif topic.startswith("liquidation"):
                    # Evento de liquidação
                    # Formato: "liquidation.{symbol}" ou "liquidation"
                    parts = topic.split(".")
                    symbol = parts[1] if len(parts) >= 2 else None
                    await self._process_liquidation_update(data, symbol)
            
            elif "op" in data:
                # Mensagem de controle
                op = data["op"]
                
                if op == "pong":
                    # Resposta a ping
                    logger.debug("Pong recebido do WebSocket público")
                
                elif op == "subscribe":
                    # Confirmação de subscrição
                    success = data.get("success", False)
                    if success:
                        logger.info(f"Subscrição bem-sucedida: {data.get('args')}")
                    else:
                        logger.error(f"Erro na subscrição: {data.get('ret_msg')}")
                
                elif op == "unsubscribe":
                    # Confirmação de cancelamento de subscrição
                    success = data.get("success", False)
                    if success:
                        logger.info(f"Cancelamento de subscrição bem-sucedido: {data.get('args')}")
                    else:
                        logger.error(f"Erro no cancelamento de subscrição: {data.get('ret_msg')}")
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem pública: {str(e)}", exc_info=True)
    
    async def _handle_private_message(self, message: str) -> None:
        """
        Processa mensagens recebidas via WebSocket privado.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = json.loads(message)
            
            # Processa diferentes tipos de mensagens
            if "topic" in data:
                # Mensagem de dados privados
                # Não implementado neste adaptador, mas poderia processar ordens, posições, etc.
                pass
                
            elif "op" in data:
                # Mensagem de controle
                op = data["op"]
                
                if op == "pong":
                    # Resposta a ping
                    logger.debug("Pong recebido do WebSocket privado")
                
                elif op == "auth":
                    # Confirmação de autenticação
                    success = data.get("success", False)
                    if success:
                        logger.info("Autenticação bem-sucedida no WebSocket privado")
                    else:
                        logger.error(f"Erro na autenticação do WebSocket privado: {data.get('ret_msg')}")
                
                elif op == "subscribe":
                    # Confirmação de subscrição
                    success = data.get("success", False)
                    if success:
                        logger.info(f"Subscrição privada bem-sucedida: {data.get('args')}")
                    else:
                        logger.error(f"Erro na subscrição privada: {data.get('ret_msg')}")
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem privada: {str(e)}", exc_info=True)
    
    async def _initialize_orderbook(self, trading_pair: str) -> None:
        """
        Inicializa o orderbook local com snapshot da Bybit.
        
        Args:
            trading_pair: Par de negociação
        """
        try:
            # Obtém uma snapshot do orderbook via REST API
            orderbook_data = await self._exchange.fetch_order_book(trading_pair, limit=25)
            
            # Inicializa o orderbook local
            self._local_orderbooks[trading_pair] = {
                "bids": {Decimal(str(price)): Decimal(str(amount)) for price, amount in orderbook_data['bids']},
                "asks": {Decimal(str(price)): Decimal(str(amount)) for price, amount in orderbook_data['asks']},
                "timestamp": datetime.fromtimestamp(orderbook_data['timestamp'] / 1000) if orderbook_data['timestamp'] else datetime.utcnow(),
                "u": 0  # Update ID
            }
            
            logger.debug(f"Orderbook inicializado para {trading_pair} com {len(orderbook_data['bids'])} bids e {len(orderbook_data['asks'])} asks")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar orderbook para {trading_pair}: {str(e)}", exc_info=True)
            raise
    
    async def _process_orderbook_update(self, data: Dict[str, Any], symbol: str) -> None:
        """
        Processa uma atualização de orderbook recebida via WebSocket.
        
        Args:
            data: Dados da atualização
            symbol: Símbolo do par de negociação no formato da Bybit
        """
        try:
            # Converte o símbolo para o formato padronizado
            trading_pair = None
            for pair, market in self._markets.items():
                if market.get('id') == symbol:
                    trading_pair = pair
                    break
            
            # Se não encontrar o símbolo, tenta usá-lo como está
            if not trading_pair and symbol:
                logger.warning(f"Símbolo não mapeado: {symbol}, usando como está")
                trading_pair = symbol
            
            # Extrai os dados da liquidação
            price = Decimal(str(liquidation_data.get("price", "0")))
            amount = Decimal(str(liquidation_data.get("qty", "0")))
            side_str = liquidation_data.get("side", "").lower()
            timestamp_ms = liquidation_data.get("updatedTime", time.time() * 1000)
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            
            # Converte o lado para o formato padronizado
            side = LiquidationSide.BUY if side_str == "buy" else LiquidationSide.SELL
            
            # Cria a entidade Liquidation
            liquidation = Liquidation(
                exchange=self.name,
                trading_pair=trading_pair,
                timestamp=timestamp,
                price=price,
                amount=amount,
                side=side,
                raw_data=liquidation_data
            )
            
            # Determina qual callback chamar
            if trading_pair and trading_pair in self._liquidation_callbacks:
                # Callback para o par específico
                callback = self._liquidation_callbacks[trading_pair]
                asyncio.create_task(callback(liquidation))
            elif "*" in self._liquidation_callbacks:
                # Callback para todos os pares
                callback = self._liquidation_callbacks["*"]
                asyncio.create_task(callback(liquidation))
            
        except Exception as e:
            logger.error(f"Erro ao processar evento de liquidação: {str(e)}", exc_info=True)
    
    def _parse_funding_rate(self, data: Dict[str, Any], trading_pair: str) -> FundingRate:
        """
        Converte dados brutos de taxa de financiamento para a entidade FundingRate.
        
        Args:
            data: Dados brutos da taxa de financiamento
            trading_pair: Par de negociação
            
        Returns:
            FundingRate: Entidade FundingRate
        """
        # Extrai os dados
        rate = Decimal(str(data.get('fundingRate', 0)))
        timestamp_str = data.get('fundingRateTimestamp', '')
        next_timestamp_str = data.get('nextFundingTime', '')
        
        # Converte as strings de timestamp para datetime
        if timestamp_str:
            try:
                timestamp = datetime.fromtimestamp(int(timestamp_str) / 1000)
            except:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()
        
        # Converte o próximo timestamp
        if next_timestamp_str:
            try:
                next_timestamp = datetime.fromtimestamp(int(next_timestamp_str) / 1000)
            except:
                # Se não conseguir converter, estima 8 horas à frente
                next_timestamp = timestamp + timedelta(hours=8)
        else:
            # Bybit normalmente tem intervalos de 8 horas
            next_timestamp = timestamp + timedelta(hours=8)
        
        return FundingRate(
            exchange=self.name,
            trading_pair=trading_pair,
            timestamp=timestamp,
            rate=rate,
            next_timestamp=next_timestamp,
            raw_data=data
        )
    
    def _parse_liquidation(self, data: Dict[str, Any], trading_pair: str) -> Liquidation:
        """
        Converte dados brutos de liquidação para a entidade Liquidation.
        
        Args:
            data: Dados brutos da liquidação
            trading_pair: Par de negociação
            
        Returns:
            Liquidation: Entidade Liquidation
        """
        # Extrai os dados
        price = Decimal(str(data.get('price', 0)))
        amount = Decimal(str(data.get('size', 0)))
        side_str = data.get('side', '').lower()
        timestamp_str = data.get('time', '')
        
        # Converte a string de timestamp para datetime
        if timestamp_str:
            try:
                timestamp = datetime.fromtimestamp(int(timestamp_str) / 1000)
            except:
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()
        
        # Converte o lado para o formato padronizado
        side = LiquidationSide.BUY if side_str == "buy" else LiquidationSide.SELL
        
        return Liquidation(
            exchange=self.name,
            trading_pair=trading_pair,
            timestamp=timestamp,
            price=price,
            amount=amount,
            side=side,
            raw_data=data
        )
                    trading_pair = pair
                    break
            
            if not trading_pair:
                logger.warning(f"Símbolo não encontrado: {symbol}")
                return
                
            # Verifica se temos um orderbook local para este par
            if trading_pair not in self._local_orderbooks:
                await self._initialize_orderbook(trading_pair)
            
            local_book = self._local_orderbooks[trading_pair]
            
            # Atualiza o orderbook com os dados recebidos
            update_data = data.get("data", {})
            timestamp = datetime.fromtimestamp(update_data.get("ts", time.time() * 1000) / 1000)
            local_book['timestamp'] = timestamp
            
            # Verifica se é um snapshot ou um delta
            if "u" in update_data:
                # Atualização delta
                update_id = update_data.get("u")
                
                # Verifica se a atualização é mais recente que o orderbook local
                if update_id <= local_book.get("u", 0):
                    return
                
                # Atualiza o ID da última atualização
                local_book["u"] = update_id
                
                # Atualiza as ofertas de compra (bids)
                if "b" in update_data:
                    for bid in update_data["b"]:
                        price = Decimal(str(bid[0]))
                        amount = Decimal(str(bid[1]))
                        
                        if amount == 0:
                            local_book["bids"].pop(price, None)
                        else:
                            local_book["bids"][price] = amount
                
                # Atualiza as ofertas de venda (asks)
                if "a" in update_data:
                    for ask in update_data["a"]:
                        price = Decimal(str(ask[0]))
                        amount = Decimal(str(ask[1]))
                        
                        if amount == 0:
                            local_book["asks"].pop(price, None)
                        else:
                            local_book["asks"][price] = amount
            
            else:
                # Snapshot completo
                if "bids" in update_data:
                    local_book["bids"] = {
                        Decimal(str(bid[0])): Decimal(str(bid[1]))
                        for bid in update_data["bids"]
                    }
                
                if "asks" in update_data:
                    local_book["asks"] = {
                        Decimal(str(ask[0])): Decimal(str(ask[1]))
                        for ask in update_data["asks"]
                    }
            
            # Converte o orderbook local para a entidade OrderBook
            bids = [
                OrderBookLevel(price=price, amount=amount)
                for price, amount in sorted(local_book["bids"].items(), reverse=True)
            ]
            
            asks = [
                OrderBookLevel(price=price, amount=amount)
                for price, amount in sorted(local_book["asks"].items())
            ]
            
            # Cria a entidade OrderBook
            orderbook = OrderBook(
                exchange=self.name,
                trading_pair=trading_pair,
                timestamp=local_book["timestamp"],
                bids=bids,
                asks=asks,
                raw_data=data
            )
            
            # Chama o callback registrado para este par
            if trading_pair in self._orderbook_callbacks:
                callback = self._orderbook_callbacks[trading_pair]
                asyncio.create_task(callback(orderbook))
            
        except Exception as e:
            logger.error(f"Erro ao processar atualização de orderbook: {str(e)}", exc_info=True)
    
    async def _process_trades_update(self, data: Dict[str, Any], symbol: str) -> None:
        """
        Processa uma atualização de trades recebida via WebSocket.
        
        Args:
            data: Dados da atualização
            symbol: Símbolo do par de negociação no formato da Bybit
        """
        try:
            # Converte o símbolo para o formato padronizado
            trading_pair = None
            for pair, market in self._markets.items():
                if market.get('id') == symbol:
                    trading_pair = pair
                    break
            
            if not trading_pair:
                logger.warning(f"Símbolo não encontrado: {symbol}")
                return
                
            # Processa cada trade na atualização
            trades_data = data.get("data", [])
            if not isinstance(trades_data, list):
                trades_data = [trades_data]
                
            for trade_data in trades_data:
                # Extrai os dados do trade
                trade_id = str(trade_data.get("i", ""))
                price = Decimal(str(trade_data.get("p", "0")))
                amount = Decimal(str(trade_data.get("v", "0")))
                side_str = trade_data.get("S", "").lower()
                timestamp_ms = trade_data.get("T", time.time() * 1000)
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                
                # Converte o lado para o formato padronizado
                side = TradeSide.BUY if side_str == "buy" else TradeSide.SELL
                
                # Calcula o custo
                cost = price * amount
                
                # Cria a entidade Trade
                trade = Trade(
                    id=trade_id,
                    exchange=self.name,
                    trading_pair=trading_pair,
                    price=price,
                    amount=amount,
                    cost=cost,
                    timestamp=timestamp,
                    side=side,
                    taker=True,  # Bybit WebSocket só envia trades de takers
                    raw_data=trade_data
                )
                
                # Chama o callback registrado para este par
                if trading_pair in self._trade_callbacks:
                    callback = self._trade_callbacks[trading_pair]
                    asyncio.create_task(callback(trade))
            
        except Exception as e:
            logger.error(f"Erro ao processar atualização de trades: {str(e)}", exc_info=True)
    
    async def _process_candle_update(self, data: Dict[str, Any], interval: str, symbol: str) -> None:
        """
        Processa uma atualização de candle recebida via WebSocket.
        
        Args:
            data: Dados da atualização
            interval: Intervalo da candle no formato da Bybit
            symbol: Símbolo do par de negociação no formato da Bybit
        """
        try:
            # Converte o símbolo para o formato padronizado
            trading_pair = None
            for pair, market in self._markets.items():
                if market.get('id') == symbol:
                    trading_pair = pair
                    break
            
            if not trading_pair:
                logger.warning(f"Símbolo não encontrado: {symbol}")
                return
                
            # Converte o intervalo para o timeframe padronizado
            if interval in self.TIMEFRAME_MAP:
                timeframe = self.TIMEFRAME_MAP[interval]
            else:
                logger.warning(f"Intervalo de candle não reconhecido: {interval}")
                return
                
            # Processa cada candle na atualização
            candles_data = data.get("data", [])
            if not isinstance(candles_data, list):
                candles_data = [candles_data]
                
            for candle_data in candles_data:
                # Extrai os dados da candle
                start_time = candle_data.get("start", time.time() * 1000)
                open_price = Decimal(str(candle_data.get("open", "0")))
                high_price = Decimal(str(candle_data.get("high", "0")))
                low_price = Decimal(str(candle_data.get("low", "0")))
                close_price = Decimal(str(candle_data.get("close", "0")))
                volume = Decimal(str(candle_data.get("volume", "0")))
                timestamp = datetime.fromtimestamp(start_time / 1000)
                
                # Cria a entidade Candle
                candle = Candle(
                    exchange=self.name,
                    trading_pair=trading_pair,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    raw_data=candle_data
                )
                
                # Chama o callback registrado para este par e timeframe
                if (trading_pair in self._candle_callbacks and 
                    timeframe in self._candle_callbacks[trading_pair]):
                    callback = self._candle_callbacks[trading_pair][timeframe]
                    asyncio.create_task(callback(candle))
            
        except Exception as e:
            logger.error(f"Erro ao processar atualização de candle: {str(e)}", exc_info=True)
    
    async def _process_funding_rate_update(self, data: Dict[str, Any], symbol: str) -> None:
        """
        Processa uma atualização de taxa de financiamento recebida via WebSocket.
        
        Args:
            data: Dados da atualização
            symbol: Símbolo do par de negociação no formato da Bybit
        """
        try:
            # Converte o símbolo para o formato padronizado
            trading_pair = None
            for pair, market in self._markets.items():
                if market.get('id') == symbol:
                    trading_pair = pair
                    break
            
            if not trading_pair:
                logger.warning(f"Símbolo não encontrado: {symbol}")
                return
                
            # Processa cada atualização de taxa de financiamento
            funding_data = data.get("data", {})
            
            # Extrai os dados da taxa de financiamento
            rate = Decimal(str(funding_data.get("fundingRate", "0")))
            timestamp_ms = funding_data.get("fundingRateTimestamp", time.time() * 1000)
            next_timestamp_ms = funding_data.get("nextFundingTime", 0)
            
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            next_timestamp = datetime.fromtimestamp(next_timestamp_ms / 1000) if next_timestamp_ms else None
            
            # Cria a entidade FundingRate
            funding_rate = FundingRate(
                exchange=self.name,
                trading_pair=trading_pair,
                timestamp=timestamp,
                rate=rate,
                next_timestamp=next_timestamp,
                raw_data=funding_data
            )
            
            # Chama o callback registrado para este par
            if trading_pair in self._funding_rate_callbacks:
                callback = self._funding_rate_callbacks[trading_pair]
                asyncio.create_task(callback(funding_rate))
            
        except Exception as e:
            logger.error(f"Erro ao processar atualização de taxa de financiamento: {str(e)}", exc_info=True)
    
    async def _process_liquidation_update(self, data: Dict[str, Any], symbol: Optional[str] = None) -> None:
        """
        Processa uma atualização de liquidação recebida via WebSocket.
        
        Args:
            data: Dados da atualização
            symbol: Símbolo do par de negociação no formato da Bybit (opcional)
        """
        try:
            # Processa cada evento de liquidação
            liquidation_data = data.get("data", {})
            
            # Se symbol não for fornecido, tenta extrair do evento
            if not symbol and "symbol" in liquidation_data:
                symbol = liquidation_data.get("symbol")
            
            # Converte o símbolo para o formato padronizado
            trading_pair = None
            for pair, market in self._markets.items():
                if market.get('id') == symbol:        self._liquidation_callbacks: Dict[str, Callable[[Liquidation], Awaitable[None]]] = {}
        
        # Cache de dados de mercado
        self._markets: Dict[str, Dict] = {}
        self._trading_pairs: Set[str] = set()
        self._local_orderbooks: Dict[str, Dict] = {}
        
        # Subscriptions ativas
        self._subscriptions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Mapeamento de categoria para pares de negociação
        self._category_map: Dict[str, str] = {}
        
        # Flag para ping periódico
        self._ping_task: Optional[asyncio.Task] = None
        
        # Flags para controle de estado
        self._initialized = False
        self._ws_connected = False
    
    @property
    def name(self) -> str:
        """
        Retorna o nome da exchange.
        
        Returns:
            str: Nome da exchange
        """
        return "bybit"
    
    @property
    def supported_trading_pairs(self) -> Set[str]:
        """
        Retorna o conjunto de pares de negociação suportados pela exchange.
        
        Returns:
            Set[str]: Conjunto de pares de negociação suportados
        """
        return self._trading_pairs
    
    async def initialize(self) -> None:
        """
        Inicializa o adaptador de exchange.
        
        Carrega dados de mercado e inicializa recursos necessários.
        """
        if self._initialized:
            return
            
        try:
            logger.info("Inicializando adaptador Bybit")
            
            # Carrega os mercados via ccxt
            self._markets = await self._exchange.load_markets(reload=True)
            
            # Extrai os pares de negociação padronizados
            self._trading_pairs = {symbol for symbol in self._markets.keys() if '/' in symbol}
            
            # Mapeia os pares para suas categorias (spot, linear, inverse)
            for symbol, market in self._markets.items():
                if '/' in symbol:
                    # Determina a categoria com base nos dados do mercado
                    if market.get('linear'):
                        self._category_map[symbol] = 'linear'
                    elif market.get('inverse'):
                        self._category_map[symbol] = 'inverse'
                    else:
                        self._category_map[symbol] = 'spot'
            
            # Inicializa o WebSocket público
            await self._initialize_public_websocket()
            
            # Inicializa o WebSocket privado se as credenciais foram fornecidas
            if self._api_key and self._api_secret:
                await self._initialize_private_websocket()
            
            logger.info(f"Adaptador Bybit inicializado com {len(self._trading_pairs)} pares de negociação")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar adaptador Bybit: {str(e)}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """
        Finaliza o adaptador de exchange.
        
        Libera recursos e fecha conexões.
        """
        if not self._initialized:
            return
            
        try:
            logger.info("Finalizando adaptador Bybit")
            
            # Cancela a tarefa de ping
            if self._ping_task and not self._ping_task.done():
                self._ping_task.cancel()
                try:
                    await self._ping_task
                except asyncio.CancelledError:
                    pass
            
            # Fecha o WebSocket público
            if self._public_ws:
                await self._public_ws.close()
                self._public_ws = None
            
            # Fecha o WebSocket privado
            if self._private_ws:
                await self._private_ws.close()
                self._private_ws = None
            
            # Fecha o cliente ccxt
            await self._exchange.close()
            
            # Limpa estruturas de dados
            self._orderbook_callbacks.clear()
            self._trade_callbacks.clear()
            self._candle_callbacks.clear()
            self._funding_rate_callbacks.clear()
            self._liquidation_callbacks.clear()
            self._local_orderbooks.clear()
            
            self._initialized = False
            self._ws_connected = False
            
            logger.info("Adaptador Bybit finalizado")
            
        except Exception as e:
            logger.error(f"Erro ao finalizar adaptador Bybit: {str(e)}", exc_info=True)
            raise
    
    async def fetch_trading_pairs(self) -> List[str]:
        """
        Obtém a lista de pares de negociação disponíveis na exchange.
        
        Returns:
            List[str]: Lista de pares de negociação no formato padronizado (ex: BTC/USDT)
        """
        if not self._initialized:
            await self.initialize()
        
        # Recarrega os mercados para garantir dados atualizados
        self._markets = await self._exchange.load_markets(reload=True)
        
        # Extrai os pares de negociação padronizados
        self._trading_pairs = {symbol for symbol in self._markets.keys() if '/' in symbol}
        
        # Atualiza o mapeamento de categorias
        for symbol, market in self._markets.items():
            if '/' in symbol:
                if market.get('linear'):
                    self._category_map[symbol] = 'linear'
                elif market.get('inverse'):
                    self._category_map[symbol] = 'inverse'
                else:
                    self._category_map[symbol] = 'spot'
        
        return sorted(list(self._trading_pairs))
    
    async def fetch_ticker(self, trading_pair: str) -> Dict[str, Any]:
        """
        Obtém informações de ticker para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            
        Returns:
            Dict[str, Any]: Informações de ticker no formato padronizado
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Obtém o ticker via ccxt
        ticker = await self._exchange.fetch_ticker(trading_pair)
        
        # Formata a resposta
        return {
            "exchange": self.name,
            "trading_pair": trading_pair,
            "timestamp": datetime.fromtimestamp(ticker['timestamp'] / 1000),
            "last": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "high": ticker['high'],
            "low": ticker['low'],
            "volume": ticker['volume'],
            "change": ticker['change'],
            "percentage": ticker['percentage'],
            "average": ticker['average'],
            "vwap": ticker.get('vwap'),
            "open": ticker.get('open'),
            "close": ticker.get('close'),
            "last_trade_id": ticker.get('info', {}).get('last_trade_id'),
            "trades": None  # Bybit não fornece o número de trades no ticker
        }
    
    async def fetch_orderbook(self, trading_pair: str, depth: int = 20) -> OrderBook:
        """
        Obtém o livro de ofertas para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            depth: Profundidade do orderbook a ser obtido
            
        Returns:
            OrderBook: Entidade OrderBook com os dados do livro de ofertas
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Ajusta a profundidade para os valores permitidos pela Bybit
        allowed_depths = [1, 25, 50, 100, 200, 500, 1000]
        adjusted_depth = min([d for d in allowed_depths if d >= depth], default=25)
        
        # Obtém o orderbook via ccxt
        orderbook_data = await self._exchange.fetch_order_book(trading_pair, limit=adjusted_depth)
        
        # Converte para o formato da entidade OrderBook
        bids = [
            OrderBookLevel(
                price=Decimal(str(price)),
                amount=Decimal(str(amount))
            ) for price, amount in orderbook_data['bids']
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(price)),
                amount=Decimal(str(amount))
            ) for price, amount in orderbook_data['asks']
        ]
        
        return OrderBook(
            exchange=self.name,
            trading_pair=trading_pair,
            timestamp=datetime.fromtimestamp(orderbook_data['timestamp'] / 1000) if orderbook_data['timestamp'] else datetime.utcnow(),
            bids=bids,
            asks=asks,
            raw_data=orderbook_data
        )
    
    async def fetch_trades(
        self, 
        trading_pair: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """
        Obtém transações recentes para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            since: Timestamp a partir do qual obter transações (opcional)
            limit: Número máximo de transações a retornar (opcional)
            
        Returns:
            List[Trade]: Lista de entidades Trade com os dados das transações
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Converte datetime para timestamp em milissegundos
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Obtém as transações via ccxt
        trades_data = await self._exchange.fetch_trades(
            trading_pair, 
            since=since_ts, 
            limit=limit
        )
        
        # Converte para o formato da entidade Trade
        trades = []
        for trade_data in trades_data:
            side = TradeSide.BUY if trade_data['side'] == 'buy' else TradeSide.SELL
            
            trade = Trade(
                id=str(trade_data['id']),
                exchange=self.name,
                trading_pair=trading_pair,
                price=Decimal(str(trade_data['price'])),
                amount=Decimal(str(trade_data['amount'])),
                cost=Decimal(str(trade_data['cost'])),
                timestamp=datetime.fromtimestamp(trade_data['timestamp'] / 1000),
                side=side,
                taker=trade_data.get('takerOrMaker', 'taker') == 'taker',
                raw_data=trade_data
            )
            
            trades.append(trade)
        
        return trades
    
    async def fetch_candles(
        self, 
        trading_pair: str, 
        timeframe: TimeFrame,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Obtém velas históricas para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
            since: Timestamp a partir do qual obter velas (opcional)
            limit: Número máximo de velas a retornar (opcional)
            
        Returns:
            List[Candle]: Lista de entidades Candle com os dados das velas
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Converte o timeframe para o formato da Bybit
        bybit_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not bybit_timeframe:
            raise ValueError(f"Timeframe não suportado pela Bybit: {timeframe}")
        
        # Converte datetime para timestamp em milissegundos
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Obtém as velas via ccxt
        candles_data = await self._exchange.fetch_ohlcv(
            trading_pair, 
            timeframe=bybit_timeframe, 
            since=since_ts, 
            limit=limit
        )
        
        # Converte para o formato da entidade Candle
        candles = []
        for candle_data in candles_data:
            timestamp = datetime.fromtimestamp(candle_data[0] / 1000)
            open_price = Decimal(str(candle_data[1]))
            high_price = Decimal(str(candle_data[2]))
            low_price = Decimal(str(candle_data[3]))
            close_price = Decimal(str(candle_data[4]))
            volume = Decimal(str(candle_data[5]))
            
            candle = Candle(
                exchange=self.name,
                trading_pair=trading_pair,
                timestamp=timestamp,
                timeframe=timeframe,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                raw_data=candle_data
            )
            
            candles.append(candle)
        
        return candles
    
    async def fetch_funding_rates(
        self,
        trading_pair: Optional[str] = None
    ) -> List[FundingRate]:
        """
        Obtém taxas de financiamento para pares de negociação perpétuos.
        
        Args:
            trading_pair: Par de negociação específico (opcional)
            
        Returns:
            List[FundingRate]: Lista de entidades FundingRate
        """
        if not self._initialized:
            await self.initialize()
            
        funding_rates = []
        
        try:
            # Bybit permite obter taxas de financiamento para todos os pares de uma vez
            if trading_pair:
                # Verifica se o par é suportado e é um contrato perpétuo
                if not self.validate_trading_pair(trading_pair):
                    raise ValueError(f"Par de negociação inválido: {trading_pair}")
                
                category = self._category_map.get(trading_pair)
                if category not in ['linear', 'inverse']:
                    logger.warning(f"Par {trading_pair} não é um contrato perpétuo, não possui taxa de financiamento")
                    return []
                
                # Obtém a taxa de financiamento para o par específico
                symbol = self._exchange.market_id(trading_pair)
                response = await self._exchange.publicGetV5MarketFundingHistory({
                    'category': category,
                    'symbol': symbol,
                    'limit': 1
                })
                
                # Processa a resposta
                if response.get('retCode') == 0 and response.get('result'):
                    result = response.get('result', {})
                    for item in result.get('list', []):
                        funding_rates.append(self._parse_funding_rate(item, trading_pair))
            else:
                # Obtém as taxas de financiamento para todos os pares
                categories = ['linear', 'inverse']
                
                for category in categories:
                    response = await self._exchange.publicGetV5MarketFundingHistory({
                        'category': category,
                        'limit': 200  # Máximo permitido pela API
                    })
                    
                    if response.get('retCode') == 0 and response.get('result'):
                        result = response.get('result', {})
                        for item in result.get('list', []):
                            symbol = item.get('symbol')
                            
                            # Converte o símbolo para o formato padronizado
                            std_symbol = None
                            for pair, market in self._markets.items():
                                if market.get('id') == symbol:
                                    std_symbol = pair
                                    break
                            
                            if std_symbol:
                                funding_rates.append(self._parse_funding_rate(item, std_symbol))
            
            return funding_rates
            
        except Exception as e:
            logger.error(f"Erro ao obter taxas de financiamento: {str(e)}", exc_info=True)
            return []
    
    async def fetch_liquidations(
        self,
        trading_pair: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Liquidation]:
        """
        Obtém eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação específico (opcional)
            since: Timestamp a partir do qual obter liquidações (opcional)
            limit: Número máximo de liquidações a retornar (opcional)
            
        Returns:
            List[Liquidation]: Lista de entidades Liquidation
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Bybit tem um endpoint público para liquidações
            liquidations = []
            
            # Define a categoria com base no par de negociação
            categories = []
            if trading_pair:
                category = self._category_map.get(trading_pair)
                if category in ['linear', 'inverse']:
                    categories = [category]
                else:
                    logger.warning(f"Par {trading_pair} não é um contrato, não possui liquidações")
                    return []
            else:
                categories = ['linear', 'inverse']
            
            # Parâmetros da requisição
            params = {}
            if trading_pair:
                params['symbol'] = self._exchange.market_id(trading_pair)
            if limit:
                params['limit'] = min(limit, 100)  # Bybit limita a 100
                
            # Obtém as liquidações para cada categoria
            for category in categories:
                params['category'] = category
                
                response = await self._exchange.publicGetV5MarketRecentLiqRecords(params)
                
                if response.get('retCode') == 0 and response.get('result'):
                    result = response.get('result', {})
                    for item in result.get('list', []):
                        symbol = item.get('symbol')
                        
                        # Converte o símbolo para o formato padronizado
                        std_symbol = None
                        for pair, market in self._markets.items():
                            if market.get('id') == symbol:
                                std_symbol = pair
                                break
                        
                        if std_symbol:
                            liquidations.append(self._parse_liquidation(item, std_symbol))
            
            return liquidations
            
        except Exception as e:
            logger.error(f"Erro ao obter liquidações: {str(e)}", exc_info=True)
            return []
    
    async def subscribe_orderbook(
        self, 
        trading_pair: str, 
        callback: Callable[[OrderBook], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de orderbook em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            callback: Função assíncrona a ser chamada com cada atualização de orderbook
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._orderbook_callbacks[trading_pair] = callback
        
        # Inicializa o orderbook local
        await self._initialize_orderbook(trading_pair)
        
        # Obtém a categoria e o símbolo para o par
        category = self._category_map.get(trading_pair, 'spot')
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de subscrição
        subscription = {
            "op": "subscribe",
            "args": [f"orderbook.25.{symbol}"]
        }
        
        # Envia a subscrição
        await self._public_ws.send(json.dumps(subscription))
        
        # Armazena a subscrição
        if 'orderbook' not in self._subscriptions:
            self._subscriptions['orderbook'] = []
            
        self._subscriptions['orderbook'].append({
            'trading_pair': trading_pair,
            'symbol': symbol
        })
    
    async def subscribe_trades(
        self, 
        trading_pair: str, 
        callback: Callable[[Trade], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de trades em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            callback: Função assíncrona a ser chamada com cada nova transação
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._trade_callbacks[trading_pair] = callback
        
        # Obtém a categoria e o símbolo para o par
        category = self._category_map.get(trading_pair, 'spot')
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de subscrição
        subscription = {
            "op": "subscribe",
            "args": [f"publicTrade.{symbol}"]
        }
        
        # Envia a subscrição
        await self._public_ws.send(json.dumps(subscription))
        
        # Armazena a subscrição
        if 'trades' not in self._subscriptions:
            self._subscriptions['trades'] = []
            
        self._subscriptions['trades'].append({
            'trading_pair': trading_pair,
            'symbol': symbol
        })
    
    async def subscribe_candles(
        self, 
        trading_pair: str, 
        timeframe: TimeFrame,
        callback: Callable[[Candle], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de velas em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
            callback: Função assíncrona a ser chamada com cada nova vela
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Verifica se o timeframe é suportado
        bybit_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not bybit_timeframe:
            raise ValueError(f"Timeframe não suportado pela Bybit: {timeframe}")
        
        # Armazena o callback
        if trading_pair not in self._candle_callbacks:
            self._candle_callbacks[trading_pair] = {}
            
        self._candle_callbacks[trading_pair][timeframe] = callback
        
        # Obtém a categoria e o símbolo para o par
        category = self._category_map.get(trading_pair, 'spot')
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de subscrição
        # Na Bybit, o formato é "kline.{interval}.{symbol}"
        interval = f"{bybit_timeframe}"
        
        subscription = {
            "op": "subscribe",
            "args": [f"kline.{interval}.{symbol}"]
        }
        
        # Envia a subscrição
        await self._public_ws.send(json.dumps(subscription))
        
        # Armazena a subscrição
        if 'candles' not in self._subscriptions:
            self._subscriptions['candles'] = []
            
        self._subscriptions['candles'].append({
            'trading_pair': trading_pair,
            'symbol': symbol,
            'timeframe': timeframe,
            'interval': interval
        })
    
    async def subscribe_funding_rates(
        self,
        trading_pair: str,
        callback: Callable[[FundingRate], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            callback: Função assíncrona a ser chamada com cada atualização
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Verifica se é um contrato perpétuo
        category = self._category_map.get(trading_pair)
        if category not in ['linear', 'inverse']:
            raise ValueError(f"Par {trading_pair} não é um contrato perpétuo, não possui taxa de financiamento")
        
        # Armazena o callback
        self._funding_rate_callbacks[trading_pair] = callback
        
        # Obtém o símbolo para o par
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de subscrição
        subscription = {
            "op": "subscribe",
            "args": [f"funding.{symbol}"]
        }
        
        # Envia a subscrição
        await self._public_ws.send(json.dumps(subscription))
        
        # Armazena a subscrição
        if 'funding' not in self._subscriptions:
            self._subscriptions['funding'] = []
            
        self._subscriptions['funding'].append({
            'trading_pair': trading_pair,
            'symbol': symbol
        })
    
    async def subscribe_liquidations(
        self,
        trading_pair: Optional[str] = None,
        callback: Callable[[Liquidation], Awaitable[None]] = None
    ) -> None:
        """
        Subscreve para eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação (opcional)
            callback: Função assíncrona a ser chamada com cada liquidação
        """
        if not self._initialized:
            await self.initialize()
            
        # Bybit permite subscrever para todos os eventos de liquidação
        # ou para um símbolo específico
        
        if trading_pair:
            # Verifica se o par é suportado e é um contrato
            if not self.validate_trading_pair(trading_pair):
                raise ValueError(f"Par de negociação inválido: {trading_pair}")
            
            category = self._category_map.get(trading_pair)
            if category not in ['linear', 'inverse']:
                raise ValueError(f"Par {trading_pair} não é um contrato, não possui liquidações")
            
            # Armazena o callback
            self._liquidation_callbacks[trading_pair] = callback
            
            # Obtém o símbolo para o par
            symbol = self._exchange.market_id(trading_pair)
            
            # Prepara a mensagem de subscrição
            subscription = {
                "op": "subscribe",
                "args": [f"liquidation.{symbol}"]
            }
        else:
            # Subscreve para todos os eventos de liquidação
            # Bybit não suporta isso diretamente, então precisamos subscrever
            # para cada categoria separadamente
            
            # Armazena o callback para "todos os pares"
            self._liquidation_callbacks["*"] = callback
            
            # Prepara a mensagem de subscrição para todas as categorias
            subscription = {
                "op": "subscribe",
                "args": ["liquidation"]
            }
        
        # Envia a subscrição
        await self._public_ws.send(json.dumps(subscription))
        
        # Armazena a subscrição
        if 'liquidation' not in self._subscriptions:
            self._subscriptions['liquidation'] = []
            
        self._subscriptions['liquidation'].append({
            'trading_pair': trading_pair or "*"
        })
    
    async def unsubscribe_orderbook(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de orderbook.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        if not self._initialized or not self._public_ws:
            return
            
        # Remove o callback
        if trading_pair in self._orderbook_callbacks:
            del self._orderbook_callbacks[trading_pair]
        
        # Obtém o símbolo para o par
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "op": "unsubscribe",
            "args": [f"orderbook.25.{symbol}"]
        }
        
        # Envia o cancelamento
        await self._public_ws.send(json.dumps(subscription))
        
        # Remove o orderbook local
        if trading_pair in self._local_orderbooks:
            del self._local_orderbooks[trading_pair]
        
        # Remove a subscrição
        if 'orderbook' in self._subscriptions:
            self._subscriptions['orderbook'] = [
                sub for sub in self._subscriptions['orderbook'] 
                if sub['trading_pair'] != trading_pair
            ]
    
    async def unsubscribe_trades(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de trades.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        if not self._initialized or not self._public_ws:
            return
            
        # Remove o callback
        if trading_pair in self._trade_callbacks:
            del self._trade_callbacks[trading_pair]
        
        # Obtém o símbolo para o par
        symbol = self._exchange.market_id(trading_pair)
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "op": "unsubscribe",
            "args": [f"publicTrade.{symbol}"]
        }
        
        # Envia o cancelamento
        await self._public_ws.send(json.dumps(subscription))
        
        # Remove a subscrição
        if 'trades' in self._"""
Adaptador para a exchange Bybit.

Este módulo implementa o adaptador para a exchange Bybit, seguindo
a interface comum definida para todos os adaptadores de exchanges.
"""
import asyncio
import hmac
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable

import ccxt.async_support as ccxt
from websockets.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed
import orjson

from data_collection.adapters.exchange_adapter_interface import ExchangeAdapterInterface
from data_collection.domain.entities.candle import Candle, TimeFrame
from data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from data_collection.domain.entities.trade import Trade, TradeSide
from data_collection.domain.entities.funding_rate import FundingRate
from data_collection.domain.entities.liquidation import Liquidation, LiquidationSide
from data_collection.infrastructure.websocket_client import WebSocketClient


logger = logging.getLogger(__name__)


class BybitAdapter(ExchangeAdapterInterface):
    """
    Adaptador para a exchange Bybit.
    
    Implementa a interface ExchangeAdapterInterface para a exchange Bybit,
    utilizando a biblioteca ccxt para requisições REST e WebSockets nativos
    da Bybit para streaming de dados.
    """
    
    # Mapeamento de timeframes da Bybit para o formato padronizado
    TIMEFRAME_MAP = {
        "1": TimeFrame.MINUTE_1,
        "3": TimeFrame.MINUTE_3,
        "5": TimeFrame.MINUTE_5,
        "15": TimeFrame.MINUTE_15,
        "30": TimeFrame.MINUTE_30,
        "60": TimeFrame.HOUR_1,
        "120": TimeFrame.HOUR_2,
        "240": TimeFrame.HOUR_4,
        "360": TimeFrame.HOUR_6,
        "720": TimeFrame.HOUR_12,
        "D": TimeFrame.DAY_1,
        "W": TimeFrame.WEEK_1,
        "M": TimeFrame.MONTH_1
    }
    
    # Mapeamento reverso, do formato padronizado para o formato da Bybit
    REVERSE_TIMEFRAME_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Inicializa o adaptador para a Bybit.
        
        Args:
            api_key: Chave de API da Bybit (opcional)
            api_secret: Chave secreta da API da Bybit (opcional)
            testnet: Se True, utiliza o ambiente de testes da Bybit
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        
        # Inicializa o cliente ccxt para a Bybit
        self._exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'recvWindow': 5000
            }
        })
        
        # Configura para usar testnet se necessário
        if testnet:
            self._exchange.urls['api'] = {
                'public': 'https://api-testnet.bybit.com',
                'private': 'https://api-testnet.bybit.com'
            }
        
        # Define os URLs dos WebSockets
        if testnet:
            self._ws_public_url = "wss://stream-testnet.bybit.com/v5/public"
            self._ws_private_url = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self._ws_public_url = "wss://stream.bybit.com/v5/public"
            self._ws_private_url = "wss://stream.bybit.com/v5/private"
        
        # WebSocket clients
        self._public_ws: Optional[WebSocketClient] = None
        self._private_ws: Optional[WebSocketClient] = None
        
        # Callbacks para eventos WebSocket
        self._orderbook_callbacks: Dict[str, Callable[[OrderBook], Awaitable[None]]] = {}
        self._trade_callbacks: Dict[str, Callable[[Trade], Awaitable[None]]] = {}
        self._candle_callbacks: Dict[str, Dict[TimeFrame, Callable[[Candle], Awaitable[None]]]] = {}
        self._funding_rate_callbacks: Dict[str, Callable[[FundingRate], Awaitable[None]]] = {}
        self._liquidation_callbacks: Dict[str, Callable[[Liquidation], Awaitable[None]]] = {}