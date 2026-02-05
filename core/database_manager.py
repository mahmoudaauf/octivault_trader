import aiosqlite
import logging
import asyncio # Import asyncio for sleep
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("DatabaseManager")

# Custom JSON serializer for types not natively serializable (e.g., datetime objects)
def default_serializer(obj):
    """
    Default JSON serializer for objects that are not JSON serializable by default.
    Converts datetime objects to ISO 8601 format.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class DatabaseManager:

    async def load_shared_state_snapshot(self) -> dict:
        """Load the latest SharedState snapshot from the database."""
        # TODO: Implement actual DB retrieval logic here
        # For now, return an empty snapshot structure
        return {
            "accepted_symbols": [],
            "positions": {},
            "balances": {},
            "exposure_target": None,
            "cooldowns": {},
            "reservations": {},
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0
        }

    async def save_shared_state_snapshot(self, snapshot: dict):
        """Save a snapshot of SharedState to the database."""
        # TODO: Implement actual DB insert/update logic here
        # This is a stub.
        pass

    # Added shared_state parameter to __init__
    def __init__(self, config, shared_state=None):
        # Configuration object should have a DATABASE_PATH attribute
        self.db_path = config.DATABASE_PATH
        self._db_connection = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"DatabaseManager initialized. DB Path: {self.db_path}")
        self.lock = asyncio.Lock() # Initialize the lock for thread-safe operations
        self.shared_state = shared_state # Store the shared_state object

    def is_connection_open(self):
        """Checks if the database connection is currently open."""
        try:
            # aiosqlite connection object has an internal _conn attribute
            # that is None if the connection is closed.
            return self._db_connection is not None and self._db_connection._conn is not None
        except Exception:
            # If any error occurs (e.g., _db_connection is not an aiosqlite connection yet)
            return False

    async def connect(self):
        """Establishes a connection to the SQLite database and creates tables if they don't exist."""
        try:
            # Add timeout and explicit journal_mode
            self._db_connection = await aiosqlite.connect(self.db_path, timeout=10) # Add timeout in seconds
            await self._db_connection.execute('PRAGMA journal_mode=WAL;') # Set WAL mode for better concurrency
            self._db_connection.row_factory = aiosqlite.Row # Access columns by name
            await self._create_tables()
            self.logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            self.logger.critical(f"Failed to connect to database at {self.db_path}: {e}", exc_info=True)
            raise # Re-raise to prevent application from starting without DB

    async def close(self):
        """Closes the database connection."""
        if self._db_connection:
            # Add any cleanup for pending tasks BEFORE event loop closes
            self.logger.info("Initiating database connection closure. Ensure all pending tasks/threads are stopped.")
            await self._db_connection.close()
            self.logger.info("Database connection closed.")

    async def disconnect(self):
        """Closes the database connection (alias for close)."""
        if self._db_connection:
            # Add any cleanup for pending tasks BEFORE event loop closes
            self.logger.info("Initiating database connection disconnection. Ensure all pending tasks/threads are stopped.")
            await self._db_connection.close()
            self.logger.info("Database connection disconnected.")

    async def _create_tables(self):
        """Creates necessary tables if they do not already exist."""
        schemas = [
            """
            CREATE TABLE IF NOT EXISTS historical_klines (
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                trade_count INTEGER,
                PRIMARY KEY (symbol, timestamp)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS trade_orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                side TEXT NOT NULL,
                type TEXT NOT NULL,
                price REAL,
                quantity REAL,
                status TEXT NOT NULL,
                fills TEXT,
                client_order_id TEXT,
                strategy_id TEXT,
                pnl REAL,
                fee REAL,
                entry_price REAL,
                close_price REAL,
                direction TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL UNIQUE,
                total_balance REAL,
                available_balance REAL,
                locked_balance REAL,
                net_worth REAL,
                pnl REAL DEFAULT 0.0,
                holdings_json TEXT -- Stores balances and positions as JSON
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """,
            """
            -- New table for symbols and their metadata
            CREATE TABLE IF NOT EXISTS symbols_metadata (
                symbol TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL -- Stores metadata as JSON string
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS symbol_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                source TEXT,
                timestamp TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS accepted_symbols (
                symbol TEXT PRIMARY KEY,
                metadata TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT PRIMARY KEY,
                source TEXT,
                added_at REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS symbol_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbols TEXT NOT NULL,  -- ✅ this is required
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS pending_position_intents (
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                target_quote REAL,
                accumulated_quote REAL,
                min_notional REAL,
                ttl_sec INTEGER,
                source_agent TEXT,
                state TEXT,
                created_at REAL,
                PRIMARY KEY (symbol, side)
            );
            """
        ]

        for schema in schemas:
            try:
                await self._db_connection.execute(schema)
                # Check connection status before committing
                if self.is_connection_open():
                    await self._db_connection.commit()
                else:
                    self.logger.warning("⚠️ Commit skipped: DB connection already closed during table creation.")
            except Exception as e:
                self.logger.error(f"Error creating table with schema: {schema[:50]}... Error: {e}", exc_info=True)
        self.logger.info("Database tables checked/created.")

    async def _execute_query(self, query: str, params: tuple = (), retries: int = 3, delay: float = 0.2):
        """Internal helper to execute a query (SELECT, INSERT, UPDATE, DELETE) with retry logic."""
        # Prevent DB operations if connection is closed
        if self._db_connection is None or not self.is_connection_open():
            self.logger.warning("⚠️ DB operation skipped: connection is closed or not established.")
            return None

        for attempt in range(retries):
            try:
                self.logger.debug(f"DEBUG DB: Attempting to execute query (Attempt {attempt+1}/{retries}): {query.strip().splitlines()[0]} with params: {params}")
                async with self._db_connection.execute(query, params) as cursor:
                    self.logger.debug("DEBUG DB: Query executed, cursor obtained.")
                    if query.strip().upper().startswith("SELECT"):
                        result = await cursor.fetchall()
                        self.logger.debug("DEBUG DB: FetchAll completed for SELECT query.")
                        return result
                    else:
                        self.logger.debug("DEBUG DB: Attempting to commit transaction for INSERT/UPDATE/DELETE.")
                        # Check connection status before committing
                        if self.is_connection_open():
                            await self._db_connection.commit()
                            self.logger.debug("DEBUG DB: Transaction committed.")
                        else:
                            self.logger.warning("⚠️ Commit skipped: DB connection already closed during query execution.")
                        return cursor.lastrowid
            except aiosqlite.OperationalError as e:
                if "database is locked" in str(e).lower(): # Check for lowercase "database is locked"
                    self.logger.warning(f"🔒 DB locked, retrying ({attempt+1}/{retries})... Error: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Error executing query: {query.strip().splitlines()[0]} with params {params}. Error: {e}", exc_info=True)
                    if self._db_connection and self.is_connection_open():
                        self.logger.debug("DEBUG DB: Rolling back due to OperationalError.")
                        await self._db_connection.rollback()
                    else:
                        self.logger.warning("⚠️ Rollback skipped: DB connection already closed.")
                    raise # Re-raise if not a database locked error
            except Exception as e:
                self.logger.error(f"An unexpected error occurred executing query: {query.strip().splitlines()[0]} with params {params}. Error: {e}", exc_info=True)
                if self._db_connection and self.is_connection_open():
                    await self._db_connection.rollback()
                else:
                    self.logger.warning("⚠️ Rollback skipped: DB connection already closed.")
                raise

        # If all retries fail
        self.logger.error(f"Failed to execute query after {retries} attempts: {query.strip().splitlines()[0]}")
        raise aiosqlite.OperationalError("Database is locked after multiple retries.")


    async def fetch_one(self, query: str, params: tuple = ()):
        """Fetches a single row from the database."""
        rows = await self._execute_query(query, params)
        return rows[0] if rows else None

    async def fetch_all(self, query: str, params: tuple = ()):
        """Fetches all rows from the database."""
        return await self._execute_query(query, params)

    async def insert_row(self, query: str, params: tuple = ()):
        """Inserts a single row and returns its last row ID."""
        return await self._execute_query(query, params)

    async def update_row(self, query: str, params: tuple = ()):
        """Updates rows."""
        await self._execute_query(query, params)

    async def delete_row(self, query: str, params: tuple = ()):
        """Deletes rows."""
        await self._execute_query(query, params)

    async def execute_query(self, query: str, params: tuple = ()):
        """A generic method to execute any query, similar to _execute_query but public."""
        return await self._execute_query(query, params)

    async def load_portfolio_snapshot(self) -> Optional[Dict[str, Any]]:
        """Loads the latest portfolio snapshot from the database."""
        query = "SELECT holdings_json, pnl FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
        rows = await self.fetch_all(query)
        if rows:
            row = rows[0]
            portfolio_data = dict(row)

            holdings_json_str = portfolio_data.get('holdings_json', '{}')
            holdings_data = json.loads(holdings_json_str) if holdings_json_str else {}

            balances = holdings_data.get('balances', {})
            positions = holdings_data.get('positions', {})
            pnl = portfolio_data.get('pnl', 0.0)

            if not isinstance(balances, dict):
                balances = {}
            if not isinstance(positions, dict):
                positions = {}

            return {
                'balances': balances,
                'positions': positions,
                'pnl': pnl
            }
        return None

    async def save_portfolio_snapshot(self, snapshot_data: Dict[str, Any]):
        self.logger.debug("DEBUG DB: Entered save_portfolio_snapshot.")
        balances = snapshot_data.get('balances', {})
        positions = snapshot_data.get('positions', {})
        pnl = snapshot_data.get('pnl', 0.0)
        total_balance = balances.get('USDT', {}).get('total', 0.0)
        available_balance = balances.get('USDT', {}).get('free', 0.0)
        locked_balance = balances.get('USDT', {}).get('locked', 0.0)
        net_worth = total_balance # Assuming total_balance is the base currency value
        self.logger.debug("DEBUG DB: save_portfolio_snapshot: Preparing holdings_json.")
        # Apply default_serializer here
        holdings_json = json.dumps({'balances': balances, 'positions': positions}, default=default_serializer)
        query = """
        INSERT INTO portfolio_snapshots (timestamp, total_balance, available_balance, locked_balance, net_worth, pnl, holdings_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (time.time(), total_balance, available_balance, locked_balance, net_worth, pnl, holdings_json)
        self.logger.debug("DEBUG DB: save_portfolio_snapshot: Preparing to call insert_row with query and params.")
        await self.insert_row(query, params)
        self.logger.debug("DEBUG DB: save_portfolio_snapshot: insert_row returned.")
        self.logger.debug("Portfolio snapshot saved via helper.")

    async def load_open_positions(self) -> List[Dict[str, Any]]:
        query = "SELECT order_id, symbol, timestamp, side, type, price, quantity, status, entry_price, direction FROM trade_orders WHERE status = 'open'"
        rows = await self.fetch_all(query)
        return [dict(row) for row in rows]

    async def save_open_position(self, position_data: Dict[str, Any]):
        query = """
        INSERT OR REPLACE INTO trade_orders (order_id, symbol, timestamp, side, type, price, quantity, status, fills, client_order_id, strategy_id, pnl, fee, entry_price, close_price, direction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        # Ensure all fields are present in position_data, providing defaults if necessary
        # Apply default_serializer here
        params = (
            position_data.get('order_id'),
            position_data.get('symbol'),
            position_data.get('timestamp'),
            position_data.get('side'),
            position_data.get('type'),
            position_data.get('price'),
            position_data.get('quantity'),
            position_data.get('status', 'open'),
            json.dumps(position_data.get('fills', []), default=default_serializer), # fills should be JSON string
            position_data.get('client_order_id'),
            position_data.get('strategy_id'),
            position_data.get('pnl'),
            position_data.get('fee'),
            position_data.get('entry_price'),
            position_data.get('close_price'),
            position_data.get('direction')
        )
        await self.insert_row(query, params)
        self.logger.debug(f"Open position for {position_data.get('symbol')} saved/updated via helper.")

    async def delete_open_position(self, symbol: str):
        # This method updates the status to 'closed' rather than deleting the row entirely
        query = "UPDATE trade_orders SET status = 'closed', close_price = ?, pnl = ?, timestamp = ? WHERE symbol = ? AND status = 'open'"
        params = (None, None, time.time(), symbol) # You might want to pass actual close_price and pnl here
        await self.update_row(query, params)
        self.logger.debug(f"Open position for {symbol} marked as closed via helper.")

    async def load_realized_pnl(self) -> float:
        query = "SELECT value FROM app_state WHERE key = 'realized_pnl_cumulative'"
        row = await self.fetch_one(query)
        return float(row['value']) if row and row['value'] else 0.0

    async def save_realized_pnl(self, pnl: float):
        query = "INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)"
        params = ('realized_pnl_cumulative', str(pnl))
        await self.insert_row(query, params)
        self.logger.debug(f"Cumulative realized PnL saved via helper: {pnl:.2f}")

    async def load_active_symbols(self) -> List[str]:
        query = "SELECT value FROM app_state WHERE key = 'active_symbols_list'"
        row = await self.fetch_one(query)
        return json.loads(row['value']) if row and row['value'] else []

    async def save_active_symbols(self, symbols: List[str]):
        query = "INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)"
        # Apply default_serializer here
        params = ('active_symbols_list', json.dumps(symbols, default=default_serializer))
        await self.insert_row(query, params)
        self.logger.debug(f"Active symbols list saved via helper: {symbols}")

    async def save_system_health(self, component: str, status: str, message: str, timestamp: int):
        """Inserts a health status entry for a system component with explicit timestamp."""
        if self._db_connection is None or not self.is_connection_open():
            self.logger.warning("❌ Cannot update health: DB connection is closed or not established.")
            return
        await self.execute_query(
            "INSERT INTO system_health (component, status, message, timestamp) VALUES (?, ?, ?, ?)",
            (component, status, message, timestamp)
        )
        self.logger.debug(f"Saved health for {component} to DB with status '{status}'.")

    async def load_system_health(self) -> Dict[str, Dict[str, Any]]:
        """Loads recent system health statuses from the database."""
        query = "SELECT component, status, message, timestamp FROM system_health ORDER BY timestamp DESC LIMIT 10"
        rows = await self.fetch_all(query)
        health_status = {}
        for row in rows:
            row_dict = dict(row) # Convert Row object to dict
            # Assuming timestamp is in milliseconds, convert to seconds for datetime.fromtimestamp
            health_status[row_dict['component']] = {
                "status": row_dict['status'],
                "message": row_dict['message'],
                "timestamp": datetime.fromtimestamp(row_dict['timestamp'] / 1000) # Convert ms to datetime
            }
        self.logger.debug("Loaded system health from DB.")
        return health_status

    async def load_symbols(self) -> Dict[str, Any]:
        """Loads all symbols with their metadata from the database."""
        query = "SELECT symbol, metadata_json FROM symbols_metadata"
        
        rows = await self.fetch_all(query)
        if not rows:
            self.logger.warning("⚠️ No symbols loaded from DB (fetch_all returned None or DB not connected).")
            return {}

        symbols_data = {}
        for row in rows:
            row_dict = dict(row) # Convert Row object to dict
            metadata = json.loads(row_dict['metadata_json']) if row_dict['metadata_json'] else {}
            symbols_data[row_dict['symbol']] = metadata
        self.logger.debug(f"Loaded {len(symbols_data)} symbols from DB.")
        return symbols_data

    async def save_symbol(self, symbol: str, metadata: Dict[str, Any]):
        """Saves or updates a single symbol with its metadata in the database."""
        try:
            # Apply default_serializer here
            metadata_json = json.dumps(metadata, default=default_serializer)
            query = "INSERT OR REPLACE INTO symbols_metadata (symbol, metadata_json) VALUES (?, ?)"
            await self.execute_query(query, (symbol, metadata_json))
            self.logger.debug(f"Saved/Updated symbol '{symbol}' metadata to DB.")
        except Exception as e:
            self.logger.error(f"❌ Failed to save symbol '{symbol}': {e}", exc_info=True)
            raise # Re-raise the exception after logging

    async def save_symbols(self, symbols: Dict[str, Dict[str, Any]]):
        """Saves or updates multiple symbols with their metadata."""
        for symbol, metadata in symbols.items():
            await self.save_symbol(symbol, metadata)
        self.logger.info(f"✅ Saved metadata for {len(symbols)} symbols to DB.")

    async def update_symbols(self, symbols: Dict[str, Any]):
        """
        Updates all symbols in the database. Replaces the entire symbols_metadata table with new data.
        This is usually called by SharedState when syncing current in-memory symbols to DB.
        """
        try:
            # Clear existing symbols
            await self.execute_query("DELETE FROM symbols_metadata")

            # Save each new symbol
            await self.save_symbols(symbols)

            self.logger.info(f"✅ update_symbols(): Updated {len(symbols)} symbols in DB.")
        except Exception as e:
            self.logger.error(f"❌ update_symbols(): Failed to update symbols: {e}", exc_info=True)
            raise

    async def save_symbol_stats(self, symbols: List[str], source: str = "unknown"):
        """
        Save basic symbol statistics or metadata for debugging/analysis.
        Extend this based on actual schema/design later.
        """
        try:
            now = datetime.utcnow().isoformat()
            # We will insert each symbol stat individually, relying on _execute_query's
            # internal transaction handling and retry logic.
            for symbol in symbols:
                await self.execute_query("""
                    INSERT INTO symbol_stats (symbol, source, timestamp)
                    VALUES (?, ?, ?)
                """, (symbol, source, now))
            self.logger.info(f"[DatabaseManager] ✅ Saved stats for {len(symbols)} symbols with source: {source}")
        except Exception as e:
            self.logger.exception(f"[DatabaseManager] ⚠️ Failed to save symbol stats: {e}")

    async def load_symbol_snapshot(self) -> dict:
        """
        Load last saved accepted symbols from the database and parse JSON metadata.
        """
        try:
            # Use fetch_all which internally uses _execute_query with retry logic
            rows = await self.fetch_all("SELECT symbol, metadata FROM accepted_symbols")
            
            if not rows:
                self.logger.info("No symbol snapshot found in the database.")
                return {}

            symbols = {}
            for row in rows:
                symbol = row["symbol"]
                try:
                    meta = json.loads(row["metadata"])  # ✅ Critical line
                    if not isinstance(meta, dict):
                        self.logger.warning(f"⚠️ Metadata for {symbol} is not a dict ({type(meta)})")
                        continue
                    symbols[symbol] = meta
                except json.JSONDecodeError as e:
                    self.logger.error(f"❌ JSON decode failed for {symbol}: {e}")
            self.logger.info(f"✅ Loaded {len(symbols)} symbols from snapshot.")
            return symbols
        except Exception as e:
            self.logger.error(f"❌ Failed to load symbol snapshot: {e}", exc_info=True)
            return {}

    async def save_symbol_snapshot(self, symbols: dict):
        """
        Save accepted symbols and their metadata as JSON strings into the database.
        """
        try:
            # Delete existing accepted symbols
            await self.execute_query("DELETE FROM accepted_symbols")
            
            # Insert new accepted symbols
            for symbol, metadata in symbols.items():
                await self.execute_query(
                    "INSERT OR REPLACE INTO accepted_symbols (symbol, metadata) VALUES (?, ?)",
                    (symbol, json.dumps(metadata, default=default_serializer))  # ✅ Serialize properly with default_serializer
                )
            self.logger.info(f"✅ Saved {len(symbols)} accepted symbols to snapshot.")
        except Exception as e:
            self.logger.error(f"❌ Failed to save symbol snapshot: {e}", exc_info=True)

    async def write_symbols_to_db(self, symbols: Dict[str, Dict[str, Any]]):
        """
        Writes a batch of symbols and their basic metadata (source, added_at) to the 'symbols' table.
        This operation is atomic: either all symbols are written, or none are.
        """
        try:
            # Begin a transaction for atomic operations
            await self._db_connection.execute("BEGIN TRANSACTION;")
            
            # Delete all existing symbols from the 'symbols' table
            await self._db_connection.execute("DELETE FROM symbols;")
            self.logger.debug("Deleted existing symbols from 'symbols' table.")

            # Insert new symbols
            for symbol_name, metadata in symbols.items():
                source = metadata.get("source", "unknown")
                # Use time.time() for current timestamp (float)
                added_at = metadata.get("added_at", time.time()) 
                
                await self._db_connection.execute(
                    "INSERT INTO symbols (symbol, source, added_at) VALUES (?, ?, ?)",
                    (symbol_name, source, added_at)
                )
            
            # Commit the transaction if all operations were successful
            await self._db_connection.commit()
            self.logger.info(f"💾 {len(symbols)} symbols written to DB.") # Updated log message
        except Exception as e:
            # Rollback the transaction if any error occurs
            if self._db_connection and self.is_connection_open():
                await self._db_connection.rollback()
                self.logger.error(f"❌ Rolled back transaction due to error: {e}", exc_info=True)
            else:
                self.logger.error(f"❌ Failed to write symbols to DB (connection closed or not established): {e}", exc_info=True)
            raise # Re-raise the exception after logging and rollback

    async def clear_symbols(self):
        """
        Clears all symbols from the 'symbols' table in the database.
        This operation is atomic.
        """
        async with self.lock:
            try:
                # Begin a transaction for atomic operation
                await self._db_connection.execute("BEGIN TRANSACTION;")
                await self._db_connection.execute("DELETE FROM symbols")
                await self._db_connection.commit()
                self.logger.info("🧹 Cleared all symbols from database.")
            except Exception as e:
                if self._db_connection and self.is_connection_open():
                    await self._db_connection.rollback()
                    self.logger.error(f"❌ Failed to clear symbols and rolled back: {e}", exc_info=True)
                else:
                    self.logger.error(f"❌ Failed to clear symbols (DB connection closed or not established): {e}", exc_info=True)
                raise # Re-raise the exception


    async def save_symbols_metadata_batch(self, symbols: Dict[str, Dict[str, Any]]):
        """
        Saves all symbols and their metadata in batch to symbols_metadata.
        Equivalent to save_symbols(), but matches expected method name in SharedState.
        """
        await self.save_symbols(symbols)

    async def fetch_symbols(self) -> list:
        self.logger.info("📤 Fetching symbols from database...")
        query = "SELECT symbol, source, added_at FROM symbols"
        try:
            # Using self._db_connection and self.lock for consistency
            async with self.lock:
                async with self._db_connection.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    self.logger.info(f"📥 Fetched {len(rows)} symbols.")
                    return [dict(row) for row in rows]
        except Exception as e:
            self.logger.exception(f"❌ Error fetching symbols: {e}")
            return []

    async def get_all_symbols(self) -> List[str]:
        """
        Fetch all symbol names stored in the database.
        """
        self.logger.info("Retrieving all symbol names from the database...")
        query = "SELECT symbol FROM symbols"
        async with self.lock:
            try:
                async with self._db_connection.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    symbol_names = [row[0] for row in rows]
                    self.logger.info(f"Successfully retrieved {len(symbol_names)} symbol names.")
                    return symbol_names
            except Exception as e:
                self.logger.error(f"❌ Failed to retrieve all symbol names from DB: {e}", exc_info=True)
                return []

    async def get_latest_symbol_snapshot(self):
        """
        Fetches the latest symbol snapshot from the 'symbol_snapshots' table.
        Returns a list of symbol dictionaries, or None if no snapshot is found or decoding fails.
        """
        async with self.lock:
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    db.row_factory = aiosqlite.Row # Ensure row_factory is set for this connection
                    async with db.execute("SELECT symbols FROM symbol_snapshots ORDER BY created_at DESC LIMIT 1") as cursor:
                        row = await cursor.fetchone()
                        if row:
                            try:
                                # Properly parse JSON string into Python list
                                return json.loads(row[0])
                            except json.JSONDecodeError as e:
                                self.logger.error(f"❌ JSON decode error in snapshot: {e}", exc_info=True)
                                return None
                        return None
            except Exception as e:
                self.logger.error(f"❌ DB failure during symbol snapshot load: {e}", exc_info=True)
                return None


    async def save_current_symbols_list_snapshot(self): # Removed symbol_list parameter
        """
        Saves a snapshot of a list of symbols (with metadata) to the 'symbol_snapshots' table.
        This method now directly uses self.shared_state.accepted_symbols and get_symbol_metadata.
        """
        if self.shared_state is None:
            self.logger.error("❌ SharedState is not set in DatabaseManager. Cannot save symbol list snapshot.")
            return

        if not self.shared_state.accepted_symbols:
            self.logger.warning("⚠️ No symbols in SharedState.accepted_symbols for snapshot.")
            return

        created_at = datetime.utcnow().isoformat() # Use ISO format for DATETIME column
        
        # Construct the list of dictionaries with symbol and its metadata
        snapshot_data = [
            {"symbol": sym, **self.shared_state.get_symbol_metadata(sym)}
            for sym in self.shared_state.accepted_symbols
        ]
        symbols_json = json.dumps(snapshot_data, default=default_serializer)

        try:
            async with self.lock: # Ensure thread safety
                await self._db_connection.execute(
                    """
                    INSERT INTO symbol_snapshots (created_at, symbols)
                    VALUES (?, ?)
                    """,
                    (created_at, symbols_json)
                )
                await self._db_connection.commit()
                self.logger.info(f"💾 Snapshot of {len(snapshot_data)} symbols saved to DB at {created_at}.")
        except Exception as e:
            if self._db_connection and self.is_connection_open():
                await self._db_connection.rollback()
                self.logger.error(f"❌ Failed to save symbol list snapshot and rolled back: {e}", exc_info=True)
            else:
                self.logger.error(f"❌ Failed to save symbol list snapshot (DB connection closed or not established): {e}", exc_info=True)
            raise # Re-raise the exception

    # -------------------
    # Pending Position Accumulation (P9 Phase 4)
    # -------------------
    async def save_pending_intent(self, intent: Dict[str, Any]):
        """Persist or update a pending position intent."""
        query = """
        INSERT OR REPLACE INTO pending_position_intents 
        (symbol, side, target_quote, accumulated_quote, min_notional, ttl_sec, source_agent, state, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            intent.get("symbol"),
            intent.get("side"),
            intent.get("target_quote"),
            intent.get("accumulated_quote"),
            intent.get("min_notional"),
            intent.get("ttl_sec"),
            intent.get("source_agent"),
            intent.get("state"),
            intent.get("created_at")
        )
        await self.execute_query(query, params)

    async def delete_pending_intent(self, symbol: str, side: str):
        """Remove a pending intent from DB."""
        await self.execute_query(
            "DELETE FROM pending_position_intents WHERE symbol = ? AND side = ?",
            (symbol, side)
        )

    async def load_pending_intents(self) -> List[Dict[str, Any]]:
        """Load all pending intents from DB."""
        rows = await self.fetch_all("SELECT * FROM pending_position_intents")
        return [dict(row) for row in rows]

    async def snapshot_symbols(self, symbols_dict: Dict[str, Dict[str, Any]]):
        """
        Save accepted symbols snapshot to database (used by Phase 3).
        """
        if not symbols_dict:
            self.logger.warning("⚠️ No symbols to snapshot.")
            return

        created_at = datetime.utcnow().isoformat()
        
        # Prepare the list of symbol dictionaries for JSON serialization
        symbols_list_for_json = [
            {"symbol": sym, **meta} for sym, meta in symbols_dict.items()
        ]

        query = """
        INSERT INTO symbol_snapshots (created_at, symbols)
        VALUES (?, ?)
        """
        try:
            async with self.lock: # Ensure thread safety
                await self._db_connection.execute(query, (created_at, json.dumps(symbols_list_for_json)))
                await self._db_connection.commit()
                self.logger.info(f"✅ Snapshotted {len(symbols_dict)} symbols.")
        except Exception as e:
            if self._db_connection and self.is_connection_open():
                await self._db_connection.rollback()
                self.logger.error(f"❌ Failed to snapshot symbols and rolled back: {e}", exc_info=True)
            else:
                self.logger.error(f"❌ Failed to snapshot symbols (DB connection closed or not established): {e}", exc_info=True)
            raise # Re-raise the exception


    async def load_portfolio(self) -> Dict[str, Any]:
        """
        Loads the complete portfolio state from the database, combining balances,
        open positions, and realized PnL.
        """
        self.logger.info("Loading complete portfolio state from DB.")
        portfolio_state = {
            'balances': defaultdict(lambda: {'free': 0.0, 'locked': 0.0, 'total': 0.0}),
            'open_positions': {},
            'realized_pnl': 0.0
        }
        try:
            # Load from the latest portfolio snapshot
            portfolio_snapshot = await self.load_portfolio_snapshot()
            if portfolio_snapshot:
                # Update balances and realized_pnl from snapshot
                for currency, data in portfolio_snapshot.get('balances', {}).items():
                    portfolio_state['balances'][currency].update(data)
                portfolio_state['realized_pnl'] = portfolio_snapshot.get('pnl', 0.0)
                self.logger.debug("Loaded balances and PnL from portfolio snapshot.")
            else:
                self.logger.info("No portfolio snapshot found in DB.")

            # Load open positions from trade_orders table
            open_positions_list = await self.load_open_positions()
            for pos in open_positions_list:
                portfolio_state['open_positions'][pos['symbol']] = pos
            self.logger.debug(f"Loaded {len(open_positions_list)} open positions from DB.")

            # Load cumulative realized PnL (if stored separately and not just in snapshots)
            cumulative_pnl_from_app_state = await self.load_realized_pnl()
            if cumulative_pnl_from_app_state is not None:
                portfolio_state['realized_pnl'] = cumulative_pnl_from_app_state
                self.logger.debug(f"Overwrote realized PnL with cumulative from app_state: {cumulative_pnl_from_app_state:.2f}")


            self.logger.info("✅ Complete portfolio state loaded from database.")
            return portfolio_state
        except Exception as e:
            self.logger.error(f"❌ Failed to load complete portfolio from DB: {e}", exc_info=True)
            return portfolio_state # Return partial or empty state on error

    async def save_health_log(self, component: str, status: str, message: str):
        """Public wrapper to insert a health log with current timestamp (in ms)."""
        timestamp = int(time.time() * 1000)
        try:
            await self.save_system_health(component, status, message, timestamp)
        except Exception as e:
            self.logger.error(f"❌ Failed to save health log for {component}: {e}", exc_info=True)

    async def write_symbol_snapshot(self, symbols: dict, phase: str = "unknown"):
        """Writes a snapshot of the current accepted symbols to the database."""
        timestamp = datetime.utcnow().isoformat()
        
        # Ensure symbols is a list of dictionaries before serialization
        if isinstance(symbols, dict):
            symbols_to_serialize = [{'symbol': sym, **meta} for sym, meta in symbols.items()]
        else:
            symbols_to_serialize = symbols # Assume it's already in the correct list format

        serialized = json.dumps(symbols_to_serialize, default=default_serializer) # Use default_serializer for consistency
        try:
            async with self.lock: # Ensure thread safety
                await self._db_connection.execute(
                    "INSERT INTO symbol_snapshots (symbols, created_at) VALUES (?, ?)",
                    (serialized, timestamp)
                )
                await self._db_connection.commit()
                self.logger.info(f"💾 Snapshot saved with {len(symbols_to_serialize)} symbols at {timestamp} (phase: {phase})")
        except Exception as e:
            if self._db_connection and self.is_connection_open():
                await self._db_connection.rollback()
                self.logger.error(f"❌ Failed to write symbol snapshot and rolled back: {e}", exc_info=True)
            else:
                self.logger.error(f"❌ Failed to write symbol snapshot (DB connection closed or not established): {e}", exc_info=True)
            raise # Re-raise the exception
