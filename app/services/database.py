from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
import sqlglot

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Database service for read-only query execution.
    Implements connection pooling and query safety.
    """

    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.engine = create_async_engine(
                settings.database_url,
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                echo=False
            )

            self.async_session_maker = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Also create asyncpg pool for raw queries
            self.pool = await asyncpg.create_pool(
                settings.database_url.replace("+asyncpg", ""),
                min_size=5,
                max_size=settings.database_pool_size,
                command_timeout=settings.max_query_execution_time
            )

            logger.info("Database service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}", exc_info=True)
            raise

    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
        if self.pool:
            await self.pool.close()
        logger.info("Database service closed")

    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager."""
        async with self.async_session_maker() as session:
            yield session

    async def execute_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[List[Any]], int, float]:
        """
        Execute a read-only SQL query.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            Tuple of (columns, rows, row_count, execution_time_ms)
        """
        import time
        start_time = time.time()

        try:
            async with self.pool.acquire() as conn:
                # Set read-only transaction
                async with conn.transaction(read_only=True):
                    # Execute query
                    results = await conn.fetch(sql, *(params.values() if params else []))

                    # Extract columns and rows
                    if results:
                        columns = list(results[0].keys())
                        rows = [list(row.values()) for row in results]
                    else:
                        columns = []
                        rows = []

                    row_count = len(rows)
                    execution_time_ms = (time.time() - start_time) * 1000

                    logger.info(
                        f"Query executed successfully: {row_count} rows in {execution_time_ms:.2f}ms"
                    )

                    return columns, rows, row_count, execution_time_ms

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise

    async def get_schema_info(self) -> Dict[str, Any]:
        """
        Get database schema information for SQL generation context.

        Returns:
            Schema information including tables, columns, and relationships
        """
        try:
            async with self.pool.acquire() as conn:
                # Get all tables
                tables_query = """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """
                tables = await conn.fetch(tables_query)
                table_names = [row["table_name"] for row in tables]

                # Get columns for each table
                schema_info = {}

                for table in table_names:
                    columns_query = """
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = $1
                        ORDER BY ordinal_position
                    """
                    columns = await conn.fetch(columns_query, table)

                    schema_info[table] = {
                        "columns": [
                            {
                                "name": col["column_name"],
                                "type": col["data_type"],
                                "nullable": col["is_nullable"] == "YES"
                            }
                            for col in columns
                        ]
                    }

                # Get table statistics
                stats_query = """
                    SELECT
                        schemaname,
                        tablename,
                        n_live_tup as row_count
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                """
                stats = await conn.fetch(stats_query)

                for stat in stats:
                    table_name = stat["tablename"]
                    if table_name in schema_info:
                        schema_info[table_name]["row_count"] = stat["row_count"]

                return schema_info

        except Exception as e:
            logger.error(f"Failed to get schema info: {e}", exc_info=True)
            raise

    async def get_schema_context(self, tables: Optional[List[str]] = None) -> str:
        """
        Get formatted schema context for LLM prompt.

        Args:
            tables: Optional list of tables to include (all if None)

        Returns:
            Formatted schema string
        """
        schema_info = await self.get_schema_info()

        context_lines = []

        for table_name, table_info in schema_info.items():
            if tables and table_name not in tables:
                continue

            context_lines.append(f"Table: {table_name}")
            context_lines.append(f"  Columns:")

            for col in table_info["columns"]:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                context_lines.append(f"    - {col['name']}: {col['type']} {nullable}")

            if "row_count" in table_info:
                context_lines.append(f"  Row count: {table_info['row_count']:,}")

            context_lines.append("")

        return "\n".join(context_lines)

    async def check_connection(self) -> bool:
        """Check database connection."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return False
