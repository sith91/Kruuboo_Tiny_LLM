"""
Live Database Knowledge Connector for Kruuboo Tiny LLM RAG
==========================================================

Allows connecting live external databases directly to `RAGSystem` so that
runtime queries retrieve matching rows/documents from live tables or collections
alongside the local vector/BM25 semantic index.

Supported Engines:
  1. SQLite: `sqlite:///path/to/file.db` or direct `.db` / `.sqlite` / `.sqlite3` path
     (uses Python's built-in `sqlite3` in read-only mode — zero extra dependencies).
  2. PostgreSQL / MySQL / Generic SQL: `postgresql://...`, `mysql://...`, `mariadb://...`
     (uses `sqlalchemy` if available, or native DB-API drivers `psycopg2` / `pymysql`).
  3. MongoDB: `mongodb://...` or `mongodb+srv://...`
     (uses `pymongo`).
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .document import Document
from .search import STOPWORDS

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful search keywords from a natural-language query."""
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", query.lower())
    casual = {
        "hi", "hello", "hey", "greetings", "thanks", "thank", "bye",
        "show", "list", "table", "database", "db", "record", "records", "row", "rows",
    }
    seen = set()
    keywords = []
    for w in raw_tokens:
        if w not in STOPWORDS and w not in casual and len(w) > 1 and w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords


class DatabaseKnowledgeSource:
    """
    Live read-only database connector that queries SQL tables or MongoDB collections
    at runtime and returns matching records as `Document` objects with relevance scores.
    """

    def __init__(
        self,
        uri: str,
        name: Optional[str] = None,
        tables: Optional[List[str]] = None,
        custom_query: Optional[str] = None,
        max_rows_per_table: int = 5,
    ):
        self.uri = uri.strip()
        self.allowed_tables = [t.strip() for t in tables if t.strip()] if tables else None
        self.custom_query = custom_query
        self.max_rows_per_table = max_rows_per_table

        self.engine_type = self._detect_engine(self.uri)
        self.name = name or self._default_name(self.uri)

        # Internal connection handles & cached schema
        self._sqlite_path: Optional[Path] = None
        self._sa_engine = None
        self._mongo_db = None
        self.schema: Dict[str, List[str]] = {}

        self._connect_and_introspect()

    @staticmethod
    def _detect_engine(uri: str) -> str:
        lower = uri.lower()
        if lower.startswith("sqlite:///"):
            return "sqlite"
        if lower.startswith(("mongodb://", "mongodb+srv://")):
            return "mongodb"
        if "://" in lower:
            return "sqlalchemy"
        # Assume local SQLite file path
        return "sqlite"

    @staticmethod
    def _default_name(uri: str) -> str:
        if "://" not in uri:
            return Path(uri).stem
        parsed = urlparse(uri)
        db_name = parsed.path.lstrip("/") or parsed.hostname or "db"
        return Path(db_name).stem

    def _connect_and_introspect(self) -> None:
        """Connect to the database and discover searchable tables/columns."""
        if self.engine_type == "sqlite":
            raw_path = self.uri[len("sqlite:///"):] if self.uri.lower().startswith("sqlite:///") else self.uri
            db_path = Path(raw_path).expanduser().resolve()
            if not db_path.exists():
                raise FileNotFoundError(f"SQLite database file not found: {db_path}")
            self._sqlite_path = db_path
            self._introspect_sqlite()

        elif self.engine_type == "sqlalchemy":
            try:
                from sqlalchemy import create_engine, inspect
            except ImportError as e:
                raise ImportError(
                    "Connecting to PostgreSQL/MySQL/SQL databases requires `sqlalchemy`. "
                    "Install it with: pip install sqlalchemy"
                ) from e

            self._sa_engine = create_engine(self.uri, pool_pre_ping=True)
            insp = inspect(self._sa_engine)
            all_tables = insp.get_table_names()
            target_tables = (
                [t for t in all_tables if t in self.allowed_tables]
                if self.allowed_tables
                else all_tables
            )
            for tbl in target_tables:
                cols = [c["name"] for c in insp.get_columns(tbl)]
                if cols:
                    self.schema[tbl] = cols

        elif self.engine_type == "mongodb":
            try:
                from pymongo import MongoClient
            except ImportError as e:
                raise ImportError(
                    "Connecting to MongoDB requires `pymongo`. Install it with: pip install pymongo"
                ) from e

            client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            parsed = urlparse(self.uri)
            db_name = parsed.path.lstrip("/").split("?")[0]
            if not db_name:
                raise ValueError("MongoDB URI must include a database name, e.g. mongodb://localhost:27017/mydb")
            self._mongo_db = client[db_name]
            all_colls = self._mongo_db.list_collection_names()
            target_colls = (
                [c for c in all_colls if c in self.allowed_tables]
                if self.allowed_tables
                else all_colls
            )
            for coll_name in target_colls:
                sample = self._mongo_db[coll_name].find_one()
                if sample and isinstance(sample, dict):
                    self.schema[coll_name] = [k for k in sample.keys() if k != "_id"]
                else:
                    self.schema[coll_name] = []

        logger.info(
            f"🔌 Connected live DB source '{self.name}' ({self.engine_type}) — "
            f"{len(self.schema)} tables/collections discovered: {list(self.schema.keys())}"
        )

    def _get_sqlite_conn(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection."""
        uri_ro = f"file:{self._sqlite_path}?mode=ro"
        conn = sqlite3.connect(uri_ro, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _introspect_sqlite(self) -> None:
        conn = self._get_sqlite_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = [r[0] for r in cur.fetchall()]
            if self.allowed_tables:
                tables = [t for t in tables if t in self.allowed_tables]

            for tbl in tables:
                if not _IDENTIFIER_RE.match(tbl):
                    continue
                cur.execute(f'PRAGMA table_info("{tbl}")')
                cols = [row[1] for row in cur.fetchall() if _IDENTIFIER_RE.match(row[1])]
                if cols:
                    self.schema[tbl] = cols
        finally:
            conn.close()

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Execute a live read-only search across the connected database for `query`
        and return up to `k` scored `(Document, score)` tuples.
        """
        keywords = _extract_keywords(query)
        if not keywords:
            return []

        try:
            if self.engine_type == "sqlite":
                return self._search_sqlite(keywords, k=k)
            elif self.engine_type == "sqlalchemy":
                return self._search_sqlalchemy(keywords, k=k)
            elif self.engine_type == "mongodb":
                return self._search_mongodb(keywords, k=k)
        except Exception as e:
            logger.warning(f"Live DB search error on '{self.name}': {e}")
        return []

    def _score_and_format_row(
        self,
        table: str,
        row_dict: Dict[str, Any],
        keywords: List[str],
    ) -> Optional[Tuple[Document, float]]:
        """Format a database row as a semantic Document and compute keyword relevance."""
        parts = []
        combined_text_lower = []
        for col, val in row_dict.items():
            if val is None:
                continue
            val_str = str(val).strip()
            if not val_str:
                continue
            parts.append(f"{col}: {val_str}")
            combined_text_lower.append(f"{col.lower()} {val_str.lower()}")

        if not parts:
            return None

        full_row_lower = " ".join(combined_text_lower)
        matched_kw = sum(1 for kw in keywords if kw in full_row_lower)
        if matched_kw == 0:
            return None

        # Score scales between 0.55 and 0.95 based on fraction of query keywords matched
        kw_ratio = matched_kw / max(1, len(keywords))
        score = min(0.95, 0.50 + 0.45 * kw_ratio)

        text = f"[Database: {self.name} | Table: {table}] " + " | ".join(parts)
        doc = Document(
            text=text,
            source=f"db:{self.name}:{table}",
            metadata={
                "db_name": self.name,
                "db_engine": self.engine_type,
                "table": table,
                "matched_keywords": matched_kw,
            },
        )
        return doc, score

    def _search_sqlite(self, keywords: List[str], k: int = 3) -> List[Tuple[Document, float]]:
        results: List[Tuple[Document, float]] = []
        conn = self._get_sqlite_conn()
        try:
            cur = conn.cursor()

            # Optional custom SQL query (must contain :query or ? placeholder)
            if self.custom_query:
                like_param = f"%{keywords[0]}%"
                cur.execute(self.custom_query, (like_param,))
                for row in cur.fetchall()[:k]:
                    res = self._score_and_format_row("custom_query", dict(row), keywords)
                    if res:
                        results.append(res)
                return sorted(results, key=lambda x: x[1], reverse=True)[:k]

            for table, cols in self.schema.items():
                if not cols:
                    continue
                # Build parameterized WHERE clause matching any keyword across any column
                clauses = []
                params = []
                for kw in keywords[:6]:
                    col_ors = [f'CAST("{col}" AS TEXT) LIKE ?' for col in cols]
                    clauses.append("(" + " OR ".join(col_ors) + ")")
                    params.extend([f"%{kw}%"] * len(cols))

                where_sql = " OR ".join(clauses)
                sql = f'SELECT * FROM "{table}" WHERE {where_sql} LIMIT ?'
                params.append(self.max_rows_per_table)

                try:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    for row in rows:
                        res = self._score_and_format_row(table, dict(row), keywords)
                        if res:
                            results.append(res)
                except Exception as tbl_err:
                    logger.debug(f"Skipping table {table} in {self.name}: {tbl_err}")
        finally:
            conn.close()

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _search_sqlalchemy(self, keywords: List[str], k: int = 3) -> List[Tuple[Document, float]]:
        from sqlalchemy import text

        results: List[Tuple[Document, float]] = []
        with self._sa_engine.connect() as conn:
            for table, cols in self.schema.items():
                if not cols or not _IDENTIFIER_RE.match(table):
                    continue
                safe_cols = [c for c in cols if _IDENTIFIER_RE.match(c)]
                if not safe_cols:
                    continue

                clauses = []
                params: Dict[str, Any] = {"lim": self.max_rows_per_table}
                for idx, kw in enumerate(keywords[:6]):
                    pname = f"kw_{idx}"
                    params[pname] = f"%{kw}%"
                    col_ors = [f"LOWER(CAST({col} AS CHAR)) LIKE :{pname}" for col in safe_cols]
                    clauses.append("(" + " OR ".join(col_ors) + ")")

                where_sql = " OR ".join(clauses)
                stmt = text(f"SELECT * FROM {table} WHERE {where_sql} LIMIT :lim")
                try:
                    rows = conn.execute(stmt, params).mappings().all()
                    for row in rows:
                        res = self._score_and_format_row(table, dict(row), keywords)
                        if res:
                            results.append(res)
                except Exception as tbl_err:
                    logger.debug(f"Skipping SQL table {table} in {self.name}: {tbl_err}")

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _search_mongodb(self, keywords: List[str], k: int = 3) -> List[Tuple[Document, float]]:
        results: List[Tuple[Document, float]] = []
        regex_pattern = "|".join(re.escape(kw) for kw in keywords[:6])

        for coll_name, fields in self.schema.items():
            coll = self._mongo_db[coll_name]
            if not fields:
                continue
            or_conditions = [{field: {"$regex": regex_pattern, "$options": "i"}} for field in fields]
            try:
                cursor = coll.find({"$or": or_conditions}, {"_id": 0}).limit(self.max_rows_per_table)
                for doc_dict in cursor:
                    res = self._score_and_format_row(coll_name, doc_dict, keywords)
                    if res:
                        results.append(res)
            except Exception as coll_err:
                logger.debug(f"Skipping MongoDB collection {coll_name}: {coll_err}")

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def execute_readonly_query(self, sql_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Execute a direct read-only SELECT query for inspection or CLI `/db query`."""
        cleaned = sql_query.strip()
        if not cleaned.lower().startswith(("select", "pragma", "explain", "show", "describe")):
            raise ValueError("Only read-only SELECT / SHOW / DESCRIBE / PRAGMA queries are allowed.")

        if self.engine_type == "sqlite":
            conn = self._get_sqlite_conn()
            try:
                cur = conn.cursor()
                cur.execute(cleaned)
                return [dict(r) for r in cur.fetchmany(limit)]
            finally:
                conn.close()
        elif self.engine_type == "sqlalchemy":
            from sqlalchemy import text

            with self._sa_engine.connect() as conn:
                rows = conn.execute(text(cleaned)).mappings().fetchmany(limit)
                return [dict(r) for r in rows]
        else:
            raise ValueError("Direct SQL queries are only supported on SQLite and SQLAlchemy sources.")

    def to_info_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "engine": self.engine_type,
            "tables": self.schema,
        }
