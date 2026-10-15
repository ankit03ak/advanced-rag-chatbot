from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import sqlite3
import requests
import os
import tempfile
import hashlib
import hmac
import json
import logging
import pickle
import secrets
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None


load_dotenv()

# -------------------
# 0. Variables
# -------------------
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
AUTH_SESSION_TTL = timedelta(hours=12)
DATABASE_URL = os.getenv("DATABASE_URL")
_USING_POSTGRES = bool(DATABASE_URL)

if _USING_POSTGRES:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set, but PostgreSQL dependencies are missing. "
            "Install psycopg[binary] and langgraph-checkpoint-postgres."
        ) from exc
    if PostgresSaver is None:
        raise RuntimeError(
            "DATABASE_URL is set, but langgraph-checkpoint-postgres is missing."
        )
    _AUTH_DB = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
    )
else:
    _AUTH_DB = sqlite3.connect(
        os.getenv("SQLITE_DB_PATH", "chatbot.db"),
        check_same_thread=False,
    )
    _AUTH_DB.row_factory = sqlite3.Row

_AUTH_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        expires_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS thread_owners (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT 'New chat'
    );
"""

if _USING_POSTGRES:
    with _AUTH_DB.cursor() as cursor:
        for statement in _AUTH_SCHEMA.split(";"):
            if statement.strip():
                cursor.execute(statement)
else:
    _AUTH_DB.executescript(_AUTH_SCHEMA)
    _AUTH_DB.commit()


def _db_execute(query: str, parameters: tuple = ()):
    if _USING_POSTGRES:
        query = query.replace("?", "%s")
        cursor = _AUTH_DB.cursor()
        cursor.execute(query, parameters)
        return cursor
    return _AUTH_DB.execute(query, parameters)


def _db_commit() -> None:
    if not _USING_POSTGRES:
        _AUTH_DB.commit()


def _db_fetchone(query: str, parameters: tuple = ()):
    cursor = _db_execute(query, parameters)
    try:
        return cursor.fetchone()
    finally:
        if _USING_POSTGRES:
            cursor.close()


def _db_fetchall(query: str, parameters: tuple = ()):
    cursor = _db_execute(query, parameters)
    try:
        return cursor.fetchall()
    finally:
        if _USING_POSTGRES:
            cursor.close()


def _db_rollback() -> None:
    if not _USING_POSTGRES:
        _AUTH_DB.rollback()


def _ensure_thread_title_column() -> None:
    if _USING_POSTGRES:
        _db_execute(
            "ALTER TABLE thread_owners ADD COLUMN IF NOT EXISTS "
            "title TEXT NOT NULL DEFAULT 'New chat'"
        )
        _db_commit()
        return

    columns = _AUTH_DB.execute("PRAGMA table_info(thread_owners)").fetchall()
    if not any(column[1] == "title" for column in columns):
        _AUTH_DB.execute(
            "ALTER TABLE thread_owners ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat'"
        )
        _AUTH_DB.commit()


_ensure_thread_title_column()


_DB_INTEGRITY_ERROR = (
    psycopg.IntegrityError if _USING_POSTGRES else sqlite3.IntegrityError
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt_hex, digest_hex = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        candidate = _hash_password(password, bytes.fromhex(salt_hex))
        return hmac.compare_digest(candidate, encoded)
    except (ValueError, TypeError):
        return False


def register_user(username: str, password: str) -> str:
    username = username.strip().lower()
    if len(username) < 3 or len(username) > 100:
        raise ValueError("Username must be between 3 and 100 characters.")
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")

    user_id = str(uuid.uuid4())
    try:
        _db_execute(
            "INSERT INTO users(user_id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, _hash_password(password), _utc_now().isoformat()),
        )
        _db_commit()
    except _DB_INTEGRITY_ERROR as exc:
        _db_rollback()
        raise ValueError("Unable to create account.") from exc
    return user_id


def authenticate_user(username: str, password: str) -> str | None:
    row = _db_fetchone(
        "SELECT user_id, password_hash FROM users WHERE username = ?",
        (username.strip().lower(),),
    )
    if row is None or not _verify_password(password, row["password_hash"]):
        return None

    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    _db_execute(
        "INSERT INTO sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (
            token_hash,
            row["user_id"],
            (_utc_now() + AUTH_SESSION_TTL).isoformat(),
        ),
    )
    _db_commit()
    return raw_token


def logout_user(session_token: str) -> None:
    token_hash = hashlib.sha256(session_token.encode("utf-8")).hexdigest()
    _db_execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
    _db_commit()


def _get_authenticated_user(session_token: str) -> str | None:
    if not session_token:
        return None
    token_hash = hashlib.sha256(session_token.encode("utf-8")).hexdigest()
    row = _db_fetchone(
        """
        SELECT user_id, expires_at
        FROM sessions
        WHERE token_hash = ?
        """,
        (token_hash,),
    )
    if row is None:
        return None
    try:
        if datetime.fromisoformat(row["expires_at"]) <= _utc_now():
            logout_user(session_token)
            return None
    except ValueError:
        logout_user(session_token)
        return None
    return row["user_id"]


def session_is_valid(session_token: str) -> bool:
    return _get_authenticated_user(session_token) is not None


def _require_user(session_token: str) -> str:
    user_id = _get_authenticated_user(session_token)
    if user_id is None:
        raise PermissionError("Authentication required.")
    return user_id


def _session_token_from_config(config: RunnableConfig) -> str:
    """Read authentication context without relying on checkpoint config fields."""
    configurable = config.get("configurable", {})
    session_token = configurable.get("session_token")
    if session_token:
        return str(session_token)

    metadata = config.get("metadata", {})
    session_token = metadata.get("session_token")
    return str(session_token) if session_token else ""


def create_thread(session_token: str, title: str = "New chat") -> str:
    user_id = _require_user(session_token)
    thread_id = str(uuid.uuid4())
    title = title.strip()[:120] or "New chat"
    _db_execute(
        """
        INSERT INTO thread_owners(thread_id, user_id, created_at, title)
        VALUES (?, ?, ?, ?)
        """,
        (thread_id, user_id, _utc_now().isoformat(), title),
    )
    _db_commit()
    return thread_id


def update_thread_title(session_token: str, thread_id: str, title: str) -> None:
    _authorize_thread(session_token, thread_id)
    title = title.strip()[:120] or "New chat"
    _db_execute(
        "UPDATE thread_owners SET title = ? WHERE thread_id = ?",
        (title, str(thread_id)),
    )
    _db_commit()


def _authorize_thread(session_token: str, thread_id: str) -> str:
    user_id = _require_user(session_token)
    row = _db_fetchone(
        "SELECT user_id FROM thread_owners WHERE thread_id = ?",
        (str(thread_id),),
    )
    if row is None or not hmac.compare_digest(row["user_id"], user_id):
        raise PermissionError("Conversation not found.")
    return user_id


def retrieve_user_threads(session_token: str) -> list[str]:
    user_id = _require_user(session_token)
    rows = _db_fetchall(
        """
        SELECT thread_id FROM thread_owners
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    return [row["thread_id"] for row in rows]


def retrieve_user_thread_details(session_token: str) -> list[dict]:
    user_id = _require_user(session_token)
    rows = _db_fetchall(
        """
        SELECT thread_id, title, created_at
        FROM thread_owners
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )
    return [
        {
            "thread_id": row["thread_id"],
            "title": row["title"] or "New chat",
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_conversation(session_token: str, thread_id: str) -> list[BaseMessage]:
    _authorize_thread(session_token, thread_id)
    state = chatbot.get_state(
        config={"configurable": {"thread_id": str(thread_id)}}
    )
    return state.values.get("messages", [])

# -------------------
# 1. LLM and embeddings
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
MAX_PDF_SIZE_BYTES = 5 * 1024 * 1024
_RAG_STORE_DIR = Path(
    os.getenv(
        "RAG_STORE_DIR",
        str(Path(__file__).resolve().parent / ".rag_stores"),
    )
)
_RAG_STORE_VERSION = 2
_logger = logging.getLogger(__name__)
_RAG_STORE_WRITE_RETRIES = 5
_RAG_STORE_WRITE_LOCK = threading.RLock()


def _thread_store_path(thread_id: str) -> Path:
    """Return an application-owned path without using the raw thread ID."""
    safe_id = hashlib.sha256(str(thread_id).encode("utf-8")).hexdigest()
    return _RAG_STORE_DIR / safe_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_rag_store(temporary_store: Path, store_path: Path) -> None:
    """Replace a thread's store, tolerating short Windows file-lock races."""
    last_error = None
    with _RAG_STORE_WRITE_LOCK:
        for attempt in range(_RAG_STORE_WRITE_RETRIES):
            try:
                if store_path.exists():
                    shutil.rmtree(store_path)
                temporary_store.replace(store_path)
                return
            except PermissionError as exc:
                last_error = exc
                if attempt == _RAG_STORE_WRITE_RETRIES - 1:
                    raise
                time.sleep(0.2 * (attempt + 1))

    if last_error is not None:
        raise last_error


def _load_persisted_retriever(thread_id: str, user_id: str):
    store_path = _thread_store_path(thread_id)
    manifest_path = store_path / "manifest.json"

    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("version") != _RAG_STORE_VERSION
            or manifest.get("thread_id") != thread_id
            or manifest.get("user_id") != user_id
            or manifest.get("files")
            != ["index.faiss", "index.pkl", "metadata.json", "source.pdf"]
        ):
            raise ValueError("Invalid RAG store manifest.")

        for filename, expected_hash in manifest["sha256"].items():
            path = store_path / filename
            if not path.is_file() or _sha256(path) != expected_hash:
                raise ValueError(f"RAG store file failed integrity check: {filename}")

        metadata = json.loads(
            (store_path / "metadata.json").read_text(encoding="utf-8")
        )
        vector_store = FAISS.load_local(
            str(store_path),
            embeddings,
            # This is safe only after the application-owned manifest and hashes
            # have verified the files created by this application.
            allow_dangerous_deserialization=True,
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )
        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = metadata
        return retriever
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        RuntimeError,
        EOFError,
        ImportError,
        AttributeError,
        pickle.UnpicklingError,
    ) as exc:
        _logger.warning("Unable to load RAG store for thread %s: %s", thread_id, exc)
        return None


def _get_retriever(thread_id: Optional[str], user_id: str):
    """Fetch the retriever for a thread if available."""
    thread_key = str(thread_id) if thread_id else None
    if thread_key and thread_key in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_key]
    if thread_key:
        return _load_persisted_retriever(thread_key, user_id)
    return None


def ingest_pdf(
    file_bytes: bytes,
    session_token: str,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    if len(file_bytes) > MAX_PDF_SIZE_BYTES:
        limit_mb = MAX_PDF_SIZE_BYTES // (1024 * 1024)
        raise ValueError(f"PDF is too large. Please upload a file under {limit_mb} MB.")
    user_id = _authorize_thread(session_token, thread_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        thread_key = str(thread_id)
        metadata = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
            "user_id": user_id,
        }
        store_path = _thread_store_path(thread_key)
        _RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
        temporary_store = Path(
            tempfile.mkdtemp(prefix=f"{store_path.name}.", dir=_RAG_STORE_DIR)
        )
        try:
            vector_store.save_local(str(temporary_store))
            (temporary_store / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            (temporary_store / "source.pdf").write_bytes(file_bytes)
            files = ["index.faiss", "index.pkl", "metadata.json", "source.pdf"]
            manifest = {
                "version": _RAG_STORE_VERSION,
                "thread_id": thread_key,
                "user_id": user_id,
                "files": files,
                "sha256": {
                    name: _sha256(temporary_store / name) for name in files
                },
            }
            (temporary_store / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            _replace_rag_store(temporary_store, store_path)
        except Exception:
            shutil.rmtree(temporary_store, ignore_errors=True)
            raise

        _THREAD_RETRIEVERS[thread_key] = retriever
        _THREAD_METADATA[thread_key] = metadata

        return metadata
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# 3. Tools

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"error": "Stock service is not configured."}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": ALPHA_VANTAGE_API_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("Error Message") or data.get("Note"):
            return {"error": "Stock quote unavailable or rate limit reached."}
        return data
    except (requests.RequestException, ValueError):
        return {"error": "Unable to retrieve the stock quote."}

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the PDF uploaded to the current chat.
    The current thread ID is obtained from runtime configuration.
    """
    configurable = config.get("configurable", {})
    session_token = _session_token_from_config(config)
    thread_id = configurable.get("thread_id")

    if not thread_id or not session_token:
        return {
            "error": "Authenticated conversation context is missing.",
            "query": query,
        }

    thread_id = str(thread_id)
    try:
        user_id = _authorize_thread(session_token, thread_id)
    except PermissionError:
        return {"error": "Conversation not found.", "query": query}

    retriever = _get_retriever(thread_id, user_id)

    if retriever is None:
        return {
            "error": (
                f"No retriever found for thread {thread_id}. "
                "The PDF may need to be uploaded again."
            ),
            "query": query,
        }

    result = retriever.invoke(query)

    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(thread_id, {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# 4. State and nodes
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------

def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that may answer or request a tool call."""
    configurable = config.get("configurable", {})
    session_token = _session_token_from_config(config)
    thread_id = configurable.get("thread_id")
    try:
        user_id = _authorize_thread(session_token, str(thread_id))
    except PermissionError as exc:
        if _get_authenticated_user(session_token) is None:
            raise PermissionError("Authentication required.") from exc
        raise PermissionError("Conversation not found.") from exc

    system_message = SystemMessage(
        content=(
            " For questions about an uploaded PDF, call rag_tool with a specific "
            "search query. The tool automatically identifies the current chat's "
            "thread ID. Use the retrieved context to answer the user."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
if _USING_POSTGRES:
    _CHECKPOINT_DB = psycopg.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row,
    )
    checkpointer = PostgresSaver(_CHECKPOINT_DB)
    checkpointer.setup()
else:
    conn = sqlite3.connect(
        database=os.getenv("SQLITE_DB_PATH", "chatbot.db"),
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)


graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer = checkpointer)

# -------------------
# 8. Helpers
# -------------------

def thread_has_document(session_token: str, thread_id: str) -> bool:
    user_id = _authorize_thread(session_token, thread_id)
    return _get_retriever(str(thread_id), user_id) is not None


def thread_document_metadata(session_token: str, thread_id: str) -> dict:
    thread_key = str(thread_id)
    user_id = _authorize_thread(session_token, thread_key)
    _get_retriever(thread_key, user_id)
    return _THREAD_METADATA.get(thread_key, {})