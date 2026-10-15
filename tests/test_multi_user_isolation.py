import os
import sys
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

os.environ.setdefault("GOOGLE_API_KEY", "test-key")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import langraph_rag_backend_dep as backend


def _login(username: str) -> str:
    backend.register_user(username, "password123")
    token = backend.authenticate_user(username, "password123")
    assert token
    return token


def test_users_only_list_and_load_their_own_threads():
    first_token = _login(f"user_{uuid.uuid4().hex}")
    second_token = _login(f"user_{uuid.uuid4().hex}")
    first_thread = backend.create_thread(first_token)

    assert first_thread in backend.retrieve_user_threads(first_token)
    assert first_thread not in backend.retrieve_user_threads(second_token)

    with pytest.raises(PermissionError):
        backend.get_conversation(second_token, first_thread)


def test_users_cannot_upload_or_read_documents_for_another_thread():
    first_token = _login(f"user_{uuid.uuid4().hex}")
    second_token = _login(f"user_{uuid.uuid4().hex}")
    first_thread = backend.create_thread(first_token)

    with pytest.raises(PermissionError):
        backend.ingest_pdf(
            b"not a PDF",
            session_token=second_token,
            thread_id=first_thread,
            filename="private.pdf",
        )

    with pytest.raises(PermissionError):
        backend.thread_document_metadata(second_token, first_thread)


def test_logout_invalidates_the_session():
    token = _login(f"user_{uuid.uuid4().hex}")
    backend.logout_user(token)

    with pytest.raises(PermissionError):
        backend.retrieve_user_threads(token)


def test_authenticated_graph_stream_uses_runtime_session_metadata(monkeypatch):
    token = _login(f"user_{uuid.uuid4().hex}")
    thread_id = backend.create_thread(token)

    class FakeModel:
        def invoke(self, messages, config=None):
            return AIMessage(content="authenticated response")

    monkeypatch.setattr(backend, "llm_with_tools", FakeModel())
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "thread_id": thread_id,
            "session_token": token,
        },
    }

    chunks = list(
        backend.chatbot.stream(
            {"messages": [HumanMessage(content="hello")]},
            config=config,
            stream_mode="updates",
        )
    )

    assert chunks


def test_graph_rejects_another_users_thread(monkeypatch):
    owner_token = _login(f"user_{uuid.uuid4().hex}")
    other_token = _login(f"user_{uuid.uuid4().hex}")
    thread_id = backend.create_thread(owner_token)

    class FakeModel:
        def invoke(self, messages, config=None):
            return AIMessage(content="should not run")

    monkeypatch.setattr(backend, "llm_with_tools", FakeModel())
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"session_token": other_token},
    }

    with pytest.raises(PermissionError, match="Conversation not found"):
        list(
            backend.chatbot.stream(
                {"messages": [HumanMessage(content="hello")]},
                config=config,
                stream_mode="updates",
            )
        )
