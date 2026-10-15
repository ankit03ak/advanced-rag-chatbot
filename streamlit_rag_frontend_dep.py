import streamlit as st
from langraph_rag_backend_dep import (
    authenticate_user,
    chatbot,
    create_thread,
    get_conversation,
    ingest_pdf,
    logout_user,
    MAX_PDF_SIZE_BYTES,
    register_user,
    retrieve_user_thread_details,
    thread_document_metadata,
    update_thread_title,
)

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def reset_chat():
    thread_id = create_thread(st.session_state["session_token"])
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state["thread_titles"][thread_id] = "New chat"
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return get_conversation(st.session_state["session_token"], thread_id)

def message_content_to_text(content):
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts).strip()

    return ""

def conversation_to_history(messages):
    history = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue

        content = message_content_to_text(msg.content)
        if content:
            history.append({"role": role, "content": content})

    return history


# ======================= Authentication ==========================
if "session_token" not in st.session_state:
    st.session_state["session_token"] = None

if not st.session_state["session_token"]:
    st.title("Multi Utility Chatbot")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            token = authenticate_user(username, password)
            if token is None:
                st.error("Invalid username or password.")
            else:
                for key in (
                    "thread_id",
                    "chat_threads",
                    "thread_titles",
                    "message_history",
                    "ingested_docs",
                ):
                    st.session_state.pop(key, None)
                st.session_state["session_token"] = token
                st.rerun()
    with register_tab:
        with st.form("register_form"):
            username = st.text_input("New username")
            password = st.text_input("New password", type="password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            try:
                register_user(username, password)
                st.success("Account created. You can now log in.")
            except ValueError as exc:
                st.error(str(exc))
    st.stop()

try:
    authenticated_thread_details = retrieve_user_thread_details(
        st.session_state["session_token"]
    )
except PermissionError:
    st.session_state.pop("session_token", None)
    st.session_state.pop("thread_id", None)
    st.session_state.pop("chat_threads", None)
    st.session_state.pop("thread_titles", None)
    st.session_state.pop("message_history", None)
    st.session_state.pop("ingested_docs", None)
    st.warning("Your session has expired. Please log in again.")
    st.stop()

# ======================= Session Initialization ===================
if 'message_history' not in st.session_state:
    st.session_state["message_history"] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = create_thread(st.session_state["session_token"])

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = [
        item["thread_id"] for item in authenticated_thread_details
    ]

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {
        item["thread_id"]: item["title"]
        for item in authenticated_thread_details
    }
st.session_state["thread_titles"].setdefault(
    st.session_state["thread_id"], "New chat"
)

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
document_metadata = thread_document_metadata(
    st.session_state["session_token"], thread_key
)
if document_metadata:
    thread_docs[document_metadata.get("filename", "document")] = document_metadata
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

if not st.session_state["message_history"]:
    st.session_state["message_history"] = conversation_to_history(
        load_conversation(st.session_state["thread_id"])
    )

# ************ SIDEBAR UI ************
st.sidebar.title("LangGraph Chatbot Configurations")
current_title = st.session_state["thread_titles"].get(thread_key, "New chat")
st.sidebar.markdown(f"**Current chat:** {current_title}")
with st.sidebar.expander("Rename current chat"):
    renamed_title = st.text_input(
        "Chat name",
        value=current_title,
        max_chars=120,
        key=f"rename-{thread_key}",
    )
    if st.button("Save name", key=f"save-rename-{thread_key}"):
        update_thread_title(
            st.session_state["session_token"], thread_key, renamed_title
        )
        st.session_state["thread_titles"][thread_key] = (
            renamed_title.strip()[:120] or "New chat"
        )
        st.rerun()
if st.sidebar.button(
    "Log out",
    type="primary",
    use_container_width=True,
):
    logout_user(st.session_state["session_token"])
    for key in (
        "session_token",
        "thread_id",
        "chat_threads",
        "thread_titles",
        "message_history",
        "ingested_docs",
    ):
        st.session_state.pop(key, None)
    st.rerun()

if st.sidebar.button(
    "New Chat",
    type="secondary",
    use_container_width=True,
):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key=f"pdf-upload-{thread_key}",
)
if uploaded_pdf:
    max_pdf_size_mb = MAX_PDF_SIZE_BYTES // (1024 * 1024)
    if uploaded_pdf.size > MAX_PDF_SIZE_BYTES:
        st.sidebar.error(
            f"PDF is too large ({uploaded_pdf.size / (1024 * 1024):.1f} MB). "
            f"Please upload a file under {max_pdf_size_mb} MB."
        )
    elif uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    session_token=st.session_state["session_token"],
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
            except ValueError as exc:
                status_box.update(label="PDF upload rejected", state="error")
                st.sidebar.error(str(exc))
            else:
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(
                    label="✅ PDF indexed", state="complete", expanded=False
                )

st.sidebar.header("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        title = st.session_state["thread_titles"].get(thread_id, "New chat")
        label = f"● {title}" if thread_id == thread_key else title
        if st.sidebar.button(
            label,
            key=f"side-thread-{thread_id}",
            type="primary" if thread_id == thread_key else "secondary",
            use_container_width=True,
        ):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    if current_title == "New chat":
        generated_title = " ".join(user_input.strip().split())[:60] or "New chat"
        update_thread_title(
            st.session_state["session_token"], thread_key, generated_title
        )
        st.session_state["thread_titles"][thread_key] = generated_title
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # will implement streaming
    # CONFIG = {'configurable' : {'thread_id': st.session_state['thread_id']}}
    CONFIG = {
        "configurable": {
            "thread_id": thread_key,
        },
        "metadata": {
            "thread_id": thread_key,
            "session_token": st.session_state["session_token"],
        },
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                node_name = metadata.get("langgraph_node")

                # Handle tool execution separately; never render tool output as chat text.
                if node_name == "tools":
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", None) or "tool"

                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` ...",
                                expanded=True,
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` ...",
                                state="running",
                                expanded=True,
                            )
                    continue

                # Only render messages produced by the assistant node.
                if node_name != "chat_node":
                    continue

                
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.tool_calls:
                        continue

                    text = message_content_to_text(message_chunk.content)
                    if text:
                        yield text

        try:
            ai_message = st.write_stream(ai_only_stream())
        except PermissionError:
            st.session_state.pop("session_token", None)
            st.session_state.pop("thread_id", None)
            st.session_state.pop("chat_threads", None)
            st.session_state.pop("thread_titles", None)
            st.session_state.pop("message_history", None)
            st.session_state.pop("ingested_docs", None)
            st.warning("Your session is no longer valid. Please log in again.")
            st.stop()

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )


    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    doc_meta = thread_document_metadata(
        st.session_state["session_token"], thread_key
    )
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["message_history"] = conversation_to_history(
        load_conversation(selected_thread)
    )
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()