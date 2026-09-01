import os
import re
import subprocess
import tempfile
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Tuple, TypedDict

import requests
import streamlit as st

# Load all credentials from a local .env file (if present) into os.environ.
# This means GMAIL_USER, MIRO_TOKEN, FIGMA_TOKEN, etc. can ALL live together
# in one file instead of being set one-by-one in the terminal each session.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning(
        "`python-dotenv` not installed — .env file won't be read automatically. "
        "Run: pip install python-dotenv"
    )

try:
    import ollama
except ImportError:
    ollama = None

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Local Multi-Agent Grid", page_icon="⚙️", layout="wide")
st.title("⚙️ Local LangGraph Multi-Agent Grid")
st.caption("Real StateGraph loop • Local Ollama model • Human-in-the-loop • Live/Simulated tool connectors")

# ─────────────────────────────────────────────────────────────
# 2. SHARED GRAPH STATE
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_query: str
    next_step: str
    tool_output: str
    review_status: str
    iterations: int
    max_iterations: int
    model: str


# ─────────────────────────────────────────────────────────────
# 3. LOCAL LLM HELPER (Ollama — fully offline, no internet needed)
# ─────────────────────────────────────────────────────────────
def call_local_llm(prompt: str, model: str, system: str | None = None) -> str:
    if ollama is None:
        return "[LLM ERROR] `ollama` package not installed. Run: pip install ollama"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM ERROR] Could not reach local Ollama server: {e}"


# ─────────────────────────────────────────────────────────────
# 4. TOOL INTEGRATIONS
# Each tool checks for real credentials in env vars.
# If missing -> runs in SIMULATED mode (safe demo, no crash).
# If present -> makes a real call.
# ─────────────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(r'[\w\.\-+]+@[\w\-]+\.[\w\.\-]+')

# Matches text inside straight quotes "..." OR curly/smart quotes “...”
QUOTED_TEXT_REGEX = re.compile(r'["“]([^"”]+)["”]')

# Signature automatically appended to every outgoing email body
EMAIL_SIGNATURE = "Best,\nHumaira Bibi"


def extract_recipient(query: str, default_to: str) -> Tuple[str, str]:
    """
    Look inside the user's message for an email address (recipient) and,
    separately, for a quoted message body.

    Priority for the BODY that actually gets emailed:
      1. If the user wrapped text in quotes ("...") — that exact quoted text
         is the body. Everything outside the quotes (e.g. "send an email to
         X and say her:") is just an instruction to the agent and is dropped.
      2. Otherwise, fall back to the whole message minus the email address
         (old behavior), so short one-line requests still work.

    Recipient is always taken from any email address found in the message,
    falling back to GMAIL_TO / GMAIL_USER if none is present.

    Every body has EMAIL_SIGNATURE appended before sending.
    """
    match = EMAIL_REGEX.search(query)
    recipient = match.group(0) if match else default_to

    quoted = QUOTED_TEXT_REGEX.search(query)
    if quoted:
        body = quoted.group(1).strip()
    elif match:
        body = query.replace(recipient, "").strip()
        body = re.sub(r'\b(to|pe|par|ko)\b\s*$', '', body, flags=re.IGNORECASE).strip()
    else:
        body = query

    body = body or "(no message body provided)"
    body = f"{body}\n\n{EMAIL_SIGNATURE}"
    return recipient, body


def gmail_tool(query: str) -> str:
    user = os.environ.get("GMAIL_USER")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    default_to = os.environ.get("GMAIL_TO", user)

    to_addr, body = extract_recipient(query, default_to or "")

    if not to_addr:
        return "[GMAIL ERROR] No recipient found — mention an email address in your message " \
               "or set GMAIL_TO as a default."

    if not (user and app_password):
        return f"[SIMULATED] Would send email to {to_addr}: '{body}' " \
               f"(set GMAIL_USER + GMAIL_APP_PASSWORD env vars to send for real via an app password)"
    try:
        msg = MIMEText(body)
        msg["Subject"] = "Message from Multi-Agent Grid"
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(user, app_password)
            server.sendmail(user, [to_addr], msg.as_string())
        return f"✅ Real email sent to {to_addr}"
    except Exception as e:
        return f"[GMAIL ERROR] {e}"


def figma_tool(query: str) -> str:
    token = os.environ.get("FIGMA_TOKEN")
    file_key = os.environ.get("FIGMA_FILE_KEY")
    if not (token and file_key):
        return f"[SIMULATED] Would fetch/update Figma file for: '{query}' " \
               f"(set FIGMA_TOKEN + FIGMA_FILE_KEY, a personal access token from Figma settings, to connect for real)"
    try:
        r = requests.get(
            f"https://api.figma.com/v1/files/{file_key}",
            headers={"X-Figma-Token": token}, timeout=10,
        )
        r.raise_for_status()
        name = r.json().get("name", "unknown")
        return f"✅ Connected to real Figma file: '{name}'"
    except Exception as e:
        return f"[FIGMA ERROR] {e}"


def miro_tool(query: str) -> str:
    token = os.environ.get("MIRO_TOKEN")
    if not token:
        return f"[SIMULATED] Would create Miro board for: '{query}' " \
               f"(set MIRO_TOKEN, from a Miro developer app, to connect for real)"
    try:
        r = requests.post(
            "https://api.miro.com/v2/boards",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": query[:60]}, timeout=10,
        )
        r.raise_for_status()
        board_id = r.json().get("id", "unknown")
        return f"✅ Real Miro board created: {board_id}"
    except Exception as e:
        return f"[MIRO ERROR] {e}"


def blender_tool(query: str) -> str:
    blender_path = os.environ.get("BLENDER_EXE")
    if not blender_path:
        return f"[SIMULATED] Would run a Blender background script for: '{query}' " \
               f"(set BLENDER_EXE to your local blender.exe path to run for real — this one is 100% local, no internet)"
    try:
        script = f"""
import bpy
bpy.ops.mesh.primitive_cube_add()
bpy.ops.wm.save_as_mainfile(filepath=r"{tempfile.gettempdir()}\\agent_output.blend")
"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(script)
            script_path = f.name
        result = subprocess.run(
            [blender_path, "--background", "--python", script_path],
            capture_output=True, text=True, timeout=60,
        )
        return f"✅ Blender ran locally (exit code {result.returncode}), saved agent_output.blend"
    except Exception as e:
        return f"[BLENDER ERROR] {e}"


def linkedin_tool(query: str) -> str:
    token = os.environ.get("LINKEDIN_TOKEN")
    author_urn = os.environ.get("LINKEDIN_URN")
    if not (token and author_urn):
        return f"[SIMULATED] Would post to LinkedIn: '{query}' " \
               f"(needs LINKEDIN_TOKEN + LINKEDIN_URN from an approved LinkedIn OAuth2 app to post for real)"
    try:
        r = requests.post(
            "https://api.linkedin.com/v2/ugcPosts",
            headers={"Authorization": f"Bearer {token}", "X-Restli-Protocol-Version": "2.0.0"},
            json={
                "author": author_urn,
                "lifecycleState": "PUBLISHED",
                "specificContent": {"com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": query},
                    "shareMediaCategory": "NONE",
                }},
                "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
            }, timeout=10,
        )
        r.raise_for_status()
        return "✅ Real LinkedIn post published"
    except Exception as e:
        return f"[LINKEDIN ERROR] {e}"


def fallback_tool(query: str, model: str) -> str:
    return call_local_llm(query, model=model, system="You are a helpful assistant. Answer directly and concisely.")


TOOL_MAP = {
    "LINKEDIN": lambda q, model: linkedin_tool(q),
    "GMAIL": lambda q, model: gmail_tool(q),
    "FIGMA": lambda q, model: figma_tool(q),
    "MIRO": lambda q, model: miro_tool(q),
    "BLENDER": lambda q, model: blender_tool(q),
    "FALLBACK": lambda q, model: fallback_tool(q, model),
}

# ─────────────────────────────────────────────────────────────
# 5. GRAPH NODES (this is the actual autonomous loop)
# ─────────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are a routing supervisor for a multi-agent system.
Given a user's request, decide which single tool should handle it.
Valid tools: LINKEDIN, GMAIL, FIGMA, MIRO, BLENDER, FALLBACK.
Reply with ONLY the tool name, nothing else, no punctuation."""

REVIEWER_SYSTEM = """You are a strict QA reviewer for an agent's tool output.
Note: outputs starting with "[SIMULATED]" or "✅" are SUCCESSFUL outputs, not errors —
they mean the tool ran correctly (either in simulated demo mode or for real). These
should be PASSED. Only reply FAILED if the output starts with an error tag like
"[GMAIL ERROR]", "[LLM ERROR]", is empty, or clearly ignores the user's request.
Reply with exactly one word: PASSED or FAILED. No punctuation, no explanation."""


def supervisor_router_node(state: AgentState) -> Dict:
    raw = call_local_llm(state["user_query"], model=state["model"], system=SUPERVISOR_SYSTEM)
    choice = raw.strip().upper()
    valid = {"LINKEDIN", "GMAIL", "FIGMA", "MIRO", "BLENDER", "FALLBACK"}
    next_step = choice if choice in valid else "FALLBACK"
    return {"next_step": next_step}


def tool_execution_node(state: AgentState) -> Dict:
    fn = TOOL_MAP.get(state["next_step"], TOOL_MAP["FALLBACK"])
    output = fn(state["user_query"], state["model"])
    iterations = state.get("iterations", 0) + 1
    return {"tool_output": output, "iterations": iterations}


def reviewer_quality_node(state: AgentState) -> Dict:
    verdict_prompt = f"User request: {state['user_query']}\nTool output: {state['tool_output']}"
    raw = call_local_llm(verdict_prompt, model=state["model"], system=REVIEWER_SYSTEM).strip().upper()
    status = "PASSED" if "PASSED" in raw else "FAILED"
    return {"review_status": status}


def route_after_review(state: AgentState) -> str:
    if state["review_status"] == "FAILED" and state["iterations"] < state.get("max_iterations", 2):
        return "retry"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_router_node)
    graph.add_node("executor", tool_execution_node)
    graph.add_node("reviewer", reviewer_quality_node)
    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges("reviewer", route_after_review, {"retry": "executor", "end": END})

    checkpointer = MemorySaver()
    # HUMAN-IN-THE-LOOP: the graph pauses right before "executor" runs.
    # Nothing (email, LinkedIn post, etc.) actually happens until the human
    # clicks "Approve" in the UI below.
    return graph.compile(checkpointer=checkpointer, interrupt_before=["executor"])


# ─────────────────────────────────────────────────────────────
# 6. SIDEBAR — model picker + connector status
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧠 Local Model (Ollama)")
    available_models = ["qwen2.5:1.5b"]
    if ollama is not None:
        try:
            available_models = [m["model"] for m in ollama.list().get("models", [])] or available_models
        except Exception:
            st.warning("Ollama server not reachable — is `ollama serve` running?")
    model_name = st.selectbox("Model", available_models)
    max_iterations = st.slider("Max self-correction retries", 1, 4, 2)

    st.write("---")
    st.header("🔌 Connector Status")

    def status(env_vars):
        return "🟢 Live" if all(os.environ.get(v) for v in env_vars) else "🟡 Simulated"

    st.write(f"Gmail: {status(['GMAIL_USER', 'GMAIL_APP_PASSWORD'])}")
    st.write(f"Figma: {status(['FIGMA_TOKEN', 'FIGMA_FILE_KEY'])}")
    st.write(f"Miro: {status(['MIRO_TOKEN'])}")
    st.write(f"Blender: {status(['BLENDER_EXE'])}")
    st.write(f"LinkedIn: {status(['LINKEDIN_TOKEN', 'LINKEDIN_URN'])}")
    st.caption("Set GMAIL_USER + GMAIL_APP_PASSWORD to go live. Just mention any recipient's "
               "email address in your chat message — it will be extracted automatically.")

# ─────────────────────────────────────────────────────────────
# 7. CHAT STATE + UI LOOP (with human-in-the-loop approval)
# ─────────────────────────────────────────────────────────────
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat-1"
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for msg in st.session_state.agent_memory:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_intent = st.chat_input(
    "e.g., 'Email ali@example.com and tell him the meeting is at 5pm'..."
)

if user_intent:
    st.session_state.agent_memory.append({"role": "user", "content": user_intent})
    with st.chat_message("user"):
        st.write(user_intent)

    initial_state: AgentState = {
        "user_query": user_intent,
        "next_step": "",
        "tool_output": "",
        "review_status": "PENDING",
        "iterations": 0,
        "max_iterations": max_iterations,
        "model": model_name,
    }

    with st.spinner(f"Supervisor ({model_name}) deciding which tool to use..."):
        for _ in st.session_state.graph.stream(initial_state, config):
            pass  # graph auto-pauses before "executor" because of interrupt_before

    snap = st.session_state.graph.get_state(config)
    st.session_state.pending_action = snap.values

# ── Show approval UI if a tool call is waiting for human sign-off ──
if st.session_state.pending_action:
    pending = st.session_state.pending_action
    with st.chat_message("assistant"):
        st.warning(
            f"🧠 **Supervisor** wants to run **{pending['next_step']}** "
            f"for request: \"{pending['user_query']}\"\n\n"
            f"Approve to actually execute this tool (this may send a real email, "
            f"post publicly, etc. if that connector is Live)."
        )
        col1, col2 = st.columns(2)
        approved = col1.button("✅ Approve & Run", key=f"approve_{len(st.session_state.agent_memory)}")
        rejected = col2.button("❌ Reject", key=f"reject_{len(st.session_state.agent_memory)}")

        if approved:
            with st.spinner("Running approved action..."):
                for event in st.session_state.graph.stream(None, config):
                    for node_name, node_output in event.items():
                        if node_name == "executor":
                            st.write(f"⚙️ **[Executor]** {node_output['tool_output']}")
                        elif node_name == "reviewer":
                            badge = "✅" if node_output["review_status"] == "PASSED" else "⚠️"
                            st.write(f"{badge} **[Reviewer]** verdict: {node_output['review_status']}")
            final = st.session_state.graph.get_state(config).values
            st.success(f"Final status: {final['review_status']}")
            st.session_state.agent_memory.append({"role": "assistant", "content": final["tool_output"]})
            st.session_state.pending_action = None
            st.rerun()

        if rejected:
            st.info("Action cancelled by human review — nothing was executed.")
            st.session_state.agent_memory.append(
                {"role": "assistant", "content": "[Cancelled by human review — no tool was executed]"}
            )
            st.session_state.pending_action = None
            st.rerun()