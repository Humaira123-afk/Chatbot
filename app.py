import json
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
    task_list: list
    current_task_index: int
    current_task_query: str
    all_outputs: list
    next_step: str
    tool_output: str
    review_status: str
    iterations: int
    max_iterations: int
    model: str
    miro_test_mode: bool


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


MIRO_PARSE_SYSTEM = """You extract a structured action plan from a user's request about a Miro board.
Reply with ONLY valid JSON, nothing else — no markdown fences, no explanation.

Format exactly like this:
{
  "action": "create" or "edit" or "read",
  "board_name": "<short board title, a few words — only used when action is 'create'>",
  "target_board": "<name of an EXISTING board the user is referring to — only used when action is 'edit' or 'read'>",
  "items": [
    {"type": "sticky_note", "content": "<text>"},
    {"type": "shape", "shape_type": "rectangle" or "circle" or "triangle", "content": "<text, can be empty>"},
    {"type": "text", "content": "<text>"},
    {"type": "frame", "content": "<frame title>"}
  ],
  "connectors": [
    {"from": <0-based index into items>, "to": <0-based index into items>, "label": "<optional, can be empty>"}
  ],
  "images": ["<direct image URL>", ...]
}

Rules for "action":
- "create": the user wants a brand NEW board. Use this by default if unclear.
- "edit": the user wants to ADD something (sticky notes/shapes/etc.) to an EXISTING board they name or refer to.
- "read": the user wants to SEE/LIST/CHECK what's already on an existing board, without adding anything.
- If action is "edit" or "read", set "target_board" to the existing board's name as the user wrote it.
- If action is "create", set "board_name" and leave "target_board" empty.

Rules for the rest:
- "items" is the ordered list of every sticky note / shape / text / frame the user mentioned. Preserve their order. Empty list [] if none (always empty for action "read").
- "connectors" only if the user explicitly asked to connect/link/arrow between specific items. Empty list [] otherwise.
- "images" only if the user gave an actual image URL. Empty list [] otherwise.
- Always return valid JSON matching this exact structure, nothing else."""


def parse_miro_plan(query: str, model: str) -> dict:
    """
    Ask the local LLM to read the user's natural-language request and return
    a full action plan (action type, board name/target, items, connectors,
    images) as JSON. Fully dynamic — no hardcoded patterns.
    """
    raw = call_local_llm(query, model=model, system=MIRO_PARSE_SYSTEM)
    cleaned = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
        action = str(data.get("action") or "create").strip().lower()
        if action not in ("create", "edit", "read"):
            action = "create"
        return {
            "action": action,
            "board_name": str(data.get("board_name") or "Untitled Board").strip()[:60] or "Untitled Board",
            "target_board": str(data.get("target_board") or "").strip(),
            "items": [i for i in data.get("items", []) if isinstance(i, dict) and i.get("type")],
            "connectors": [c for c in data.get("connectors", []) if isinstance(c, dict)],
            "images": [u for u in data.get("images", []) if isinstance(u, str) and u.strip()],
        }
    except Exception:
        # Model didn't return clean JSON — fall back to just a plain new board
        # instead of crashing.
        return {"action": "create", "board_name": query.strip()[:60] or "Untitled Board",
                 "target_board": "", "items": [], "connectors": [], "images": []}


def _miro_post(token: str, board_id: str, endpoint: str, payload: dict) -> str | None:
    r = requests.post(
        f"https://api.miro.com/v2/boards/{board_id}/{endpoint}",
        headers={"Authorization": f"Bearer {token}"},
        json=payload, timeout=10,
    )
    r.raise_for_status()
    return r.json().get("id")


def find_board_by_name(token: str, name: str) -> str | None:
    """Look up an existing board by (partial) name. Returns its ID, or None if not found."""
    if not name:
        return None
    r = requests.get(
        "https://api.miro.com/v2/boards",
        headers={"Authorization": f"Bearer {token}"},
        params={"query": name}, timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("data", [])
    return results[0]["id"] if results else None


def list_board_items(token: str, board_id: str) -> list:
    """Fetch every item currently on a board, as (type, short content) pairs."""
    r = requests.get(
        f"https://api.miro.com/v2/boards/{board_id}/items",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 50}, timeout=10,
    )
    r.raise_for_status()
    items = r.json().get("data", [])
    summary = []
    for it in items:
        content = (it.get("data") or {}).get("content", "") or (it.get("data") or {}).get("title", "")
        content = re.sub(r'<[^>]+>', '', content).strip()  # strip Miro's HTML formatting
        summary.append(f"{it.get('type', 'item')}: {content or '(no text)'}")
    return summary


def describe_miro_plan(plan: dict) -> str:
    """Human-readable preview of a plan — used by Test Mode so nothing real is created."""
    action = plan["action"]
    if action == "read":
        return f"🧪 TEST MODE — would look up board \"{plan['target_board']}\" and list its contents (no changes made)."
    lines = ["🧪 TEST MODE — nothing was created/changed on Miro. Here's what WOULD happen:"]
    if action == "create":
        lines.append(f"• New board: \"{plan['board_name']}\"")
    else:
        lines.append(f"• Add to EXISTING board: \"{plan['target_board']}\"")
    for i, item in enumerate(plan["items"]):
        extra = f" ({item.get('shape_type')})" if item.get("type") == "shape" else ""
        lines.append(f"• Item {i}: {item.get('type')}{extra} — \"{item.get('content', '')}\"")
    for c in plan["connectors"]:
        label = f" labeled \"{c.get('label')}\"" if c.get("label") else ""
        lines.append(f"• Connector: item {c.get('from')} → item {c.get('to')}{label}")
    for url in plan["images"]:
        lines.append(f"• Image: {url}")
    if not plan["items"] and not plan["connectors"] and not plan["images"]:
        lines.append("• (just a blank board, no items)")
    lines.append("\nTurn OFF Test Mode in the sidebar when this looks right, then run it again to do it for real.")
    return "\n".join(lines)


def add_items_to_board(token: str, board_id: str, plan: dict, x_start: float = 0) -> Tuple[int, int, int]:
    """Create every item/connector/image in `plan` on an existing board_id. Returns counts."""
    item_ids = []
    x = x_start
    for item in plan["items"]:
        item_type = item.get("type")
        content = item.get("content", "")
        new_id = None
        try:
            if item_type == "sticky_note":
                new_id = _miro_post(token, board_id, "sticky_notes", {
                    "data": {"content": content, "shape": "square"},
                    "style": {"fillColor": "light_yellow"},
                    "position": {"x": x, "y": 0},
                })
            elif item_type == "shape":
                new_id = _miro_post(token, board_id, "shapes", {
                    "data": {"shape": item.get("shape_type", "rectangle"), "content": content},
                    "position": {"x": x, "y": 250},
                })
            elif item_type == "text":
                new_id = _miro_post(token, board_id, "texts", {
                    "data": {"content": content},
                    "position": {"x": x, "y": 500},
                })
            elif item_type == "frame":
                new_id = _miro_post(token, board_id, "frames", {
                    "data": {"title": content},
                    "position": {"x": x, "y": 750},
                    "geometry": {"width": 300, "height": 300},
                })
        except Exception:
            new_id = None  # one bad item shouldn't stop the rest
        item_ids.append(new_id)
        x += 220

    connectors_made = 0
    for c in plan["connectors"]:
        frm, to = c.get("from"), c.get("to")
        if isinstance(frm, int) and isinstance(to, int) and 0 <= frm < len(item_ids) and 0 <= to < len(item_ids):
            start_id, end_id = item_ids[frm], item_ids[to]
            if start_id and end_id:
                payload = {"startItem": {"id": start_id}, "endItem": {"id": end_id}}
                if c.get("label"):
                    payload["captions"] = [{"content": c["label"]}]
                try:
                    _miro_post(token, board_id, "connectors", payload)
                    connectors_made += 1
                except Exception:
                    pass

    images_made = 0
    for url in plan["images"]:
        try:
            _miro_post(token, board_id, "images", {"data": {"url": url}, "position": {"x": x, "y": 1000}})
            x += 220
            images_made += 1
        except Exception:
            pass

    return len(plan["items"]), connectors_made, images_made


def miro_tool(query: str, model: str, test_mode: bool = False) -> str:
    plan = parse_miro_plan(query, model)

    if test_mode:
        return describe_miro_plan(plan)

    token = os.environ.get("MIRO_TOKEN")
    if not token:
        return f"[SIMULATED] Would run Miro plan: {plan} " \
               f"(set MIRO_TOKEN, from a Miro developer app, to connect for real)"

    try:
        if plan["action"] == "read":
            board_id = find_board_by_name(token, plan["target_board"])
            if not board_id:
                return f"[MIRO ERROR] Couldn't find a board named '{plan['target_board']}'"
            contents = list_board_items(token, board_id)
            if not contents:
                return f"✅ Board '{plan['target_board']}' found — it's empty."
            listing = "\n".join(f"  • {c}" for c in contents)
            return f"✅ Contents of '{plan['target_board']}':\n{listing}"

        elif plan["action"] == "edit":
            board_id = find_board_by_name(token, plan["target_board"])
            if not board_id:
                return f"[MIRO ERROR] Couldn't find a board named '{plan['target_board']}' to edit"
            n_items, n_conn, n_img = add_items_to_board(token, board_id, plan)
            summary = f"✅ Updated board '{plan['target_board']}'"
            if n_items:
                summary += f", added {n_items} item(s)"
            if n_conn:
                summary += f", {n_conn} connector(s)"
            if n_img:
                summary += f", {n_img} image(s)"
            return summary

        else:  # create
            r = requests.post(
                "https://api.miro.com/v2/boards",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": plan["board_name"]}, timeout=10,
            )
            r.raise_for_status()
            board_id = r.json().get("id", "unknown")
            n_items, n_conn, n_img = add_items_to_board(token, board_id, plan)
            summary = f"✅ Real Miro board created: {board_id}"
            if n_items:
                summary += f", {n_items} item(s)"
            if n_conn:
                summary += f", {n_conn} connector(s)"
            if n_img:
                summary += f", {n_img} image(s)"
            return summary
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
    "LINKEDIN": lambda state: linkedin_tool(state["current_task_query"]),
    "GMAIL": lambda state: gmail_tool(state["current_task_query"]),
    "FIGMA": lambda state: figma_tool(state["current_task_query"]),
    "MIRO": lambda state: miro_tool(state["current_task_query"], state["model"], state.get("miro_test_mode", False)),
    "BLENDER": lambda state: blender_tool(state["current_task_query"]),
    "FALLBACK": lambda state: fallback_tool(state["current_task_query"], state["model"]),
}

# ─────────────────────────────────────────────────────────────
# 5. GRAPH NODES (this is the actual autonomous loop)
# ─────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a task planner for a multi-agent system.
Break the user's message into an ORDERED list of separate, atomic sub-tasks.
Each sub-task should map to exactly ONE tool action (e.g. one board creation,
one board lookup, one email, one LinkedIn post, one Blender render, etc).

Reply with ONLY a JSON array of strings, nothing else — no markdown fences, no explanation.
Example:
["make a miro board named New Team with sticky notes cat, parrot, fish",
 "show me what's on the marketing ideas miro board",
 "send an email to jane@example.com saying the meeting is at 5pm"]

If the message is already a single simple task, return an array with just that one string."""

SUPERVISOR_SYSTEM = """You are a routing supervisor for a multi-agent system.
Given a single task, decide which tool should handle it.
Valid tools: LINKEDIN, GMAIL, FIGMA, MIRO, BLENDER, FALLBACK.
Reply with ONLY the tool name, nothing else, no punctuation."""

REVIEWER_SYSTEM = """You are a strict QA reviewer for an agent's tool output.
Note: outputs starting with "[SIMULATED]", "✅", or "🧪 TEST MODE" are SUCCESSFUL outputs,
not errors — they mean the tool ran correctly (real, simulated, or test-mode preview).
These should be PASSED. Only reply FAILED if the output starts with an error tag like
"[GMAIL ERROR]", "[LLM ERROR]", is empty, or clearly ignores the task.
Reply with exactly one word: PASSED or FAILED. No punctuation, no explanation."""


def planner_node(state: AgentState) -> Dict:
    """
    Split the user's raw message into an ordered list of atomic sub-tasks.
    This is what makes multi-part requests ("make a board AND email someone")
    actually work — each piece becomes its own task that loops through
    supervisor -> executor -> reviewer on its own.
    """
    raw = call_local_llm(state["user_query"], model=state["model"], system=PLANNER_SYSTEM)
    cleaned = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        tasks = json.loads(cleaned)
        tasks = [str(t).strip() for t in tasks if str(t).strip()]
    except Exception:
        tasks = []
    if not tasks:
        tasks = [state["user_query"]]  # fallback: treat the whole message as one task
    return {
        "task_list": tasks,
        "current_task_index": 0,
        "current_task_query": tasks[0],
        "all_outputs": [],
        "iterations": 0,
    }


def supervisor_router_node(state: AgentState) -> Dict:
    raw = call_local_llm(state["current_task_query"], model=state["model"], system=SUPERVISOR_SYSTEM)
    choice = raw.strip().upper()
    valid = {"LINKEDIN", "GMAIL", "FIGMA", "MIRO", "BLENDER", "FALLBACK"}
    next_step = choice if choice in valid else "FALLBACK"
    return {"next_step": next_step}


def tool_execution_node(state: AgentState) -> Dict:
    fn = TOOL_MAP.get(state["next_step"], TOOL_MAP["FALLBACK"])
    output = fn(state)
    iterations = state.get("iterations", 0) + 1
    return {"tool_output": output, "iterations": iterations}


def reviewer_quality_node(state: AgentState) -> Dict:
    verdict_prompt = f"Task: {state['current_task_query']}\nTool output: {state['tool_output']}"
    raw = call_local_llm(verdict_prompt, model=state["model"], system=REVIEWER_SYSTEM).strip().upper()
    status = "PASSED" if "PASSED" in raw else "FAILED"
    return {"review_status": status}


def advance_task_node(state: AgentState) -> Dict:
    """
    Called after a task is done (passed, or retries exhausted). Banks the
    result, moves on to the next sub-task in the list, and resets the
    retry counter for it.
    """
    all_outputs = state.get("all_outputs", []) + [
        f"[{state['current_task_query']}] → {state['tool_output']}"
    ]
    next_index = state["current_task_index"] + 1
    return {
        "all_outputs": all_outputs,
        "current_task_index": next_index,
        "current_task_query": state["task_list"][next_index],
        "iterations": 0,
        "review_status": "PENDING",
    }


def route_after_review(state: AgentState) -> str:
    if state["review_status"] == "FAILED" and state["iterations"] < state.get("max_iterations", 2):
        return "retry"
    if state["current_task_index"] + 1 < len(state["task_list"]):
        return "next_task"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("supervisor", supervisor_router_node)
    graph.add_node("executor", tool_execution_node)
    graph.add_node("reviewer", reviewer_quality_node)
    graph.add_node("advance_task", advance_task_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("supervisor", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer", route_after_review,
        {"retry": "executor", "next_task": "advance_task", "end": END},
    )
    graph.add_edge("advance_task", "supervisor")

    checkpointer = MemorySaver()
    # HUMAN-IN-THE-LOOP: the graph pauses right before EVERY "executor" run —
    # meaning before EACH sub-task's real action, not just the first one.
    # Nothing (email, board, post, etc.) actually happens until the human
    # clicks "Approve" in the UI below, for every single step.
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
    st.header("🧪 Testing")
    miro_test_mode = st.checkbox(
        "Miro Test Mode (preview only, no real API calls)", value=True
    )
    st.caption(
        "Keep this ON while you're experimenting with prompts — it shows you "
        "exactly what would be created without touching your real Miro "
        "account or using up API rate limits. Turn it OFF only when you're "
        "ready to actually create things for real."
    )

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
    "e.g., 'Make a miro board named X with sticky notes A, B and email "
    "ali@example.com saying hi'..."
)

if user_intent:
    st.session_state.agent_memory.append({"role": "user", "content": user_intent})
    with st.chat_message("user"):
        st.write(user_intent)

    initial_state: AgentState = {
        "user_query": user_intent,
        "task_list": [],
        "current_task_index": 0,
        "current_task_query": "",
        "all_outputs": [],
        "next_step": "",
        "tool_output": "",
        "review_status": "PENDING",
        "iterations": 0,
        "max_iterations": max_iterations,
        "model": model_name,
        "miro_test_mode": miro_test_mode,
    }

    with st.spinner(f"Planning tasks and deciding which tool to use ({model_name})..."):
        for _ in st.session_state.graph.stream(initial_state, config):
            pass  # graph auto-pauses before the first "executor" call

    snap = st.session_state.graph.get_state(config)
    st.session_state.pending_action = snap.values

# ── Show approval UI if a tool call is waiting for human sign-off ──
if st.session_state.pending_action:
    pending = st.session_state.pending_action
    tasks = pending.get("task_list", [pending.get("user_query", "")])
    task_num = pending.get("current_task_index", 0) + 1
    total_tasks = len(tasks)

    with st.chat_message("assistant"):
        progress_label = f" (task {task_num} of {total_tasks})" if total_tasks > 1 else ""
        st.warning(
            f"🧠 **Supervisor** wants to run **{pending['next_step']}**{progress_label}\n\n"
            f"Task: \"{pending['current_task_query']}\"\n\n"
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
                        elif node_name == "advance_task":
                            st.write("➡️ Moving on to the next task...")

            snap = st.session_state.graph.get_state(config)

            if snap.next:
                # Graph paused again — either retrying this task or starting
                # the next one in the plan. Show the approval UI again.
                st.session_state.pending_action = snap.values
            else:
                # Whole plan finished — build a combined summary of every task.
                final = snap.values
                completed = list(final.get("all_outputs", []))
                completed.append(f"[{final['current_task_query']}] → {final['tool_output']}")
                summary = "\n\n".join(completed)
                st.success(f"✅ All {len(tasks)} task(s) completed!" if total_tasks > 1 else "✅ Done!")
                st.session_state.agent_memory.append({"role": "assistant", "content": summary})
                st.session_state.pending_action = None
            st.rerun()

        if rejected:
            completed = pending.get("all_outputs", [])
            note = "[Cancelled by human review — remaining task(s) were not executed]"
            summary = "\n\n".join(completed + [note]) if completed else note
            st.info("Action cancelled by human review — nothing further was executed.")
            st.session_state.agent_memory.append({"role": "assistant", "content": summary})
            st.session_state.pending_action = None
            st.rerun()