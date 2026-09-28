"""
ISDO Lab C3 — Triage Agent (+ follow-up handling extension)
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling (ReAct loop: Reason -> Act -> Observe).

Extension: when a user says the issue is NOT resolved, the agent either
  - gives troubleshooting steps from the C1 knowledge base (get_troubleshooting_steps), or
  - hands the ticket to a human agent (escalate_to_human).

Run from the project root:
    python agents/triage_agent.py               # lab tickets + follow-up tickets
    python agents/triage_agent.py --lab         # only the 5 original lab tickets
    python agents/triage_agent.py --followups   # only the follow-up tickets
"""

import csv
import json
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Windows consoles can choke on the arrow characters below — force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── CONFIG ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # C:\my_project
INCIDENTS_CSV = PROJECT_ROOT / "data" / "incidents.csv"
KB_DIR = PROJECT_ROOT / "data" / "kb"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
KB_COLLECTION = "isdo_kb"
HANDOFF_QUEUE = PROJECT_ROOT / "logs" / "handoff_queue.jsonl"   # logs/ is already in .gitignore
SNOW_SHIM_URL = "http://localhost:5001/api/now/table/incident"  # C2 ServiceNow shim

MODEL = "claude-opus-5"
MAX_TOKENS = 1500
MAX_LOOP_TURNS = 6          # safety net so the agentic loop can never spin forever
KB_MIN_CONFIDENCE = 0.35    # below this, the KB "has no good answer" -> escalate

# Load the API key: project-root .env first, then data/kb/.env (where it lives today)
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "data" / "kb" / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not found. Put it in C:\\my_project\\.env as:\n"
             "ANTHROPIC_API_KEY=sk-ant-...")

client = anthropic.Anthropic()   # picks up ANTHROPIC_API_KEY from the environment

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

tools = [
    {
        "name": "classify_ticket",
        "description": "Classify an IT support ticket. Returns category, priority, "
                       "assignment_group, and whether PII was detected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"],
                    "description": "The ticket category"
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=Critical/many users affected, P2=High/some users, "
                                   "P3=Medium/single user, P4=Low/request"
                },
                "assignment_group": {
                    "type": "string",
                    "description": "Team to assign the ticket to e.g. Network-Ops, App-Support, "
                                   "Desktop-Support, Service-Desk, Security-Ops, Server-Ops, Email-Support"
                },
                "pii_detected": {
                    "type": "boolean",
                    "description": "True if the ticket contains names, email addresses, employee IDs, or IP addresses"
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision"
                }
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"]
        }
    },
    {
        "name": "get_open_tickets",
        "description": "Get a summary count of currently open tickets by category from the incidents CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to incidents.csv file (default: data/incidents.csv)"
                }
            },
            "required": ["csv_path"]
        }
    },
    {
        "name": "get_troubleshooting_steps",
        "description": "Search the ISDO knowledge base for troubleshooting steps for an IT problem. "
                       "Returns the best-matching KB article section and a confidence score (0-1). "
                       "Use when the user asks for help fixing an issue themselves.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The problem in plain words, e.g. 'VPN authentication failed after password reset'"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the ticket over to a human service desk agent. Queues a handoff and marks "
                       "the ticket Escalated. Use when the user asks for a person, the issue is still "
                       "unresolved after previous attempts, the ticket is P1, or the KB has no good answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "reason": {
                    "type": "string",
                    "enum": ["user_requested_human", "repeat_failure", "p1_critical", "no_kb_match"],
                    "description": "Main reason for the handoff"
                },
                "assignment_group": {
                    "type": "string",
                    "description": "Team the human agent belongs to (same as classify_ticket)"
                },
                "handoff_summary": {
                    "type": "string",
                    "description": "2-3 sentence note for the human agent: problem, what was already tried. "
                                   "Do NOT include names, emails, employee IDs or IP addresses."
                }
            },
            "required": ["ticket_number", "reason", "assignment_group", "handoff_summary"]
        }
    },
]

# ── TOOL IMPLEMENTATIONS ──────────────────────────────────────────────────────

def get_open_tickets(csv_path=INCIDENTS_CSV):
    """Read incidents.csv and return {category: open_count}."""
    path = Path(csv_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path          # works no matter which folder you run from
    counts = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("state") == "Open":
                    cat = row.get("category", "Unknown")
                    counts[cat] = counts.get(cat, 0) + 1
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    return counts


def _kb_search_chroma(query):
    """Semantic search in the C1 ChromaDB collection. Raises if Chroma isn't usable."""
    import chromadb   # imported here so the lab still runs if chromadb has a problem
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = db.get_collection(KB_COLLECTION)
    res = collection.query(query_texts=[query], n_results=3)
    matches = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        matches.append({
            "article": meta.get("article"),
            "section": meta.get("section"),
            "confidence": round(1 - dist, 2),
            "text": doc[:1500],
        })
    return matches


def _kb_search_keywords(query):
    """Fallback: simple keyword overlap over data/kb/*.md, returns Resolution Steps section."""
    words = set(re.findall(r"[a-z]{3,}", query.lower()))
    best, best_score = None, 0
    for md in KB_DIR.glob("*.md"):
        text = md.read_text(encoding="utf-8")
        score = len(words & set(re.findall(r"[a-z]{3,}", text.lower())))
        if score > best_score:
            best, best_score = (md, text), score
    if not best:
        return []
    md, text = best
    m = re.search(r"(?ms)^## Resolution Steps.*?(?=^## |\Z)", text)
    section_text = m.group(0) if m else text
    confidence = round(min(1.0, best_score / max(len(words), 1)), 2)
    return [{"article": md.name, "section": "Resolution Steps",
             "confidence": confidence, "text": section_text[:1500]}]


def get_troubleshooting_steps(query):
    """Look up KB steps: ChromaDB first (Lab C1), keyword search as fallback."""
    try:
        matches = _kb_search_chroma(query)
        source = "chromadb"
    except Exception as e:
        print(f"    (ChromaDB unavailable: {type(e).__name__} — using keyword search)")
        matches = _kb_search_keywords(query)
        source = "keyword"

    if not matches or matches[0]["confidence"] < KB_MIN_CONFIDENCE:
        return {"status": "no_match", "source": source,
                "message": "No KB article matches this issue well enough. Escalate to a human agent."}
    return {"status": "found", "source": source, "matches": matches}


def escalate_to_human(ticket_number, reason, assignment_group, handoff_summary):
    """Queue a human handoff (logs/handoff_queue.jsonl) and mark the ticket Escalated in the SNOW shim."""
    handoff_id = f"HO-{uuid.uuid4().hex[:6].upper()}"
    record = {
        "handoff_id": handoff_id,
        "ticket_number": ticket_number,
        "reason": reason,
        "assignment_group": assignment_group,
        "handoff_summary": handoff_summary,
        "queued_at": datetime.now().isoformat(timespec="seconds"),
    }
    HANDOFF_QUEUE.parent.mkdir(exist_ok=True)
    with open(HANDOFF_QUEUE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Best effort: update ServiceNow shim from Lab C2 (only works if it's running)
    snow_status = "skipped"
    if ticket_number.startswith("INC"):
        try:
            import requests
            r = requests.patch(f"{SNOW_SHIM_URL}/{ticket_number}",
                               json={"state": "Escalated"}, timeout=2)
            snow_status = f"HTTP {r.status_code}"
        except Exception:
            snow_status = "shim not running"

    return {"status": "queued", "handoff_id": handoff_id,
            "assignment_group": assignment_group, "servicenow_update": snow_status,
            "expected_response": "A human agent will contact the user within the SLA for this priority."}


def handle_tool_call(tool_name, tool_input):
    """Route tool calls to their implementations."""
    if tool_name == "get_open_tickets":
        return get_open_tickets(tool_input.get("csv_path", INCIDENTS_CSV))
    if tool_name == "classify_ticket":
        return tool_input   # the classification IS the tool input — already schema-validated
    if tool_name == "get_troubleshooting_steps":
        return get_troubleshooting_steps(tool_input["query"])
    if tool_name == "escalate_to_human":
        return escalate_to_human(**tool_input)
    return {"error": f"Unknown tool: {tool_name}"}

# ── TRIAGE AGENT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

Your job is to classify incoming IT support tickets. For each ticket:
1. Use the classify_ticket tool to assign category, priority, and assignment group
2. Flag if any PII (names, emails, employee IDs, IP addresses) is present

Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)

Follow-up tickets (the user says the issue is still not resolved):
After classify_ticket, decide the next action:
- Call escalate_to_human if ANY of these is true:
    * the user asks for a human / real person / agent / to talk to someone
    * the same issue failed again after a previous fix, or this is the 2nd+ time it was reported
    * the ticket is P1
    * get_troubleshooting_steps returned no_match
- Otherwise, if the user asks for help fixing it, call get_troubleshooting_steps, then give
  the user at most 5 numbered steps based ONLY on the KB result. Skip steps the user says
  they already tried. End with: "If this doesn't fix it, reply and I'll connect you with a human agent."
- When you escalate, tell the user their handoff ID and which team will contact them.
  Never put names, emails, employee IDs or IP addresses in the handoff_summary.

For brand-new tickets that are not follow-ups, just classify and reply with one short confirmation line.
Be consistent and rule-based: the same ticket must always get the same classification."""


def triage_ticket(ticket_number, short_description, description):
    """Run the triage agent on a single ticket.
    Returns {"classification": {...}, "action": "classified|troubleshoot|human_handoff", "handoff_id": ...}."""
    print(f"\n{'=' * 55}")
    print(f"Triaging: {ticket_number}")
    print(f"{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [
        {
            "role": "user",
            "content": f"Please triage this ticket:\n\nTicket: {ticket_number}\n"
                       f"Summary: {short_description}\nDetails: {description}"
        }
    ]
    outcome = {"classification": None, "action": "classified", "handoff_id": None}

    # Agentic loop (ReAct): Reason -> Act (tool) -> Observe (tool_result) -> Reason ...
    for turn in range(MAX_LOOP_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": "low"},   # SDK 1.8 has no `temperature`; low effort + strict schema keeps it consistent
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  → Tool called: {block.name}")
                result = handle_tool_call(block.name, block.input)

                if block.name == "classify_ticket":
                    outcome["classification"] = result
                    print(f"  → Category:    {result.get('category')}")
                    print(f"  → Priority:    {result.get('priority')}")
                    print(f"  → Assign To:   {result.get('assignment_group')}")
                    print(f"  → PII Found:   {result.get('pii_detected')}")
                    print(f"  → Reason:      {result.get('reasoning')}")

                elif block.name == "get_troubleshooting_steps":
                    if outcome["action"] != "human_handoff":
                        outcome["action"] = "troubleshoot"
                    if result["status"] == "found":
                        top = result["matches"][0]
                        print(f"  → KB match:    {top['article']} / {top['section']} "
                              f"(confidence {top['confidence']:.0%}, via {result['source']})")
                    else:
                        print(f"  → KB match:    none (via {result['source']})")

                elif block.name == "escalate_to_human":
                    outcome["action"] = "human_handoff"
                    outcome["handoff_id"] = result["handoff_id"]
                    print(f"  → HANDOFF:     {result['handoff_id']} → {result['assignment_group']} "
                          f"(reason: {block.input.get('reason')}, ServiceNow: {result['servicenow_update']})")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})
            continue   # Observe -> back to Claude to Reason again

        # end_turn (or max_tokens / refusal etc.) — print the reply to the user and stop
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print("  Agent reply:")
                for line in block.text.strip().splitlines():
                    print(f"    {line}")
        if response.stop_reason != "end_turn":
            print(f"  ! Loop stopped: stop_reason={response.stop_reason}")
        break
    else:
        print(f"  ! Loop stopped after {MAX_LOOP_TURNS} turns without end_turn")

    if outcome["classification"] is None:
        print("  ! Agent did not call classify_ticket for this ticket")
    return outcome

# ── TEST TICKETS ──────────────────────────────────────────────────────────────

# Lab C3 — 5 tickets from incidents.csv (+ REQ-1002 for the PII check)
LAB_TICKETS = [
    ("INC0001001", "VPN not connecting after password change",
     "User reports VPN client fails to connect after AD password was reset. Error: authentication failed."),
    ("INC0001002", "Cannot access ERP system - login error",
     "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today."),
    ("INC0001008", "Network switch down - Building C",
     "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
    ("INC0001006", "Password reset request",
     "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
    ("REQ-1002", "VPN access for new contractor joining project Phoenix",
     "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. Email: contractor@client.com"),

    # ── STEP 5: remove the '#' from the next two lines, then re-run ──
    # ("TEST-006", "Cannot access Salesforce CRM",
    #  "User cannot access Salesforce CRM from company laptop since this morning."),

    # ── STEP 5 (variation): same issue, multiple users — does priority change? ──
    # ("TEST-007", "Cannot access Salesforce CRM",
    #  "Entire Sales team (25 users) cannot access Salesforce CRM from company laptops since this morning."),
]

# Follow-ups: issue NOT resolved -> troubleshoot or connect to a human agent
# Expected action shown in the comment on each ticket.
FOLLOWUP_TICKETS = [
    # expected: troubleshoot (asks for help, first follow-up, KB has VPN article)
    ("INC0001001", "VPN still not connecting - need help troubleshooting",
     "Follow-up: VPN still fails with 'authentication failed'. I already restarted my laptop. "
     "Can you tell me what else I can try?"),

    # expected: human_handoff (user explicitly asks for a person)
    ("INC0001004", "Email still not syncing - please connect me to a human agent",
     "Follow-up: Outlook on my iPhone has not synced for 2 days now. I removed and re-added the account "
     "as the article said, nothing changed. Please connect me with a human agent."),

    # expected: human_handoff (repeat failure + P1, many users)
    ("INC0001002", "ERP login failing AGAIN for Finance team",
     "Follow-up: SAP DBCON_FAIL is back for the whole Finance team after yesterday's fix. "
     "This is the third time this week. Month-end close is blocked."),

    # expected: troubleshoot (single user, asks for steps, KB has password article)
    ("INC0001006", "Password reset link expired - still locked out",
     "Follow-up: The reset link I got has expired and I am still locked out of my AD account. "
     "Can you walk me through the reset steps?"),

    # expected: human_handoff (user frustrated, wants to talk to someone)
    ("INC0001003", "Laptop still slow - I want to talk to someone",
     "Follow-up: The chatbot keeps giving me the same answer and my laptop still takes 10 minutes to boot. "
     "I want to speak to a real person."),

    # expected: troubleshoot OR human_handoff if no KB match (no Wi-Fi article in the KB)
    ("INC0001016", "Wi-Fi keeps dropping in Building B conference room",
     "Follow-up: Wi-Fi still drops every few minutes in the Building B conference room even after "
     "the access point was restarted. Any other steps I can try?"),
]

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--lab" in sys.argv:
        test_tickets = LAB_TICKETS
    elif "--followups" in sys.argv:
        test_tickets = FOLLOWUP_TICKETS
    else:
        test_tickets = LAB_TICKETS + FOLLOWUP_TICKETS

    results = []
    for number, short_desc, desc in test_tickets:
        results.append((number, triage_ticket(number, short_desc, desc)))

    # Summary table
    print("\n" + "=" * 72)
    print("TRIAGE SUMMARY")
    print("=" * 72)
    print(f"  {'Ticket':<12} {'Category':<12} {'Pri':<4} {'Assign To':<16} {'PII':<6} Action")
    for number, out in results:
        c = out["classification"]
        if not c:
            print(f"  {number:<12} (not classified)")
            continue
        action = out["action"] + (f" ({out['handoff_id']})" if out["handoff_id"] else "")
        print(f"  {number:<12} {c['category']:<12} {c['priority']:<4} "
              f"{c['assignment_group']:<16} {str(c['pii_detected']):<6} {action}")

    handoffs = [o for _, o in results if o["action"] == "human_handoff"]
    if handoffs:
        print(f"\n  {len(handoffs)} handoff(s) queued in {HANDOFF_QUEUE.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 55)
    print("OPEN TICKET COUNTS BY CATEGORY")
    print("=" * 55)
    # Also demo the get_open_tickets tool directly
    counts = get_open_tickets(INCIDENTS_CSV)
    for cat, count in sorted(counts.items()):
        print(f"  {cat:<20} {count} open")
