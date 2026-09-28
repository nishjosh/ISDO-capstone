"""
ISDO Lab C3 — Triage Agent
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling (ReAct loop: Reason -> Act -> Observe).

Run from the project root:
    python agents/triage_agent.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Windows consoles can choke on the arrow characters below — force UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── CONFIG ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # C:\my_project
INCIDENTS_CSV = PROJECT_ROOT / "data" / "incidents.csv"

MODEL = "claude-opus-5"
MAX_TOKENS = 1024
MAX_LOOP_TURNS = 5          # safety net so the agentic loop can never spin forever

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
    }
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

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


def handle_tool_call(tool_name, tool_input):
    """Route tool calls to their implementations."""
    if tool_name == "get_open_tickets":
        return get_open_tickets(tool_input.get("csv_path", INCIDENTS_CSV))
    elif tool_name == "classify_ticket":
        return tool_input   # the classification IS the tool input — already schema-validated
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

Be consistent and rule-based: the same ticket must always get the same classification.
After calling classify_ticket, reply with one short confirmation line only."""


def triage_ticket(ticket_number, short_description, description):
    """Run the triage agent on a single ticket. Returns the classification dict."""
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
    classification = None

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
                    classification = result
                    print(f"  → Category:    {result.get('category')}")
                    print(f"  → Priority:    {result.get('priority')}")
                    print(f"  → Assign To:   {result.get('assignment_group')}")
                    print(f"  → PII Found:   {result.get('pii_detected')}")
                    print(f"  → Reason:      {result.get('reasoning')}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})
            continue   # Observe -> back to Claude to Reason again

        # end_turn (or max_tokens / refusal etc.) — print any final text and stop
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"  Agent: {block.text.strip()}")
        if response.stop_reason != "end_turn":
            print(f"  ! Loop stopped: stop_reason={response.stop_reason}")
        break
    else:
        print(f"  ! Loop stopped after {MAX_LOOP_TURNS} turns without end_turn")

    if classification is None:
        print("  ! Agent did not call classify_ticket for this ticket")
    return classification

# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test on 5 tickets from incidents.csv (+ REQ-1002 for the PII check)
    test_tickets = [
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

    results = {}
    for number, short_desc, desc in test_tickets:
        results[number] = triage_ticket(number, short_desc, desc)

    # Summary table
    print("\n" + "=" * 55)
    print("TRIAGE SUMMARY")
    print("=" * 55)
    for number, c in results.items():
        if c:
            print(f"  {number:<12} {c['category']:<12} {c['priority']:<4} "
                  f"{c['assignment_group']:<16} PII={c['pii_detected']}")
        else:
            print(f"  {number:<12} (not classified)")

    print("\n" + "=" * 55)
    print("OPEN TICKET COUNTS BY CATEGORY")
    print("=" * 55)
    # Also demo the get_open_tickets tool directly
    counts = get_open_tickets(INCIDENTS_CSV)
    for cat, count in sorted(counts.items()):
        print(f"  {cat:<20} {count} open")