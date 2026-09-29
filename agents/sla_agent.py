"""
ISDO Lab C5 - SLA & Escalation Agent
Monitors SLA deadlines, predicts breach risk, and escalates CRITICAL/BREACHED tickets.
A HITL gate pauses for human approval before any P1 escalation is executed.
Every ticket ends with a plain-English "Next step" suggestion.

Run from the project root:  python agents/sla_agent.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # safe printing of warning icons on Windows

# Load the API key from C:\my_project\.env, or C:\my_project\data\kb\.env
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "data" / "kb" / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY not found. Run:  Copy-Item data\\kb\\.env .env")
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")   # override via .env if needed
SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)                  # fixed "now" for reproducible demo output
SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
MAX_ROUNDS = 4

ESCALATION_TEAMS = {
    "Network": "L2-Network-Ops", "Application": "L2-App-Support",
    "Server": "L2-Server-Ops", "Access": "L2-Security-Ops", "Security": "L2-Security-Ops",
}

# -- Tool definitions ----------------------------------------------------------

tools = [
    {
        "name": "get_sla_status",
        "description": "Check the SLA status of a ticket. Returns minutes remaining and a breach_risk level.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string", "description": "SLA due datetime, format YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
        },
    },
    {
        "name": "update_ticket",
        "description": "Update the ticket in ServiceNow: escalate, add a work note, or change state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
                "escalation_team": {"type": "string", "description": "Team to escalate to, e.g. L2-Network-Ops"},
                "note": {"type": "string"},
                "new_state": {"type": "string"},
            },
            "required": ["ticket_number", "action"],
        },
    },
]

# -- Tool implementation --------------------------------------------------------

def get_sla_status(ticket_number, sla_due, priority):
    """breach_risk: BREACHED (past due) / CRITICAL (<20% left) / AT_RISK (<50% left) / ON_TRACK."""
    try:
        due_dt = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due format: {sla_due}"}

    minutes_remaining = int((due_dt - SIMULATED_NOW).total_seconds() / 60)
    total_minutes = SLA_MINUTES.get(priority, 480)

    if minutes_remaining < 0:
        risk, msg = "BREACHED", f"SLA BREACHED by {abs(minutes_remaining)} minutes"
    elif minutes_remaining < total_minutes * 0.2:
        risk, msg = "CRITICAL", f"Only {minutes_remaining} minutes remaining - breach imminent"
    elif minutes_remaining < total_minutes * 0.5:
        risk, msg = "AT_RISK", f"{minutes_remaining} minutes remaining - at risk"
    else:
        risk, msg = "ON_TRACK", f"{minutes_remaining} minutes remaining - on track"

    return {
        "ticket_number": ticket_number, "sla_due": sla_due, "priority": priority,
        "minutes_remaining": minutes_remaining, "breach_risk": risk, "status_message": msg,
        # only P1/P2 are escalated - P3/P4 are monitored even when breached
        "requires_escalation": risk in ("BREACHED", "CRITICAL") and priority in ("P1", "P2"),
    }


def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """Simulated ServiceNow PATCH."""
    result = {"ticket_number": ticket_number, "action": action, "success": True,
              "timestamp": SIMULATED_NOW.strftime("%Y-%m-%d %H:%M:%S")}
    if action == "escalate":
        result["message"] = f"Ticket {ticket_number} escalated to {escalation_team}"
        print(f"  [ServiceNow Mock] ESCALATED {ticket_number} -> {escalation_team}")
    elif action == "add_note":
        result["message"] = f"Work note added to {ticket_number}: {(note or '')[:50]}"
        print(f"  [ServiceNow Mock] NOTE ADDED to {ticket_number}: {(note or '')[:60]}")
    elif action == "update_state":
        result["message"] = f"Ticket {ticket_number} state changed to: {new_state}"
        print(f"  [ServiceNow Mock] STATE CHANGED {ticket_number} -> {new_state}")
    return result

# -- HITL gate ------------------------------------------------------------------

def hitl_approve(ticket_number, action, detail):
    """Pause for human approval. Returns True only if the operator types 'y'."""
    print(f"\n  {'!!! ' * 5}")
    print("  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print(f"  {'!!! ' * 5}")
    try:
        decision = input("  Approve escalation? [y/n]: ").strip().lower()
    except EOFError:                    # nobody at the keyboard (e.g. automated run) -> NO
        decision = "n"
    print(f"  Decision: {'APPROVED' if decision == 'y' else 'REJECTED'}")
    return decision == "y"

# -- Next step suggestion ---------------------------------------------------------

def next_step(final, team):
    """Plain-English next action for the service desk, decided in code (not by the model)."""
    risk, priority = final["breach_risk"], final["priority"]
    if final["escalated"]:
        return f"Escalated to {team} - L2 engineer to pick it up now and update the requester."
    if final["hitl_decision"] == "REJECTED":
        return "Escalation rejected - ticket stays with current team; operator to review again in 15 min."
    if risk in ("BREACHED", "CRITICAL"):
        return f"{priority} is not auto-escalated - notify the {team} team lead to decide."
    if risk == "AT_RISK":
        return "Recheck in 30 min; escalate if it turns CRITICAL."
    if risk == "ON_TRACK":
        return "No action needed - keep monitoring."
    return "SLA check did not complete - re-run the agent for this ticket."

# -- SLA agent --------------------------------------------------------------------

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status ONCE to check breach risk.
2. If requires_escalation is true, call update_ticket with action="escalate" and an
   escalation_team appropriate to the ticket's category (Network -> L2-Network-Ops,
   Application -> L2-App-Support, Server -> L2-Server-Ops, Access/Security -> L2-Security-Ops,
   otherwise L2-Service-Desk).
3. If breach_risk is AT_RISK, call update_ticket with action="add_note" warning that the SLA is at risk.
4. If ON_TRACK, do nothing - the ticket is being monitored only.

If an escalation is rejected by the human approver, do NOT retry it; add a work note instead.
Never call get_sla_status more than once for the same ticket.
Finish with one short summary line."""


def monitor_ticket(ticket_number, short_description, category, priority, sla_due) -> dict:
    """Run SLA monitoring for one ticket. Returns the final status (used by the C6 orchestrator).
    The HITL gate is enforced here in code for every P1 escalation, regardless of what the
    model does or doesn't ask for - the model cannot bypass it."""
    print(f"\n{'=' * 55}")
    print(f"SLA Check: {ticket_number} | {priority} | Category: {category}")
    print("=" * 55)

    team = ESCALATION_TEAMS.get(category, "L2-Service-Desk")
    messages = [{
        "role": "user",
        "content": (f"Monitor SLA for this ticket and escalate if needed:\n\nTicket: {ticket_number}\n"
                    f"Description: {short_description}\nCategory: {category}\nPriority: {priority}\n"
                    f"SLA Due: {sla_due}"),
    }]
    final = {"ticket_number": ticket_number, "priority": priority, "breach_risk": None,
             "escalated": False, "hitl_decision": None}
    sla = {}

    for _ in range(MAX_ROUNDS):
        response = client.messages.create(
            model=MODEL, max_tokens=600, output_config={"effort": "low"},
            system=SYSTEM_PROMPT, tools=tools, messages=messages,
        )

        if response.stop_reason != "tool_use":
            for block in response.content:
                if hasattr(block, "text") and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "get_sla_status":
                # Use the ticket's real SLA data, not whatever the model typed
                sla = result = get_sla_status(ticket_number, sla_due, priority)
                final["breach_risk"] = result.get("breach_risk")
                print(f"  -> Risk Level: {result.get('breach_risk')}")
                print(f"  -> Status:     {result.get('status_message')}")

            elif block.name == "update_ticket" and block.input.get("action") == "escalate":
                if not sla.get("requires_escalation"):
                    # Guardrail: P3/P4 or AT_RISK/ON_TRACK tickets are never escalated
                    result = {"success": False, "message": "Escalation not allowed for this ticket - monitor only."}
                    print("  ! Guardrail: escalation blocked")
                elif final["hitl_decision"] == "REJECTED":
                    # Guardrail: no second attempt after a human said no
                    result = {"success": False, "message": "Already rejected by human approver - do not retry."}
                elif priority == "P1" and not hitl_approve(ticket_number, "Escalate ticket", f"Escalate to {team}"):
                    # HITL gate: enforced by priority, not by a flag passed in from the caller
                    final["hitl_decision"] = "REJECTED"
                    result = {"success": False, "message": "Escalation rejected by human approver"}
                    print("  Escalation cancelled and logged.")
                else:
                    if priority == "P1":
                        final["hitl_decision"] = "APPROVED"
                    result = update_ticket(ticket_number, "escalate", escalation_team=team)
                    final["escalated"] = True

            elif block.name == "update_ticket":
                inp = block.input
                result = update_ticket(ticket_number, inp["action"], inp.get("escalation_team"),
                                       inp.get("note"), inp.get("new_state"))
            else:
                result = {"error": "Unknown tool"}

            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})

        messages.append({"role": "user", "content": tool_results})

    final["next_step"] = next_step(final, team)
    print(f"  ➜ Next step: {final['next_step']}")
    return final

# -- Run SLA monitoring -----------------------------------------------------------

if __name__ == "__main__":
    # Simulated "now" = 2024-01-15 10:30. Four tickets, one per breach_risk state.
    test_tickets = [
        # P1, 10 min remaining of 60 (<20%) -> CRITICAL -> HITL gate fires: type y
        ("INC0001002", "Cannot access ERP - SAP login failure", "Application", "P1", "2024-01-15 10:40:00"),
        # P1, already past due -> BREACHED -> HITL gate fires again: type n
        ("INC0001010", "Exchange server high CPU", "Server", "P1", "2024-01-15 09:30:00"),
        # P2, 90 min remaining of 240 (20-50%) -> AT_RISK -> warning note, no escalation
        # STEP 5: change 12:00:00 to 10:00:00 -> BREACHED -> auto-escalates (P2, no HITL)
        ("INC0001001", "VPN not connecting", "Network", "P2", "2024-01-15 12:00:00"),
        # P3, days remaining -> ON_TRACK -> monitored only
        ("INC0001003", "Laptop running slowly", "Hardware", "P3", "2024-01-17 09:00:00"),
    ]

    results = [monitor_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSUMMARY - NEXT STEPS\n{'=' * 55}")
    for r in results:
        print(f"  {r['ticket_number']:<11} {r['priority']}  {str(r['breach_risk']):<9} "
              f"HITL: {r['hitl_decision'] or '-'}")
        print(f"      ➜ {r['next_step']}")