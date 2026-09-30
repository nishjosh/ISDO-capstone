"""
ISDO Lab C6/C7 - LangGraph Orchestrator with extended HITL gate
Wires the Triage, Resolution, SLA, HITL and Communication agents (Labs C3-C5)
into a single StateGraph.

Flow:
    triage -> resolution -> sla -> [hitl if hitl_required] -> communication

C7: the HITL gate fires for ANY of these (hitl_reason says which):
    1. P1 ticket with SLA CRITICAL/BREACHED            (from C6)
    2. Resolution Agent confidence is LOW              (any priority)
    3. category 'Access' + request_type 'Access Grant' (security-sensitive)

Run from the project root:  python agents/orchestrator/supervisor.py
"""

import operator
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

# Project root = the first parent folder that contains "agents" and "data"
# (works whether this file is in orchestrator/ or agents/orchestrator/)
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agents").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "agents"))

import resolution_agent   # noqa: E402  (path set above)
import sla_agent          # noqa: E402
import triage_agent       # noqa: E402

# -- Shared state ---------------------------------------------------------------

class TicketState(TypedDict, total=False):
    # Input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str              # C7: e.g. 'Access Grant' for REQ- tickets
    # Triage agent
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # Resolution agent
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    # SLA agent
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    hitl_reason: str               # C7: why the HITL gate fired (one or more triggers)
    access_grant: bool             # C7: True if this is an access grant request
    # HITL node
    hitl_approved: Optional[bool]
    escalation_team: Optional[str]
    # Communication agent
    user_message: str
    final_status: str
    # Every node appends here; the reducer concatenates instead of overwriting.
    audit_log: Annotated[List[dict], operator.add]


def log(agent: str, action: str, detail: str) -> list:
    """One audit_log entry, in the list form nodes return for the reducer to append."""
    return [{"timestamp": datetime.now(timezone.utc).isoformat(), "agent": agent, "action": action, "detail": detail}]

# -- Nodes ------------------------------------------------------------------------

def triage_node(state: TicketState) -> dict:
    print(f"\n{'#' * 60}")
    print(f"PROCESSING TICKET: {state['ticket_number']}")
    print(f"{'#' * 60}")
    print(f"\n\u25b6 TRIAGE AGENT \u2014 {state['ticket_number']}")

    result = triage_agent.triage_ticket(state["ticket_number"], state["short_description"], state["description"])
    if result is None:
        # Model never called classify_ticket - fail safe rather than crash the graph.
        print("  ! Triage did not return a classification - defaulting to P3/Service-Desk.")
        result = {"category": "Software", "priority": "P3", "assignment_group": "Service-Desk",
                  "pii_detected": False, "reasoning": "Fallback: classification unavailable."}

    return {
        "triage_category": result["category"],
        "triage_priority": result["priority"],
        "triage_assignment_group": result["assignment_group"],
        "pii_detected": result["pii_detected"],
        "audit_log": log("TriageAgent", "classify_ticket",
                          f"{result['category']} / {result['priority']} -> {result['assignment_group']}"),
    }


def resolution_node(state: TicketState) -> dict:
    print(f"\n\u25b6 RESOLUTION AGENT \u2014 searching KB")

    result = resolution_agent.resolve_ticket(
        state["ticket_number"], state["short_description"], state["description"],
        state["triage_category"], state["triage_priority"],
    )

    return {
        "kb_article": result.get("kb_article_used", "None"),
        "resolution_text": result.get("resolution_text", ""),
        "auto_resolve": bool(result.get("auto_resolve")),
        "confidence": result.get("confidence", "LOW"),
        "audit_log": log("ResolutionAgent", "search_kb",
                          f"{result.get('kb_article_used', 'None')} - {result.get('confidence')} "
                          f"({result.get('top_score', 0):.0%})"),
    }


def sla_node(state: TicketState) -> dict:
    print(f"\n\u25b6 SLA AGENT \u2014 checking deadline")

    priority = state["triage_priority"]
    status = sla_agent.get_sla_status(state["ticket_number"], state["sla_due"], priority)
    breach_risk = status.get("breach_risk", "ON_TRACK")
    escalation_required = bool(status.get("requires_escalation"))
    print(f"  SLA Risk: {breach_risk}  |  Minutes remaining: {status.get('minutes_remaining')}")

    # C7: check every HITL trigger and collect the reasons
    reasons = []
    if escalation_required and priority == "P1":
        reasons.append(f"P1 SLA {breach_risk} \u2014 escalation needs approval")
    if state.get("confidence") == "LOW":
        reasons.append("LOW KB CONFIDENCE \u2014 no clear fix found, human must decide next action")
    is_access = "Access" in (state.get("category"), state.get("triage_category"))
    access_grant = is_access and state.get("request_type") == "Access Grant"
    if access_grant:
        reasons.append(f"ACCESS GRANT \u2014 {state['short_description']} requires security approval")

    hitl_required = bool(reasons)
    hitl_reason = " | ".join(reasons)
    if hitl_required:
        print(f"  HITL required: {hitl_reason}")

    return {
        "sla_breach_risk": breach_risk,
        "escalation_required": escalation_required,
        "hitl_required": hitl_required,
        "hitl_reason": hitl_reason,
        "access_grant": access_grant,
        # a ticket that needs a human can never be auto-resolved
        "auto_resolve": state.get("auto_resolve", False) and not hitl_required,
        "escalation_team": sla_agent.ESCALATION_TEAMS.get(state["triage_category"], "L2-Service-Desk"),
        "audit_log": log("SLAAgent", "get_sla_status",
                         f"{breach_risk} ({status.get('minutes_remaining')} min remaining)"
                         + (f"; HITL: {hitl_reason}" if hitl_required else "")),
    }


def hitl_node(state: TicketState) -> dict:
    print(f"\n\u25b6 HITL GATE \u2014 human approval required")
    print(f"  {'WARNING ' * 8}")
    print(f"  Ticket: {state['ticket_number']} | Priority: {state['triage_priority']}")
    for reason in state["hitl_reason"].split(" | "):
        print(f"  Reason: {reason}")
    print(f"  {'WARNING ' * 8}")
    try:
        approved = input("  Approve action? [y/n]: ").strip().lower() == "y"
    except EOFError:                              # nobody at the keyboard -> NO
        approved = False
    decision = "APPROVED" if approved else "REJECTED"

    # Only a P1 SLA escalation changes the ticket in ServiceNow; other triggers just need sign-off
    if approved and state.get("escalation_required") and state["triage_priority"] == "P1":
        sla_agent.update_ticket(state["ticket_number"], "escalate", escalation_team=state["escalation_team"])

    entry = log("HITLGate", "approval_decision", f"{decision} \u2014 {state['hitl_reason']}")
    print(f"  [AUDIT] HITLGate: approval_decision \u2014 {decision}")
    print(f"  Decision: {decision}")
    return {"hitl_approved": approved, "audit_log": entry}


def communication_node(state: TicketState) -> dict:
    print(f"\n\u25b6 COMMUNICATION AGENT")

    ticket = state["ticket_number"]
    approved = state.get("hitl_approved")
    if state.get("auto_resolve"):
        message = (f"Dear User, regarding {ticket}: we found a known fix for this issue "
                    f"({state.get('kb_article')}) and applied it automatically.\n\n{state.get('resolution_text')}")
        final_status = "RESOLVED"
    elif state.get("hitl_required") and approved is False:
        # C7: rejected by the human -> 'pending approval', never a resolution
        message = (f"Dear User, regarding {ticket}: your request is pending approval by the service desk. "
                    f"We will update you once a decision has been made.")
        final_status = "PENDING APPROVAL"
    elif state.get("access_grant") and approved:
        message = (f"Dear Requester, your access grant request {ticket} has been approved by security. "
                    f"{state.get('triage_assignment_group')} will set up the access and confirm with you.")
        final_status = "ACCESS APPROVED"
    elif approved and state.get("escalation_required"):
        message = (f"Dear User, regarding {ticket}: this ticket has been escalated to "
                    f"{state.get('escalation_team')} following approval. You will be contacted shortly.")
        final_status = "ESCALATED"
    elif approved and state.get("confidence") == "LOW":
        message = (f"Dear User, regarding {ticket}: we could not find a known fix, so an L2 specialist "
                    f"from {state.get('escalation_team')} has been assigned to investigate.")
        final_status = "ASSIGNED TO L2"
    else:
        message = (f"Dear User, regarding {ticket}: your ticket has been assigned to "
                    f"{state.get('triage_assignment_group')} and is being worked on.")
        final_status = "ASSIGNED"

    print(f"  USER MESSAGE: {message.splitlines()[0][:100]}...")
    print(f"\u2705 FINAL STATUS: {final_status}")

    return {"user_message": message, "final_status": final_status,
            "audit_log": log("CommunicationAgent", "draft_message", final_status)}

# -- Conditional routing -----------------------------------------------------------

def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"

# -- Build the graph ----------------------------------------------------------------

def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("sla", sla_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("communication", communication_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "sla")
    graph.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    graph.add_edge("hitl", "communication")
    graph.add_edge("communication", END)

    return graph.compile()

# -- Run --------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    test_tickets = [
        # P2 VPN - same wording as the Lab C4 KB match, sla_due picked for a true
        # AT_RISK reading (90 of 240 min = 37.5%) - see the SLA math note in chat.
        {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
         "description": "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
        # C7 Step 3: to test LOW confidence, change the short_description above to
        #   "Cisco Webex not launching on MacBook M2 after Sonoma update"
        # P1 SAP outage - sla_due picked for CRITICAL (10 of 60 min = 16.7%), same as Lab C5.
        {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
         "description": "Multiple Finance users unable to login to SAP. Error: DBCON_FAIL.",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
        # C7 Step 4: access grant request - always goes through HITL, whatever the priority
        {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
         "description": "Contractor needs VPN access. Email: contractor@client.com",
         "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00",
         "request_type": "Access Grant"},
    ]

    all_results = []
    for ticket in test_tickets:
        final_state = app.invoke(ticket)
        all_results.append(final_state)

    print(f"\n\n{'=' * 60}")
    print("AUDIT LOG")
    print("=" * 60)
    for result in all_results:
        print(f"\n--- {result['ticket_number']} ({result['final_status']}) ---")
        for entry in result["audit_log"]:
            print(f"  [{entry['timestamp']}] {entry['agent']}: {entry['action']} \u2014 {entry['detail']}")