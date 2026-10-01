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

C8: when ChromaDB confidence is LOW, resolution_node asks the A2A Knowledge
Specialist (a2a/knowledge_specialist.py on port 8001) for a deeper answer.
If the specialist is not running, the ticket stays LOW and goes to HITL.

C9: PII guardrail + audit trail. triage_node redacts names, emails, IDs, phones
and usernames BEFORE any text reaches Claude (or the A2A server); every later node
uses the masked text. communication_node restores the real values only in the final
message sent to the ServiceNow mock. After the run, every audit entry is written to
logs/audit_trail.jsonl through a single AuditLogger.

Run from the project root:  python agents/orchestrator/supervisor.py
"""

import operator
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import requests
from langgraph.graph import END, START, StateGraph

A2A_URL = "http://localhost:8001"     # C8: Knowledge Specialist (uvicorn ... --port 8001)

# Project root = the first parent folder that contains "agents" and "data"
# (works whether this file is in orchestrator/ or agents/orchestrator/)
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "agents").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "agents"))

sys.path.insert(0, str(ROOT / "guardrails"))

from pii_redactor import AuditLogger, redact, restore   # noqa: E402  (C9 guardrail)
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
    # C9: PII guardrail - only the clean_* fields are ever sent to Claude
    clean_short_description: str
    clean_description: str
    pii_mapping: dict              # token -> original value, used only by restore()
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
    a2a_status: str                # C8: not_needed / completed / unavailable / error
    a2a_task_id: str               # C8: task id returned by the Knowledge Specialist
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
    # C9: mask PII once, on both fields together, so the same person gets the same token everywhere
    sep = "\n\u2016\n"
    clean_text, mapping = redact(state["short_description"] + sep + state["description"])
    clean_short, clean_desc = clean_text.split(sep, 1)
    print(f"\n\u25b6 PII GUARDRAIL \u2014 {len(mapping)} item(s) masked")
    print(f"  Sent to Claude: {clean_short} | {clean_desc}")
    pii_entry = log("PIIRedactor", "redact",
                    f"{len(mapping)} masked: {', '.join(mapping) or 'none'}")   # tokens only, never values

    print(f"\n\u25b6 TRIAGE AGENT \u2014 {state['ticket_number']}")
    result = triage_agent.triage_ticket(state["ticket_number"], clean_short, clean_desc)
    if result is None:
        # Model never called classify_ticket - fail safe rather than crash the graph.
        print("  ! Triage did not return a classification - defaulting to P3/Service-Desk.")
        result = {"category": "Software", "priority": "P3", "assignment_group": "Service-Desk",
                  "pii_detected": False, "reasoning": "Fallback: classification unavailable."}

    return {
        "clean_short_description": clean_short,
        "clean_description": clean_desc,
        "pii_mapping": mapping,
        "triage_category": result["category"],
        "triage_priority": result["priority"],
        "triage_assignment_group": result["assignment_group"],
        "pii_detected": result["pii_detected"],
        "audit_log": pii_entry + log("TriageAgent", "classify_ticket",
                          f"{result['category']} / {result['priority']} -> {result['assignment_group']}"),
    }


def call_knowledge_specialist(state: TicketState) -> dict:
    """C8: A2A call. POST /tasks -> task_id, then GET /tasks/{task_id} -> result.
    Returns {'status': 'completed', 'task_id', 'result'} or {'status': 'unavailable'|'error', 'error'}."""
    query = f"{state['clean_short_description']}. {state['clean_description']}"   # C9: masked text only
    try:
        created = requests.post(f"{A2A_URL}/tasks", timeout=90, json={
            "query": query, "ticket_number": state["ticket_number"],
            "context": f"Category: {state.get('triage_category')}, Priority: {state.get('triage_priority')}"})
        created.raise_for_status()
        task_id = created.json()["task_id"]
        print(f"  -> A2A task created: {task_id}")

        task = requests.get(f"{A2A_URL}/tasks/{task_id}", timeout=30)
        task.raise_for_status()
        return {"status": "completed", "task_id": task_id, "result": task.json()["result"]}
    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "error": f"Knowledge Specialist not running at {A2A_URL}"}
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def resolution_node(state: TicketState) -> dict:
    print(f"\n\u25b6 RESOLUTION AGENT \u2014 searching KB")

    result = resolution_agent.resolve_ticket(
        state["ticket_number"], state["clean_short_description"], state["clean_description"],   # C9: masked
        state["triage_category"], state["triage_priority"],
    )
    update = {
        "kb_article": result.get("kb_article_used", "None"),
        "resolution_text": result.get("resolution_text", ""),
        "auto_resolve": bool(result.get("auto_resolve")),
        "confidence": result.get("confidence", "LOW"),
        "a2a_status": "not_needed",
        "audit_log": log("ResolutionAgent", "search_kb",
                          f"{result.get('kb_article_used', 'None')} - {result.get('confidence')} "
                          f"({result.get('top_score', 0):.0%})"),
    }
    if update["confidence"] != "LOW":
        return update

    # C8: ChromaDB confidence is LOW -> ask the Knowledge Specialist agent via A2A
    print(f"\n\u25b6 A2A CALL \u2014 Knowledge Specialist ({A2A_URL})")
    a2a = call_knowledge_specialist(state)
    update["a2a_status"] = a2a["status"]

    if a2a["status"] == "completed":
        r = a2a["result"]
        print(f"  -> A2A result: {r.get('best_match')} | {r.get('confidence')} "
              f"({r.get('confidence_score', 0):.0%}) | escalate_to_l2: {r.get('escalate_to_l2')}")
        update.update({
            "a2a_task_id": a2a["task_id"],
            "kb_article": r.get("best_match", update["kb_article"]),
            "resolution_text": r.get("resolution", update["resolution_text"]),
            "confidence": r.get("confidence", "LOW"),
            "auto_resolve": False,     # specialist answers go to an L2 engineer, never straight to the user
        })
        update["audit_log"] = update["audit_log"] + log(
            "KnowledgeSpecialist", "a2a_task",
            f"task {a2a['task_id']}: {r.get('best_match')} - {r.get('confidence')} ({r.get('confidence_score', 0):.0%})")
    else:
        # Fallback: keep confidence LOW -> sla_node will route the ticket to HITL
        print(f"  ! A2A {a2a['status']}: {a2a['error']} \u2014 falling back to HITL")
        update["audit_log"] = update["audit_log"] + log("KnowledgeSpecialist", "a2a_task",
                                                        f"{a2a['status'].upper()} - {a2a['error']}")
    return update


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
        extra = {"unavailable": " (A2A Knowledge Specialist not running)",
                 "error": " (A2A call failed)",
                 "completed": " (A2A Knowledge Specialist also LOW)"}.get(state.get("a2a_status"), "")
        reasons.append(f"LOW KB CONFIDENCE{extra} \u2014 no clear fix found, human must decide next action")
    is_access = "Access" in (state.get("category"), state.get("triage_category"))
    access_grant = is_access and state.get("request_type") == "Access Grant"
    if access_grant:
        reasons.append(f"ACCESS GRANT \u2014 {state['clean_short_description']} requires security approval")

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
    elif state.get("a2a_status") == "completed" and not state.get("hitl_required"):
        message = (f"Dear User, regarding {ticket}: our Knowledge Specialist has prepared a detailed fix, "
                    f"and {state.get('triage_assignment_group')} will apply it and keep you updated.")
        final_status = "ASSIGNED WITH SPECIALIST FIX"
    elif approved and state.get("confidence") == "LOW":
        message = (f"Dear User, regarding {ticket}: we could not find a known fix, so an L2 specialist "
                    f"from {state.get('escalation_team')} has been assigned to investigate.")
        final_status = "ASSIGNED TO L2"
    else:
        message = (f"Dear User, regarding {ticket}: your ticket has been assigned to "
                    f"{state.get('triage_assignment_group')} and is being worked on.")
        final_status = "ASSIGNED"

    # C9: put the real names/emails back ONLY for the system of record (ServiceNow mock)
    message = restore(message, state.get("pii_mapping", {}))
    sla_agent.update_ticket(ticket, "add_note", note=message)

    print(f"  USER MESSAGE: {message.splitlines()[0][:100]}...")
    print(f"\u2705 FINAL STATUS: {final_status}")

    return {"user_message": message, "final_status": final_status,
            "audit_log": log("CommunicationAgent", "post_comment",
                             f"{final_status} - message sent to ServiceNow (PII restored)")}

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
         "description": "User John Smith (ZEN-9823) reports VPN client fails to connect after AD password was reset. "
                        "Error: authentication failed. Call +91-9876543210.",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
        # C7 Step 3: to test LOW confidence, change the short_description above to
        #   "Cisco Webex not launching on MacBook M2 after Sonoma update"
        # P1 SAP outage - sla_due picked for CRITICAL (10 of 60 min = 16.7%), same as Lab C5.
        {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
         "description": "Multiple Finance users unable to login to SAP. Error: DBCON_FAIL.",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
        # C8: low-confidence ticket (nothing about Webex in the KB) -> A2A call to the Knowledge Specialist
        {"ticket_number": "INC0001016", "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
         "description": "Webex app crashes on launch since the macOS Sonoma update. Single user, reinstall did not help.",
         "category": "Software", "priority": "P3", "sla_due": "2024-01-15 17:00:00"},
        # C7 Step 4: access grant request - always goes through HITL, whatever the priority
        {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
         "description": "Contractor Sarah Jones needs VPN access. Email: sarah.jones@client.com, username: sjones01",
         "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00",
         "request_type": "Access Grant"},
    ]

    all_results = []
    for ticket in test_tickets:
        final_state = app.invoke(ticket)
        all_results.append(final_state)

    # C9: one AuditLogger for the whole run -> logs/audit_trail.jsonl
    audit_logger = AuditLogger(str(ROOT / "logs" / "audit_trail.jsonl"))
    print(f"\n\n{'=' * 60}")
    print("AUDIT TRAIL  (written to logs/audit_trail.jsonl)")
    print("=" * 60)
    for result in all_results:
        print(f"\n--- {result['ticket_number']}  ->  FINAL STATUS: {result['final_status']} ---")
        for entry in result["audit_log"]:
            status = entry["detail"].split()[0] if entry["agent"] == "HITLGate" else "Auto"
            audit_logger.log(entry["agent"], entry["action"], result["ticket_number"],
                             tool=entry["action"], rationale=entry["detail"], approval_status=status)

    print(f"\n\n{'=' * 60}")
    print("PII CHECK  -  what Claude saw vs. the original ticket")
    print("=" * 60)
    for result in all_results:
        print(f"\n{result['ticket_number']}")
        print(f"  Original     : {result['description']}")
        print(f"  Sent to Claude: {result['clean_description']}")
        print(f"  Masked items : {', '.join(result['pii_mapping']) or 'none'}")
