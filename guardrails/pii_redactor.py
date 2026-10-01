"""
ISDO Lab C9 — PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.
Patterns covered: names (spaCy NER + cue-word rules, so names are caught even
without spaCy), usernames / login IDs, email addresses, employee IDs,
IP addresses, and phone numbers.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)
"""

import re
import json
from datetime import datetime

# Try to import spaCy — graceful fallback if not installed
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("⚠  spaCy not available — using regex-only PII detection.")

# ── REGEX PATTERNS ────────────────────────────────────────────────────────────
# Order matters: emails first, so the name inside "john.smith@corp.com" is masked as one email.

PATTERNS = {
    "EMAIL":       r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
    "IP_ADDRESS":  r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    "EMPLOYEE_ID": r'\b(?:EMP|ZEN)-?\d{3,6}\b',
    # +91-9876543210, +91 98765 43210, 9876543210, 555-123-4567 (country code is masked too)
    "PHONE":       r'(?:\+\d{1,3}[\-\s]?)?\b\d{5}[\-\s]?\d{5}\b|\b\d{3}[\-\s]\d{3}[\-\s]\d{4}\b',
}

# Usernames / login IDs. Only the captured value (group 1) is masked, the label is kept.
USERNAME_PATTERNS = [
    r'(?i)\b(?:user\s*name|user\s*id|login(?:\s*id)?|uid|account|samaccountname)\s*[:=#]?\s*([A-Za-z][\w.\-]{2,})',
    r'\b[A-Z][A-Z0-9\-]{1,14}\\([A-Za-z][\w.\-]{1,})',               # DOMAIN\username
    r'(?i)\b(?:user|by|for|from|as)\s+([a-z]+[._][a-z]+\d*|[a-z]{3,}\d{2,})\b',   # j.smith / jsmith01
    r'\(([a-z]+[._][a-z]+\d*|[a-z]{3,}\d{2,})\)',                          # (r.sharma) / (jsmith01)
]

# Person names WITHOUT spaCy: a cue word followed by Capitalised words (handles D'Souza, Anne-Marie).
_WORD = r"(?:[A-Z]'[A-Z][a-z]+|[A-Z][a-z]+(?:[A-Z][a-z]+)?(?:['\-][A-Z]?[a-z]+)*)"   # D'Souza, McDonald, Anne-Marie
NAME_PATTERNS = [
    # strong cues: one word is enough  (Mr Shah, Dear Priya, Name: Rahul Sharma)
    rf"(?:\b(?:Mr|Mrs|Ms|Miss|Dr|Dear|Hi|Hello)\.?\s+|\b[Nn]ame\s*(?:is|:)\s*)({_WORD}(?:\s+{_WORD}){{0,2}})",
    # weaker cues: need first AND last name  (User John Smith, for Michael D'Souza, Contractor Sarah Jones)
    rf"\b(?:[Uu]ser|for|by|from|[Cc]ontractor|[Ee]mployee|[Cc]olleague|[Mm]anager|[Cc]ontact|[Rr]equester|[Cc]aller)\s+({_WORD}(?:\s+{_WORD}){{1,2}})",
]
# Capitalised words that are NOT names (systems, teams, places) - never redact these
NOT_NAMES = {"Finance", "Sales", "Marketing", "Building", "Floor", "Room", "Board", "Windows", "Outlook",
             "Teams", "Office", "Microsoft", "Cisco", "Webex", "Zoom", "Adobe", "Acrobat", "Pro", "Server",
             "Exchange", "SharePoint", "Zensar", "Network", "Service", "Desk", "Support", "Team", "Project",
             "Phoenix", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
             "Error", "Ticket", "Password", "Reset", "Access", "Request", "Laptop", "Mobile", "Mac", "MacBook"}

# ── AUDIT LOGGER ──────────────────────────────────────────────────────────────

audit_log = []

def _audit(action, detail):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "module": "PIIRedactor",
        "action": action,
        "detail": detail
    }
    audit_log.append(entry)
    return entry

# ── REDACTION FUNCTION ────────────────────────────────────────────────────────

def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1]
      - mapping: dict to restore original values later

    Example:
      clean, m = redact("User John Smith, contact john.doe@corp.com or call 9876543210")
      # clean = "User [NAME_1], contact [EMAIL_1] or call [PHONE_1]"
    """
    mapping, counters, seen = {}, {}, {}      # seen: original value -> token (same value, same token)
    clean = text

    def mask(value: str, label: str) -> str:
        value = value.strip()
        if value in seen:
            return seen[value]
        counters[label] = counters.get(label, 0) + 1
        token = f"[{label}_{counters[label]}]"
        mapping[token] = value
        seen[value] = token
        return token

    # Step 1: structured patterns (email, IP, employee ID, phone)
    for label, pattern in PATTERNS.items():
        clean = re.sub(pattern, lambda m: mask(m.group(0), label), clean, flags=re.IGNORECASE)

    # Step 2: usernames / login IDs (mask only the value, keep "username:" etc.)
    for pattern in USERNAME_PATTERNS:
        clean = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), mask(m.group(1), "USERNAME")), clean)

    # Step 3: person names - spaCy NER if the model is installed
    if SPACY_AVAILABLE:
        for ent in nlp(clean).ents:
            # skip ALL-CAPS acronyms (VPN, SLA) that spaCy sometimes tags PERSON, and our own [TOKENS]
            if ent.label_ == "PERSON" and not ent.text.isupper() and "[" not in ent.text \
                    and not set(ent.text.split()) & NOT_NAMES:
                clean = clean.replace(ent.text, mask(ent.text, "NAME"))

    # Step 4: person names - cue-word rules (always on, catches what spaCy misses / works without it)
    def name_sub(m):
        name = m.group(1)
        if set(name.split()) & NOT_NAMES:
            return m.group(0)
        return m.group(0).replace(name, mask(name, "NAME"))
    for pattern in NAME_PATTERNS:
        clean = re.sub(pattern, name_sub, clean)
    # mask any later mention of an already-found name (e.g. "...John Smith ... Smith confirmed")
    for token, value in list(mapping.items()):
        if token.startswith("[NAME_"):
            for part in [value] + [p for p in value.split() if len(p) > 2 and p not in NOT_NAMES]:
                clean = re.sub(rf"\b{re.escape(part)}\b", token, clean)

    pii_count = len(mapping)
    if pii_count > 0:
        _audit("redact", f"{pii_count} PII item(s) masked: {list(mapping.keys())}")
    else:
        _audit("redact", "No PII detected")

    return clean, mapping

def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    # longest tokens first, so [NAME_1] never breaks [NAME_10]
    for token, original in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        restored = restored.replace(token, original)
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored

def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# ── AUDIT TRAIL LOGGER ────────────────────────────────────────────────────────

class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval."""

    def __init__(self, log_file: str = "logs/audit_trail.jsonl"):
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": rationale[:200] if rationale else "",
            "approval_status": approval_status
        }
        self.entries.append(entry)

        # Append to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'='*55}")
        print(f"FULL AUDIT TRAIL ({len(self.entries)} entries)")
        print(f"{'='*55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("PII REDACTION DEMO")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "username: jsmith01 locked out. Logged in as ZENSAR\\rsharma from Finance team, Dr. Priya Nair approved.",
        "Requester Rahul Sharma (r.sharma) says Sharma cannot open Outlook in Building C.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 — VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK — 210 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user — auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: logs/demo_audit.jsonl")
