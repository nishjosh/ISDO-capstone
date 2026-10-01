"""
ISDO health check — is C1 to C8 working on this machine?
Run from the project root:
    python check_labs.py          # fast checks, no Claude API calls
    python check_labs.py --api    # also sends one tiny test message to Claude
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

results = []


def check(lab, name, fn):
    try:
        detail = fn() or ""
        results.append((lab, name, "PASS", detail))
    except (Exception, SystemExit) as e:          # SystemExit: an agent stopped itself (e.g. missing key)
        results.append((lab, name, "FAIL", f"{type(e).__name__}: {e}"[:110]))


def http_ok(url):
    import requests
    try:
        r = requests.get(url, timeout=3)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"not running ({url})")
    r.raise_for_status()
    return f"{url} -> {r.status_code}"

# ── SETUP ─────────────────────────────────────────────────────────────────────

def env_key():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "data" / "kb" / ".env")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    assert key.startswith("sk-"), "ANTHROPIC_API_KEY missing — run: Copy-Item data\\kb\\.env .env"
    return "API key found (not shown)"


def packages():
    for p in ["anthropic", "chromadb", "langgraph", "fastapi", "uvicorn", "flask", "requests", "dotenv"]:
        importlib.import_module(p)
    return "anthropic, chromadb, langgraph, fastapi, uvicorn, flask, requests"

# ── C1-C8 ─────────────────────────────────────────────────────────────────────

def c1_kb():
    import chromadb
    n = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma_db")).get_collection("isdo_kb").count()
    assert n > 0, "collection is empty — run: python labs\\c1\\kb_setup.py"
    return f"isdo_kb has {n} chunks"


def c3_triage():
    m = importlib.import_module("triage_agent")
    assert callable(m.triage_ticket)
    return f"triage_ticket() ok, open tickets: {sum(m.get_open_tickets().values())}"


def c4_resolution():
    m = importlib.import_module("resolution_agent")
    assert callable(m.resolve_ticket)
    return "resolve_ticket() ok, KB loaded"


def c5_sla():
    m = importlib.import_module("sla_agent")
    cases = {("2024-01-15 10:40:00", "P1"): "CRITICAL", ("2024-01-15 09:30:00", "P1"): "BREACHED",
             ("2024-01-15 12:00:00", "P2"): "AT_RISK", ("2024-01-17 09:00:00", "P3"): "ON_TRACK"}
    for (due, pri), want in cases.items():
        got = m.get_sla_status("TEST", due, pri)["breach_risk"]
        assert got == want, f"{pri} due {due}: expected {want}, got {got}"
    return "all 4 SLA risk levels correct"


def c6_c8_graph():
    import runpy
    g = runpy.run_path(str(ROOT / "agents" / "orchestrator" / "supervisor.py"))   # does not run the tickets
    g["build_graph"]()
    fields = g["TicketState"].__annotations__
    for f in ["hitl_reason", "a2a_status"]:
        assert f in fields, f"TicketState has no '{f}' (C7/C8 changes missing)"
    return "graph compiles; C7 hitl_reason + C8 a2a_status present"


def api_ping():
    import anthropic
    r = anthropic.Anthropic().messages.create(model="claude-opus-5", max_tokens=200,
                                              messages=[{"role": "user", "content": "Reply with just: OK"}])
    return "Claude replied: " + "".join(b.text for b in r.content if b.type == "text").strip()[:30]


if __name__ == "__main__":
    check("Setup", "API key in .env", env_key)
    check("Setup", "Python packages", packages)
    check("C1", "Knowledge base (ChromaDB)", c1_kb)
    check("C2", "ServiceNow shim :5001", lambda: http_ok("http://127.0.0.1:5001/health"))
    check("C2", "Jira shim :5002", lambda: http_ok("http://127.0.0.1:5002/health"))
    check("C3", "Triage agent", c3_triage)
    check("C4", "Resolution agent", c4_resolution)
    check("C5", "SLA agent (risk levels)", c5_sla)
    check("C6-C8", "Orchestrator graph", c6_c8_graph)
    check("C8", "A2A Knowledge Specialist :8001", lambda: http_ok("http://127.0.0.1:8001/agent-card"))
    if "--api" in sys.argv:
        check("API", "Claude API call", api_ping)

    print("\n" + "=" * 78)
    print("ISDO HEALTH CHECK")
    print("=" * 78)
    for lab, name, status, detail in results:
        mark = "✅" if status == "PASS" else "❌"
        print(f" {mark} {lab:<6} {name:<32} {detail}")
    failed = [r for r in results if r[2] == "FAIL"]
    print("=" * 78)
    print(f" {len(results) - len(failed)}/{len(results)} passed"
          + ("" if not failed else "  — servers marked 'not running' just need to be started"))
