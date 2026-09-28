"""
ISDO Lab C2 — Mock Jira Service Management REST API (port 5002)

  GET  /rest/agile/1.0/board/requests   all requests (?request_type= ?priority= ?status= ?assignee=)
  GET  /rest/api/2/issue/<key>          one request, Jira-style nested "fields"
  PUT  /rest/api/2/issue/<key>          update, e.g. {"fields": {"status": {"name": "Escalated"}}}
  GET  /health                          status

Run:  python mcp_server/jira_shim.py
"""
import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "requests.csv")
FILTERS = ["request_type", "priority", "status", "assignee"]
JIRA_TO_CSV = {"issuetype": "request_type", "customfield_sla": "sla"}


def load_requests():
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            return {row["key"]: dict(row) for row in csv.DictReader(f)}
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with no requests.")
        return {}


REQUESTS = load_requests()  # in-memory store for this session


def to_jira(req):
    """Flat CSV row -> Jira issue shape with a nested 'fields' object."""
    return {"key": req["key"], "fields": {
        "summary": req.get("summary"),
        "issuetype": {"name": req.get("request_type")},
        "priority": {"name": req.get("priority")},
        "status": {"name": req.get("status")},
        "assignee": {"displayName": req.get("assignee")},
        "customfield_sla": req.get("sla"),
    }}


@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    results = list(REQUESTS.values())
    for field in FILTERS:
        value = request.args.get(field)  # Flask turns '+' into a space
        if value:
            results = [r for r in results if r.get(field, "").lower() == value.strip().lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.get("/rest/api/2/issue/<key>")
def get_request(key):
    req = REQUESTS.get(key)
    if req is None:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(req))


@app.put("/rest/api/2/issue/<key>")
def update_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"errorMessages": ["No update body provided"]}), 400
    updates = {}
    for name, value in body.get("fields", body).items():  # accept nested or flat
        if isinstance(value, dict):
            value = value.get("name") or value.get("displayName") or ""
        updates[JIRA_TO_CSV.get(name, name)] = value
    updates.pop("key", None)
    REQUESTS[key].update(updates)
    print(f"[Jira Mock] Updated {key}: {updates}")
    return jsonify(to_jira(REQUESTS[key]))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print("Jira Mock API starting on http://localhost:5002")
    print(f"Loaded {len(REQUESTS)} requests from data/requests.csv")
    app.run(port=5002, debug=False)