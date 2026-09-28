"""
ISDO Lab C2 — Mock ServiceNow Table API (port 5001)

  GET   /api/now/table/incident            all incidents (?category= ?priority= ?state=)
  GET   /api/now/table/incident/<number>   one incident
  PATCH /api/now/table/incident/<number>   update fields in memory, e.g. {"state": "Escalated"}
  GET   /health                            status

Run:  python mcp_server/snow_shim.py
"""
import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "incidents.csv")
FILTERS = ["category", "priority", "state", "assignment_group"]


def load_incidents():
    """Read incidents.csv into {number: row}. Rows whose description has
    unquoted commas (too many columns) are repaired by re-joining the extra
    pieces into the description, so the CSV never needs editing."""
    incidents = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            desc = header.index("description")
            for values in reader:
                if not values:
                    continue
                extra = len(values) - len(header)
                if extra > 0:
                    values = (values[:desc] + [",".join(values[desc:desc + extra + 1])]
                              + values[desc + extra + 1:])
                row = dict(zip(header, (v.strip() for v in values)))
                incidents[row["number"]] = row
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with no incidents.")
    return incidents


INCIDENTS = load_incidents()  # in-memory store for this session


@app.get("/api/now/table/incident")
def list_incidents():
    results = list(INCIDENTS.values())
    for field in FILTERS:
        value = request.args.get(field)
        if value:
            results = [r for r in results if r.get(field, "").lower() == value.strip().lower()]
    return jsonify({"result": results, "total": len(results)})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    incident = INCIDENTS.get(number)
    if incident is None:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = request.get_json(silent=True)
    if not updates:
        return jsonify({"error": "Send a JSON body, e.g. {\"state\": \"Escalated\"}"}), 400
    updates.pop("number", None)  # the key itself can't change
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print("ServiceNow Mock API starting on http://localhost:5001")
    print(f"Loaded {len(INCIDENTS)} incidents from data/incidents.csv")
    app.run(port=5001, debug=False)