from flask import Flask, render_template, request, jsonify, session
import uuid
from db import create_user_if_not_exists, save_chat_log
from chatb import get_chat_response

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key

@app.before_request
def assign_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start():
    email = request.json.get("email")
    session["email"] = email
    session_id = session.get("session_id")
    create_user_if_not_exists(email)
    print(f"\U0001F4E7 Email '{email}' associated with session '{session_id}'")
    return jsonify({"status": "started"})

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.json["msg"]
    session_id = session.get("session_id")
    email = session.get("email")
    bot_response = get_chat_response(user_message, session_id, email)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    print("\n\U0001F310 Flask Web UI running at: http://127.0.0.1:5000/")
    app.run(debug=True, use_reloader=False)