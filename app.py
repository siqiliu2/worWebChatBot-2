# worWebChatBot-2/worWebChatBot-2/app.py

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import uuid
import hashlib
import mysql.connector
from db import create_user_if_not_exists, save_chat_log, get_chat_history
from chatb import get_chat_response

app = Flask(__name__)
app.secret_key = "your-secret-key"  # Replace with a secure key

# 1. Assign UUID session if it doesn't exist
@app.before_request
def assign_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

# 2. Root route → redirect to /auth unless logged in, then /select unless course chosen
@app.route("/")
def index():
    if "email" not in session:
        return redirect(url_for("auth_page"))
    if "course" not in session:
        return redirect(url_for("select_course"))

    # Once we have both email and course, show the chat UI:
    chat_history = get_chat_history(session["email"])
    return render_template(
        "index.html",
        email=session["email"],
        history=chat_history,
        course=session["course"]
    )

# 3. Show login/register UI
@app.route("/auth")
def auth_page():
    return render_template("auth.html")

# 4. Handle login/register POST
@app.route("/auth", methods=["POST"])
def handle_auth():
    data = request.get_json()
    email = data["email"]
    password = hashlib.sha256(data["password"].encode()).hexdigest()
    action = data["action"]

    conn = mysql.connector.connect(
        host="Your DB hostname",
        user="Your DB Username",
        password="Your DB password",
        database="Your Database Name",
        port=Your DB PORT
    )
    cursor = conn.cursor()

    if action == "register":
        cursor.execute("SELECT * FROM accounts WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify(success=False, message="Account already exists.")
        cursor.execute("""
            INSERT INTO accounts (email, password_hash, agreed_to_terms)
            VALUES (%s, %s, 1)
        """, (email, password))
        conn.commit()
        session["email"] = email
        session.pop("course", None)        # clear any previous course selection
        return jsonify(success=True, message="Registered successfully.")

    elif action == "login":
        cursor.execute("""
            SELECT * FROM accounts WHERE email = %s AND password_hash = %s
        """, (email, password))
        if cursor.fetchone():
            session["email"] = email
            session.pop("course", None)      # clear any previous course selection
            return jsonify(success=True, message="Login successful.")
        else:
            return jsonify(success=False, message="Invalid email or password.")

    return jsonify(success=False, message="Invalid action.")

# 5. Logout route
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth_page"))

# 6. Optional: legacy email submission endpoint
@app.route("/start", methods=["POST"])
def start():
    email = request.json.get("email")
    session["email"] = email
    session_id = session.get("session_id")
    create_user_if_not_exists(email)
    print(f"📧 Email '{email}' associated with session '{session_id}'")
    return jsonify({"status": "started"})

# ——— New Course‐Selection Routes ———

# 7. Show course‐selection screen
@app.route("/select")
def select_course():
    if "email" not in session:
        return redirect(url_for("auth_page"))
    return render_template("select.html")

# 8. Handle course choice
@app.route("/select", methods=["POST"])
def handle_course():
    data = request.get_json()
    course = data.get("course")
    if course in ("ist256", "hcdd340"):
        session["course"] = course
        return jsonify(success=True)
    return jsonify(success=False, message="Invalid course"), 400

# ——— End Course‐Selection ———

# 9. Chat handler (now reads course too)
@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("msg")
    course       = data.get("course", session.get("course"))
    # persist in session for subsequent requests
    session["course"] = course

    session_id = session.get("session_id")
    email      = session.get("email")

    # pass course into the chat logic
    bot_response = get_chat_response(user_message, session_id, email, course)
    return jsonify({"response": bot_response})

# 10. Start server with a styled banner
if __name__ == "__main__":
    print("\033[1;31m\n🔥 Flask Web UI running at: http://127.0.0.1:5000/auth\033[0m")
    app.run(debug=True, use_reloader=False)
