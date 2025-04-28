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

# 2. Root route → redirect to /auth unless logged in
@app.route("/")
def index():
    if "email" not in session:
        return redirect(url_for("auth_page"))

    chat_history = get_chat_history(session["email"])
    return render_template("index.html", email=session["email"], history=chat_history)

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
        host="db-mysql-nyc2-22288-do-user-20062837-0.j.db.ondigitalocean.com",
        user="doadmin",
        password="AVNS_6s5bhCmv-j-bYzP4_aJ",
        database="defaultdb",
        port=25060
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
        return jsonify(success=True, message="Registered successfully.")

    elif action == "login":
        cursor.execute("""
            SELECT * FROM accounts WHERE email = %s AND password_hash = %s
        """, (email, password))
        if cursor.fetchone():
            session["email"] = email
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

# 7. Chat handler
@app.route("/get", methods=["POST"])
def chat():
    user_message = request.json["msg"]
    session_id = session.get("session_id")
    email = session.get("email")
    bot_response = get_chat_response(user_message, session_id, email)
    return jsonify({"response": bot_response})

# 8. Start server with a styled banner
if __name__ == "__main__":
    print("\033[1;31m\n🔥 Flask Web UI running at: http://127.0.0.1:5000/auth\033[0m")
    app.run(debug=True, use_reloader=False)
