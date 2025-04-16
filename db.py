import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def create_user_if_not_exists(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT IGNORE INTO users (email) VALUES (%s)", (email,))
    conn.commit()
    cursor.close()
    conn.close()

def save_chat_log(session_id, user_message, bot_response, email):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO chat_sessions (session_id, user_message, bot_response, email)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (session_id, user_message, bot_response, email))
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_history(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_message, bot_response, timestamp FROM chat_sessions
        WHERE email = %s ORDER BY id ASC
    """, (email,))
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history

