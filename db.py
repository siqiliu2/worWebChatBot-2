import mysql.connector
from mysql.connector import errorcode
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

def save_chat_log(session_id, user_message, bot_response, email, course):
    conn = get_connection()
    cursor = conn.cursor()
    course_value = course.lower() if course else None
    insert_query = """
        INSERT INTO chat_sessions (session_id, user_message, bot_response, email, course)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(insert_query, (session_id, user_message, bot_response, email, course_value))
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == errorcode.ER_BAD_FIELD_ERROR:
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN course VARCHAR(32) DEFAULT NULL")
            conn.commit()
            cursor.execute(insert_query, (session_id, user_message, bot_response, email, course_value))
        else:
            cursor.close()
            conn.close()
            raise
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_history(email, course=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if course:
            course_value = course.lower()
            cursor.execute("""
                SELECT user_message, bot_response, timestamp FROM chat_sessions
                WHERE email = %s AND (course = %s OR (course IS NULL AND %s = 'ist256'))
                ORDER BY id ASC
            """, (email, course_value, course_value))
        else:
            cursor.execute("""
                SELECT user_message, bot_response, timestamp FROM chat_sessions
                WHERE email = %s ORDER BY id ASC
            """, (email,))
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == errorcode.ER_BAD_FIELD_ERROR:
            cursor.execute("""
                SELECT user_message, bot_response, timestamp FROM chat_sessions
                WHERE email = %s ORDER BY id ASC
            """, (email,))
        else:
            cursor.close()
            conn.close()
            raise
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history


def has_session_history(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM chat_sessions WHERE session_id = %s LIMIT 1",
        (session_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return bool(row)

def has_email_course_history(email, course=None):
    conn = get_connection()
    cursor = conn.cursor()
    if course:
        cursor.execute(
            """
            SELECT 1 FROM chat_sessions
            WHERE email = %s AND (course = %s OR (course IS NULL AND %s = 'ist256'))
            LIMIT 1
            """,
            (email, (course or '').lower(), (course or '').lower())
        )
    else:
        cursor.execute(
            "SELECT 1 FROM chat_sessions WHERE email = %s LIMIT 1",
            (email,)
        )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return bool(row)


