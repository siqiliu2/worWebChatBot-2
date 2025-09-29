# worWebChatBot-2/worWebChatBot-2/chatb.py

import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
from collections import defaultdict, deque
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from db import save_chat_log
import html
import re

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key is missing! Set it in the .env file.")

# Utility to format code blocks for HTML output
def format_code_response(code, language="html"):
    escaped_code = html.escape(code.strip())
    return f'<pre><code class="language-{language}">{escaped_code}</code></pre>'

# Convert markdown-style lists and headings into HTML
def format_explanation_text(text):
    # Numbered lists
    text = re.sub(r'(?<!\d)(\d+)\.\s+(.*)', r'<li>\2</li>', text)
    if "<li>" in text and not text.strip().startswith("<ol>"):
        text = f"<ol>{text}</ol>"
    # Bulleted lists
    text = re.sub(r'-\s+(.*)', r'<li>\1</li>', text)
    if "<li>" in text and "<ul>" not in text:
        text = f"<ul>{text}</ul>"
    # Headings
    text = re.sub(r'###\s+(.*)', r'<h3>\1</h3>', text)
    return text

# Handle fenced code blocks and inline code
def convert_code_blocks(text):
    code_blocks = []

    def replace_block(match):
        lang = match.group(1) or "html"
        code = match.group(2)
        placeholder = f"__CODEBLOCK{len(code_blocks)}__"
        code_blocks.append(format_code_response(code, lang))
        return placeholder

    # Fenced code
    block_pattern = r"```(html|css|javascript)?\n(.*?)```"
    text_with_placeholders = re.sub(block_pattern, replace_block, text, flags=re.DOTALL)
    # Escape remaining HTML
    escaped = html.escape(text_with_placeholders)
    # Inline backticks
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    # Restore fenced blocks
    for i, block in enumerate(code_blocks):
        escaped = escaped.replace(f"__CODEBLOCK{i}__", block)
    return escaped

# JSON loaders
def load_syllabus(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Syllabus file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Syllabus file is not valid JSON: {file_path}") from exc

def load_group_projects(file_path="group_project.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Group project file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def load_course_instruction(file_path="course_instruction.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Course instruction file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

# Static data
course_instruction = load_course_instruction()
group_projects   = load_group_projects()

# Maintain context for follow-up
conversation_context = {"last_bot_message": None}

# Per-session conversation memory (in-process only)
MAX_HISTORY_TURNS = 3
conversation_buffers = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS))

# Keywords for routing questions
GROUP_PROJECT_KEYWORDS = [
    "group project", "group assignment", "week 6", "lab activity", "lab assignment",
    "group tasks", "team assignment", "project tasks", "week 7", "week 8", "week 9",
    "week 1", "week 2", "week 3", "week 4", "week 5", "week 10", "week 11", "week 12", "week 13"
]

COURSE_INSTRUCTION_KEYWORDS = [
    "grading", "grade", "late policy", "academic integrity", "canvas",
    "psualert", "emergency", "attendance", "technical requirement",
    "course policy", "schedule", "lesson", "bias", "disability", "accessibility"
]

# Helper to flatten instruction JSON into text
def flatten_instruction_text(ci):
    lines = []
    for section, content in ci.items():
        lines.append(f"**{section.upper()}**")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"- {k}: {v}")
        elif isinstance(content, list):
            for item in content:
                lines.append(f"- {item}")
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines)

def flatten_syllabus_text(data):
    lines = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                next_prefix = f"{prefix} > {child_key}" if prefix else child_key
                walk(next_prefix, child_value)
        elif isinstance(value, list):
            items = ", ".join(str(item) for item in value) if value else "No details provided"
            lines.append(f"{prefix}: {items}")
        else:
            lines.append(f"{prefix}: {value}")

    for key, value in data.items():
        walk(key, value)

    return "\n".join(f"- {line}" for line in lines)

# Embedding setup for topic matching
def _extract_topics_from_syllabus(data):
    topics = []
    def walk(key, value):
        if isinstance(value, dict):
            topics.append(str(key))
            for k, v in value.items():
                walk(k, v)
        elif isinstance(value, list):
            topics.append(str(key))
            for item in value:
                if isinstance(item, (str, int, float)):
                    topics.append(str(item))
        else:
            topics.append(str(key))
            if isinstance(value, (str, int, float)):
                topics.append(str(value))
    for k, v in data.items():
        walk(k, v)
    seen, out = set(), []
    for t in topics:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def get_allowed_topics(course):
    syllabus_file = "hcdd340_syllabus.json" if course == "hcdd340" else "syllabus.json"
    try:
        data = load_syllabus(syllabus_file)
    except (FileNotFoundError, ValueError):
        data = load_syllabus("syllabus.json")
    return _extract_topics_from_syllabus(data)

STOPWORDS = {
    'the','a','an','and','or','of','in','to','for','on','at','by','with','from','is','are','be','as','that','this','it','its','into','about','your','you','we','our','their','they','i','he','she','them','his','her','was','were','will','can','could','should','would','may','might'
}

allowed_terms_cache = {}

def _tokenize(text):
    words = re.split(r"[^A-Za-z0-9_]+", str(text).lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

def get_allowed_terms(course):
    key = (course or 'ist256').lower()
    if key in allowed_terms_cache:
        return allowed_terms_cache[key]
    syllabus_file = 'hcdd340_syllabus.json' if key == 'hcdd340' else 'syllabus.json'
    try:
        data = load_syllabus(syllabus_file)
    except (FileNotFoundError, ValueError):
        data = load_syllabus('syllabus.json')
    terms = set()
    def walk(v):
        if isinstance(v, dict):
            for k, val in v.items():
                terms.update(_tokenize(k))
                walk(val)
        elif isinstance(v, list):
            for item in v:
                walk(item)
        else:
            terms.update(_tokenize(v))
    walk(data)
    allowed_terms_cache[key] = terms
    return terms

def is_general_response(query):
    return query.lower() in ["yes", "no", "maybe", "okay", "sure", "thanks", "thank you"]

def extract_week_number(text):
    m = re.search(r"week\s*(\d+)", text.lower())
    return f"week {m.group(1)}" if m else None

def is_query_allowed(query, course, session_id):
    if is_general_response(query):
        return True
    if conversation_buffers.get(session_id):
        return True
    q_terms = _tokenize(query)
    allowed = get_allowed_terms(course)
    overlap = q_terms & allowed
    return len(overlap) > 0

# Main chat responder, now accepts `course`
def get_chat_response(user_question, session_id, email, course="ist256"):
    # 1. Load correct syllabus JSON
    syllabus_file = "hcdd340_syllabus.json" if course == "hcdd340" else "syllabus.json"
    try:
        syllabus = load_syllabus(syllabus_file)
    except (FileNotFoundError, ValueError):
        syllabus = {}

    # 2. Prepare prompts and limited conversation memory
    syllabus_str = flatten_syllabus_text(syllabus) if syllabus else 'No syllabus information is available right now.'
    instruction_str = flatten_instruction_text(course_instruction)

    history_pairs = list(conversation_buffers[session_id])
    if history_pairs:
        history_str = "\n\n" + "\n\n".join([f"Student: {u}\nTutor: {b}" for (u, b) in history_pairs])
    else:
        history_str = ""

    system_prompt = f"""
You are a { 'HCDD 340 Tutor (Mobile Computing)' if course=='hcdd340' else 'Web Development Tutor for IST 256' }.
STRICT SCOPE: Answer ONLY questions covered in this course's syllabus or foundational basics enabling those syllabus topics. If a question is outside scope, politely decline and suggest a related in-scope topic.

Here is the course syllabus (for grounding):
{syllabus_str}

Here are course policies/instructions:
{instruction_str}

Use short, clear explanations, include examples when helpful, and end with a brief follow-up question.
""".strip()

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    chain = chat_prompt | model | StrOutputParser()

    q_lower = user_question.lower()
    week_tag = extract_week_number(q_lower)
    matched = None
    if week_tag:
        for key in group_projects:
            if key.lower().startswith(week_tag):
                matched = key
                break

    # 3. Route the question
    if is_general_response(q_lower):
        response = "Got it! What would you like to learn next?"
    elif matched:
        prompt = f"You asked about {matched}. Here is the assignment:\n{group_projects[matched]}\n\nStudent: {user_question}"
        response = chain.invoke({"question": prompt})
        conversation_context["last_bot_message"] = response
    elif any(kw in q_lower for kw in GROUP_PROJECT_KEYWORDS):
        all_proj = "\n\n".join(f"{k}:\n{v}" for k, v in group_projects.items())
        prompt = f"Here are all group projects:\n{all_proj}\n\nStudent: {user_question}"
        response = chain.invoke({"question": prompt})
        conversation_context["last_bot_message"] = response
    elif any(kw in q_lower for kw in COURSE_INSTRUCTION_KEYWORDS):
        prompt = f"Here are the course instructions:\n{json.dumps(course_instruction, indent=2)}\n\nStudent: {user_question}"
        response = chain.invoke({"question": prompt})
        conversation_context["last_bot_message"] = response
    elif is_query_allowed(user_question, course, session_id):
        response = chain.invoke({"question": user_question})
        conversation_context["last_bot_message"] = response
    else:
        response = (
        "Iâ€™m focused on this courseâ€™s syllabus and foundational basics. "
        "That question seems outside scope. Would you like to ask about a "
        "topic from the syllabus instead (e.g., one listed above)?"
    )

    # 4. Format & log
    formatted = convert_code_blocks(response)
    formatted = format_explanation_text(formatted)
    save_chat_log(session_id, user_question, formatted, email, course)
    return formatted



