# worWebChatBot-2/worWebChatBot-2/chatb.py

import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
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
    with open(file_path, "r") as f:
        return json.load(f)

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

# Embedding setup for topic matching
def get_allowed_topics():
    # Default IST 256 syllabus used for embeddings
    default = load_syllabus("syllabus.json")
    return list(default.keys()) + [item for subs in default.values() for item in subs]

def get_embedding(text):
    client = openai.OpenAI(api_key=openai_api_key)
    resp = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(resp.data[0].embedding)

syllabus_embeddings = {topic: get_embedding(topic) for topic in get_allowed_topics()}

def is_general_response(query):
    return query.lower() in ["yes", "no", "maybe", "okay", "sure", "thanks", "thank you"]

def extract_week_number(text):
    m = re.search(r"week\s*(\d+)", text.lower())
    return f"week {m.group(1)}" if m else None

def is_query_allowed(query):
    if is_general_response(query):
        return True
    if conversation_context["last_bot_message"]:
        return True
    q_emb = get_embedding(query)
    sims = {
        topic: np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        for topic, emb in syllabus_embeddings.items()
    }
    best = max(sims, key=sims.get)
    return sims[best] > 0.75

# Main chat responder, now accepts `course`
def get_chat_response(user_question, session_id, email, course="ist256"):
    # 1. Load correct syllabus JSON
    syllabus_file = "hcdd340_syllabus.json" if course == "hcdd340" else "syllabus.json"
    try:
        syllabus = load_syllabus(syllabus_file)
    except FileNotFoundError:
        syllabus = {}

    # 2. Prepare prompts
    syllabus_str    = "\n".join(f"- {t}: {', '.join(s)}" for t, s in syllabus.items())
    instruction_str = flatten_instruction_text(course_instruction)

    system_prompt = f"""
You are a {'HCDD 340' if course=='hcdd340' else 'Web Development Tutor for IST 256'}.
Here is the course syllabus:
{syllabus_str}

Here are the course instructions:
{instruction_str}

If a student asks about a group project, you’ll be given that assignment text.
Always explain your reasoning and ask a follow-up question.
"""

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
    elif is_query_allowed(user_question):
        response = chain.invoke({"question": user_question})
        conversation_context["last_bot_message"] = response
    else:
        response = "I'm here to help with course topics—could you refine your question?"

    # 4. Format & log
    formatted = convert_code_blocks(response)
    formatted = format_explanation_text(formatted)
    save_chat_log(session_id, user_question, formatted, email)
    return formatted
