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


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key is missing! Set it in the .env file.")

print(os.getenv("OPENAI_API_KEY"))

def format_code_response(code, language="html"):
    """
    Formats code into a PrismJS-compatible <pre><code> block.
    Automatically escapes the HTML.
    """
    return f'<pre><code class="language-{language}">{html.escape(code.strip())}</code></pre>'


def convert_code_blocks(text):
    """
    Converts:
    - ```html\n...\n``` style blocks into proper <pre><code> blocks
    - `inline` backtick content into <code>inline</code>
    Escapes all other HTML to avoid rendering issues.
    """

    # Step 1: Extract code blocks and replace with placeholders
    code_blocks = []

    def replace_block(match):
        lang = match.group(1) or "html"
        code = match.group(2)
        placeholder = f"__CODEBLOCK{len(code_blocks)}__"
        code_blocks.append(format_code_response(code, lang))
        return placeholder

    block_pattern = r"```(html|css|javascript)?\n(.*?)```"
    text_with_placeholders = re.sub(block_pattern, replace_block, text, flags=re.DOTALL)

    # Step 2: Escape all other HTML
    escaped = html.escape(text_with_placeholders)

    # Step 3: Restore actual code blocks
    for i, block in enumerate(code_blocks):
        escaped = escaped.replace(f"__CODEBLOCK{i}__", block)

    # Step 4: Convert `inline` code to <code> tags
    inline_pattern = r"`([^`\n]+)`"
    escaped = re.sub(inline_pattern, r"<code>\\1</code>", escaped)

    return escaped







# --- Load syllabus, group project assignment and course instruction ---
def load_syllabus(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Syllabus file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def load_group_projects(file_path="group_project.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The JSON file '{file_path}' was not found.")
    with open(file_path, "r") as f:
        return json.load(f)

def load_course_instruction(file_path="course_instruction.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The JSON file '{file_path}' was not found.")
    with open(file_path, "r") as f:
        return json.load(f)

course_instruction = load_course_instruction()

group_projects = load_group_projects()

syllabus = load_syllabus("syllabus.json")

conversation_context = {"last_bot_message": None}

# --- Keywords that suggest it's about the group project ---
GROUP_PROJECT_KEYWORDS = [
    "group project", "group assignment", "week 6", "lab activity", "lab assignment"
    "group tasks", "team assignment", "project tasks","week 8", "week 9","week 7", "week 1", "week 2", "week 3", "week 4", "week 5", "week 10", "week 11", "week 12", "week 13",
]

# --- Keywords that suggest it's about the course instruction ---
COURSE_INSTRUCTION_KEYWORDS = [
    "grading", "grade", "late policy", "academic integrity", "canvas", "psualert", "emergency", "attendance",
    "technical requirement", "course policy", "schedule", "lesson", "bias", "disability", "accessibility",
]

# --- System Prompt ---
system_prompt = """
You are a Web Development Tutor. You help students based on their course syllabus and group project assignments.

Here is the course syllabus:
{syllabus}

Here are the course instructions:
{course_instruction}

If a student asks about a group project, you’ll be given the correct project description as part of the input.
Always respond with an explanation and follow up with a helpful question.

If a student asks a question, don't provide the answer directly, provide an explanation and guide the student step by step.
If a student reply with something unrelated, try to keep the conversation flowing.

If a student gives a short response like "Yes", "Okay", or "Sure", acknowledge it and move forward.
"""

syllabus_str = "\n".join([f"- {topic}: {', '.join(subtopics)}" for topic, subtopics in syllabus.items()])

# Convert course instruction into a clean string (not JSON with curly braces)
def flatten_instruction_text(course_instruction):
    lines = []
    for section, content in course_instruction.items():
        lines.append(f"**{section.upper()}**")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"- {k}: {v}")
        elif isinstance(content, list):
            for item in content:
                lines.append(f"- {item}")
        else:
            lines.append(str(content))
        lines.append("")  # spacing
    return "\n".join(lines)

course_instruction_str = flatten_instruction_text(course_instruction)

formatted_prompt = system_prompt.format(
    syllabus=syllabus_str,
    course_instruction=course_instruction_str
)

# --- LangChain setup ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", formatted_prompt),
    ("human", "{question}")
])
model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
chain = chat_prompt | model | StrOutputParser()

# --- Embedding for topic filtering ---
def get_allowed_topics():
    return list(syllabus.keys()) + [item for sublist in syllabus.values() for item in sublist]

def get_embedding(text):
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

syllabus_embeddings = {topic: get_embedding(topic) for topic in get_allowed_topics()}


def is_general_response(query):
    general_responses = ["yes", "no", "maybe", "okay", "sure", "thanks", "thank you"]
    return query.lower() in general_responses

import re

def extract_week_number(text):
    match = re.search(r"week\s*(\d+)", text.lower())
    if match:
        return f"week {match.group(1)}"
    return None



def is_query_allowed(query):
    if is_general_response(query):
        return True
    if conversation_context["last_bot_message"]:
        return True

    query_embedding = get_embedding(query)
    similarities = {
        topic: np.dot(query_embedding, topic_embedding) /
               (np.linalg.norm(query_embedding) * np.linalg.norm(topic_embedding))
        for topic, topic_embedding in syllabus_embeddings.items()
    }
    best_match = max(similarities, key=similarities.get)
    return similarities[best_match] > 0.75

# --- Final Chat Function ---
def get_chat_response(user_question, session_id, email):
    question_lower = user_question.lower()

    if is_general_response(question_lower):
        response = "Got it! What would you like to learn next in web development?"
        save_chat_log(session_id, user_question, response, email)
        return response

    # Step 1: Check if the question mentions a specific week using regex
    week_tag = extract_week_number(question_lower)
    matched_project_key = None

    if week_tag:
        for key in group_projects.keys():
            if key.lower().startswith(week_tag):
                matched_project_key = key
                break

    # Step 2: If a week-based match was found
    if matched_project_key:
        full_prompt = (
            f"You asked about {matched_project_key}. Here is the project assignment:\n\n"
            f"{group_projects[matched_project_key]}\n\n"
            f"Student's question:\n{user_question}"
        )
        response = chain.invoke({"question": full_prompt})
        conversation_context["last_bot_message"] = response

    # Step 3: If it's a general group project question
    elif any(keyword in question_lower for keyword in GROUP_PROJECT_KEYWORDS):
        all_projects = "\n\n".join([f"{k}:\n{v}" for k, v in group_projects.items()])
        full_prompt = f"Here are all group projects:\n{all_projects}\n\nStudent Question:\n{user_question}"
        response = chain.invoke({"question": full_prompt})
        conversation_context["last_bot_message"] = response
    
    # Step 3-1: If it's a course instruction question
    elif any(keyword in question_lower for keyword in COURSE_INSTRUCTION_KEYWORDS):
        full_prompt = f"Here are the course instructions:\n{json.dumps(course_instruction, indent=2)}\n\nStudent Question:\n{user_question}"
        response = chain.invoke({"question": full_prompt})
        conversation_context["last_bot_message"] = response

    # Step 4: Otherwise, check syllabus topics
    elif is_query_allowed(user_question):
        response = chain.invoke({"question": user_question})
        conversation_context["last_bot_message"] = response

    else:
        response = "I'm here to help with web development topics. Could you ask something related to HTML, CSS, or frontend development?"

    response = convert_code_blocks(response)



    save_chat_log(session_id, user_question, response, email)
    return response