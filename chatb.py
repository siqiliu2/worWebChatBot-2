import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from db import save_chat_log

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key is missing! Set it in the .env file.")

print(os.getenv("OPENAI_API_KEY"))

# --- Load syllabus and group project assignment ---
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

group_projects = load_group_projects()

syllabus = load_syllabus("syllabus.json")

conversation_context = {"last_bot_message": None}

# --- Keywords that suggest it's about the group project ---
GROUP_PROJECT_KEYWORDS = [
    "group project", "group assignment", "week 06",
    "group tasks", "team assignment", "project tasks","week 08", "week 09","week 07"
]

# --- System Prompt ---
system_prompt = """
You are a Web Development Tutor. You help students based on their course syllabus and group project assignments.

Here is the course syllabus:
{syllabus}

If a student asks about a group project, you’ll be given the correct project description as part of the input.
Always respond with an explanation and follow up with a helpful question.

If a student asks a question, don't provide the answer directly, provide an explanation and guide the student step by step.
If a student reply with something unrelated, try to keep the conversation flowing.

If a student gives a short response like "Yes", "Okay", or "Sure", acknowledge it and move forward.
"""

syllabus_str = "\n".join([f"- {topic}: {', '.join(subtopics)}" for topic, subtopics in syllabus.items()])
formatted_prompt = system_prompt.format(syllabus=syllabus_str)

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

    # Step 4: Otherwise, check syllabus topics
    elif is_query_allowed(user_question):
        response = chain.invoke({"question": user_question})
        conversation_context["last_bot_message"] = response

    else:
        response = "I'm here to help with web development topics. Could you ask something related to HTML, CSS, or frontend development?"

    save_chat_log(session_id, user_question, response, email)
    return response