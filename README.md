# 📌 AI Web Chatbot with Flask & OpenAI

A **Flask-based AI chatbot** that provides interactive learning support using **OpenAI's GPT models**.  
The chatbot runs **entirely in a web browser** and can be easily deployed.

---

## 🚀 Features

👉 **Interactive AI Chatbot** – Powered by OpenAI's API  
👉 **Fully Web-Based** – No need for a terminal interface  
👉 **Uses LangChain for Enhanced AI Responses**  
👉 **Bootstrap & jQuery UI** – Responsive and mobile-friendly  
👉 **Flask Backend** – Handles AI requests efficiently  
👉 **Easy Deployment** – Run locally or deploy to the cloud  

---

## 📌 Project File Structure

```
📂 **YourRepoName/**  
├── 📁 **templates/**  # HTML templates for the chatbot UI  
│   ├── 📄 **index.html**  # Main chat interface  
├── 📁 **static/**  # Static files (CSS, JS)  
│   ├── 🎨 **style.css**  # Custom chatbot styles  
├── 📝 **chatbot.py**  # AI processing logic  
├── 🖥 **app.py**  # Flask web server  
├── 📄 **requirements.txt**  # List of dependencies  
├── 🔑 **.env**  # OpenAI API key (not shared in GitHub)  
└── 📖 **README.md**  # Project documentation  
```

---

## 📌 Installation & Setup

### 👉 1️⃣ Create a Virtual Environment
To isolate dependencies, create a virtual environment:

```sh
python -m venv .venv
```

Activate it:

- **Windows:**
  ```sh
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```sh
  source .venv/bin/activate
  ```

---

### 👉 2️⃣ Install Dependencies
Install all required dependencies using:

```sh
pip install -r requirements.txt
```

---

## 📌 Setting Up OpenAI API Key

### 🔧 1️⃣ Create a `.env` file in the root of your project:
```sh
touch .env
```

### 🔐 2️⃣ Add your **OpenAI API key** to `.env`:
```sh
OPENAI_API_KEY=your-openai-api-key-here
```

---

## 🎯 Running the Chatbot Locally

### 💻 1️⃣ Start the Flask Web Server:
```sh
python app.py
```

### 🌍 2️⃣ Open in Your Browser:
```
http://127.0.0.1:5000/auth
```

### 💬 3️⃣ Start Chatting!  
Simply enter your message and get real-time responses.

---


# worWebChatBot-2
