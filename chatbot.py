import streamlit as st
from openai import OpenAI
import requests
import random
import time
import re
import base64
import json
import numpy as np
import os


client = OpenAI(api_key="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")

with open("faq.json", "r", encoding="utf-8") as f:
    faqs = json.load(f)

def get_faq_embeddings():
    questions = [str(faq["question"]) for faq in faqs]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=questions,
        encoding_format="float"
    )
    embeddings = [e.embedding for e in response.data]
    return questions, embeddings

faq_questions, faq_embeddings = get_faq_embeddings()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_faq_answer_with_embeddings(user_question):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[str(user_question)],
        encoding_format="float"
    )
    user_embedding = response.data[0].embedding
    scores = [cosine_similarity(user_embedding, faq_vec) for faq_vec in faq_embeddings]
    best_idx = np.argmax(scores)
    if scores[best_idx] > 0.80:
        return faqs[best_idx]["answer"]
    return None

GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
GREETING_RESPONSE = [
    "Hi there! 👋 How can I assist you with school stuff today?",
    "Hello! 😊 Need help with anything about school?",
    "Hey! I’m here to answer your school-related questions!",
    "Hi! Let me know if you have any questions about enrollment, tuition, or classes.",
    "Hello again! 👋 What would you like to know about the school?"
]
HOW_ARE_YOU_QUESTIONS = [
    "how are you?", "how are you", "how's your day", "how's your day?"
]
HOW_ARE_YOU_RESPONSES = [ 
    "I'm doing great, thanks for asking! 😊 How can I assist you today?",
    "All good here! Let me know how I can help with your school questions. 🎓",
    "Feeling helpful as always! 😄 Got any school-related questions?",
    "I'm just a chatbot, but I'm here to help you! Ask me anything about school.",
    "Doing fantastic! What can I help you with today?"
]

st.set_page_config(page_title="🎓 School Chatbot 🤖", layout='centered')

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                              url("data:image/jpg;base64,{encoded}");
            background-size: contain;
            background-position: center center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("1234.png")

st.markdown("""
    <style>
    .stApp {
        background-color: #1a1f1a;
        padding-top: 0rem;
    }
    header, footer {
        visibility: hidden;
    }
    .block-container {
        padding-top: 0rem;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        background-color: #2e2e2e;
    }
    .user-msg {
        text-align: right;
        background-color: #a3d9a5;
        color: #000000;
        padding: 11px;
        margin: 5px;
        border-radius: 10px;
    }
    .bot-msg {
        text-align: left;
        background-color: #e8f5e9;
        color: #000000;
        padding: 10px;
        margin: 5px;
        border-radius: 10px;
    }
    label[data-baseweb="form-control"] > div {
        color: #ccdf92 !important;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
    <h1 style='color: #8a9a5b;'>🎓 JRCC School Chatbot 🤖</h1>
    <p style='color: #ccdf92; font-size: 18px;'>Ask me about Enrollment, tuition, FAQs, or anything school-related.</p>
</div>
""", unsafe_allow_html=True)

chat_container = st.container()

if "messages" not in st.session_state:
    st.session_state.messages = []

def is_greeting(message):
    return message.strip().lower() in GREETINGS

def clean_input(text):
    return text.strip().lower()

def handle_input():
    user_msg = st.session_state.user_input.strip()
    if not user_msg:
        return

    st.session_state.messages.append({"role": "user", "content": user_msg})
    st.session_state.user_input = ""

    cleaned = clean_input(user_msg)
    faq_answer = find_faq_answer_with_embeddings(cleaned)

    non_school_keywords = [
        "how are you", "your name", "favorite", "who made you", 
        "do you love me", "are you real", "weather", "date", "time", "joke"
    ]

    if faq_answer:
        bot_reply = faq_answer
    elif is_greeting(cleaned):
        bot_reply = random.choice(GREETING_RESPONSE)
    elif cleaned in HOW_ARE_YOU_QUESTIONS:
        bot_reply = random.choice(HOW_ARE_YOU_RESPONSES)
    elif any(keyword in cleaned for keyword in non_school_keywords):
        bot_reply = "I'm only trained to answer school-related questions like enrollment, tuition, or class schedules. 😊"
    else:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful school chatbot for JRCC. "
                    "Only answer school-related questions. "
                    "If the question is not about school, politely say you're limited to school-related topics. "
                    "You understand both English and Tagalog, including Taglish (mixed). "
                    "The user may have typos, slang, or informal language. "
                    "Answer in a friendly and helpful tone about school-related topics. "
                    "JRCC means Jesus Reigns Christian College, it is a Christian school."
                )},
                {"role": "user", "content": user_msg}
            ]
        )
        bot_reply = completion.choices[0].message.content.strip()

    st.session_state.messages.append({"role": "assistant", "content": ""})

    with chat_container:
        placeholder = st.empty()
        for i in range(len(bot_reply)):
            st.session_state.messages[-1]["content"] = bot_reply[:i+1]

            with placeholder.container():
                st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                    elif msg["role"] == "assistant":
                        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            time.sleep(0.015)

st.text_input("Type your message...", key="user_input", on_change=handle_input, placeholder="Ask me about enrollment, tuition, schedules…")
