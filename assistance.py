import streamlit as st
import os
import json
import webbrowser
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import random
import requests
import re
import pyttsx3
import speech_recognition as sr
from streamlit_option_menu import option_menu

# === Initialization ===
st.set_page_config(page_title="✨ Alice - Virtual Assistant ✨", layout="wide")
st.sidebar.markdown(f"🕒 *{datetime.now().strftime('%A, %d %B %Y %H:%M:%S')}*")

# === Data File for Reminders ===
REMINDER_FILE = "reminders.json"
if not os.path.exists(REMINDER_FILE):
    with open(REMINDER_FILE, 'w') as f:
        json.dump([], f)

# --- Text-to-Speech ---
def text_to_speech(text):
    if pyttsx3:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    else:
        st.warning("🔊 Text-to-speech module is not available.")

# --- Speech-to-Text ---
def speech_to_text():
    if not sr:
        st.warning("🎙 Speech Recognition is not available.")
        return None
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        st.success("✅ Processing...")
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("⏱️ Listening timed out. Please try again.")
        return None
    except sr.UnknownValueError:
        st.warning("😕 Could not understand your speech. Please try again.")
        return None
    except sr.RequestError:
        st.error("🔌 Could not connect to the speech recognition service.")
        return None
    except Exception as e:
        st.error(f"🎤 Speech Recognition Error: {e}")
        return None

# === Load/Save Reminders ===
def load_reminders():
    with open(REMINDER_FILE, 'r') as f:
        return json.load(f)

def save_reminder(reminder):
    data = load_reminders()
    data.append(reminder)
    with open(REMINDER_FILE, 'w') as f:
        json.dump(data, f)

# === Weather API ===
WEATHER_API_KEY = "your_openweather_api_key"
def get_weather(city="Ongole"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url).json()
    if response.get("main"):
        temp = response['main']['temp']
        desc = response['weather'][0]['description']
        return f"Current weather in {city} is {temp}°C with {desc}."
    return "Unable to fetch weather."

# === Email Sending ===
def send_email(to, subject, message):
    try:
        sender_email = "srilavanya318@gmail.com"
        sender_password = "rhrf ldvi ywbb bcip"

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "✅ Email sent successfully!"
    except Exception as e:
        return f"❌ Failed to send email: {e}"

# === Assistant Logic ===
def assistant_action(data):
    user_data = data.lower().strip()
    jokes = [
        "Why don’t scientists trust atoms? Because they make up everything!",
        "I'm reading a book on anti-gravity. It's impossible to put down!"
    ]

    # Greetings
    if user_data in ["hi", "hello", "hey", "hai"]:
        return "Hello, my name is Alice. How can I assist you?"

    elif "name" in user_data:
        return "My name is Alice."
    elif "weather" in user_data:
        return get_weather()
    elif "time" in user_data:
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
    elif "joke" in user_data:
        return random.choice(jokes)
    elif "youtube" in user_data:
        webbrowser.open("https://youtube.com")
        return "Opened YouTube"
    elif "google" in user_data:
        webbrowser.open("https://google.com")
        return "Opened Google"
    else:
        return "I'm still learning! Could you please try rephrasing that?"

# === Streamlit Pages ===
menu = ["Home", "Assistance", "Planner", "Send Email", "About"]
with st.sidebar:
    choice = option_menu("Categories", menu, icons=["house", "mic", "calendar", "envelope", "info-circle"], default_index=0)

if 'history' not in st.session_state:
    st.session_state.history = []

if choice == "Home":
    st.title("🤖 Welcome to Alice - Your Virtual Assistant")
    st.markdown("""
        ## 👋 Hello There!
        Meet *Alice*, your always-ready AI assistant. Ask her anything!

        ### 🧠 Features:
        - 🎙 Voice & text input
        - ⛅ Real-time Weather
        - 📅 Calendar, Reminders, Notes
        - 📧 Email Sending
        - 💡 Fun with Jokes and Web help

        👉 Head to the Assistance tab to start talking with Alice!
    """)

elif choice == "Assistance":
    st.title("🧠 Talk to Alice")
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input("You:", placeholder="Type here...")
        if st.button("Send") and user_input:
            response = assistant_action(user_input)
            text_to_speech(response)
            st.session_state.history.append((user_input, response))
    with col2:
        if st.button("🎤 Speak"):
            spoken_text = speech_to_text()
            if spoken_text:
                st.success(f"You said: {spoken_text}")
                response = assistant_action(spoken_text)
                text_to_speech(response)
                st.session_state.history.append((spoken_text, response))

    st.subheader("📝 Conversation History")
    for user, bot in st.session_state.history:
        st.markdown(f"**You**: {user}")
        st.markdown(f"**Alice**: {bot}")

elif choice == "Planner":
    st.title("📅 Planner - Calendar & Reminders")
    with st.form("Add Reminder"):
        task = st.text_input("Task")
        date = st.date_input("Date", min_value=datetime.today())
        time = st.time_input("Time", value=datetime.now().time())
        if st.form_submit_button("Add Reminder"):
            reminder = {"task": task, "datetime": f"{date} {time}"}
            save_reminder(reminder)
            st.success("Reminder added!")

    st.subheader("📋 Your Reminders")
    reminders = load_reminders()
    if reminders:
        for rem in reminders:
            st.markdown(f"✅ {rem['task']} at {rem['datetime']}")
    else:
        st.info("No reminders set.")

elif choice == "Send Email":
    st.title("📧 Send an Email")
    with st.form("email_form"):
        to = st.text_input("To")
        subject = st.text_input("Subject")
        message = st.text_area("Message")
        if st.form_submit_button("Send"):
            result = send_email(to, subject, message)
            st.success(result)

elif choice == "About":
    st.title("📘 About Alice")
    st.markdown("""
        **Alice** is a fully voice-enabled AI Assistant created for a major-level project demo.

        Built with ❤️ using:
        - Python
        - Streamlit
        - Speech Recognition
        - Text-to-Speech
        - OpenWeatherMap API
        - SMTP for Email
    """)
    st.subheader("🔗 Links")
    st.markdown("[Google](https://google.com)")
    st.markdown("[YouTube](https://youtube.com)")
  
   
   
   
   
   
   
   
