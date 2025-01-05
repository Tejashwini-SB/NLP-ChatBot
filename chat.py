import streamlit as st
import random
import os
import csv
import datetime
import nltk
import json
import numpy as np
import streamlit as st
import pyttsx3
import pyaudio
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from the JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Initialize training data variables
training_sentences = []
training_labels = []
class_labels = []

# Process the intents and prepare training data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])  
    if intent['tag'] not in class_labels:  
        class_labels.append(intent['tag'])

# Lemmatize and tokenize the sentences
def preprocess_data(sentence):
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

training_sentences = [preprocess_data(sentence) for sentence in training_sentences]

# Encode labels using LabelEncoder
le = LabelEncoder()
training_labels = le.fit_transform(training_labels)

# Convert sentences to vectors using TF-IDF
vectorizer = TfidfVectorizer()
training_sentences_vectorized = vectorizer.fit_transform([' '.join(sentence) for sentence in training_sentences]).toarray()

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=training_sentences_vectorized.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(class_labels), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(training_sentences_vectorized, np.array(training_labels), epochs=15, batch_size=8)

# Evaluate the model's accuracy
loss, accuracy = model.evaluate(training_sentences_vectorized, np.array(training_labels))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def predict_intent(user_input):
    user_input = preprocess_data(user_input)
    user_input_vectorized = vectorizer.transform([' '.join(user_input)]).toarray()
    prediction = model.predict(user_input_vectorized)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:  
            return np.random.choice(i['responses'])
    return "Sorry, I didn't understand that."


# Update message display
def display_messages():
    for msg in st.session_state.messages:  # Display messages in the order they were added
        if msg["role"] == "user":
            st.markdown(f"<div style='text-align: left; color: black; background-color: lavender; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>😊 : {msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: white; background-color: MediumPurple; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>🤖 : {msg['text']}</div>", unsafe_allow_html=True)

def messages():
    # Check if there's at least one message in the session state
    if len(st.session_state.messages) > 1:
        user_message = st.session_state.messages[-2]  # Second last message is the user message
        bot_message = st.session_state.messages[-1]   # Last message is the bot response

        if user_message["role"] == "user":
            st.markdown(f"""
                <div style='text-align: left; color: black; background-color: lavender; padding: 10px;border-radius: 10px; margin-bottom: 10px;'>😊 : {user_message['text']}</div>""", unsafe_allow_html=True)

        if bot_message["role"] == "bot":
            st.markdown(f"""<div style='text-align: left; color: white; background-color: MediumPurple; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>🤖 : {bot_message['text']}</div>""", unsafe_allow_html=True)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Ask me something! I’m listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.success("Processing your speech...")
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized Speech: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio.")
            return None
        except sr.RequestError:
            print("Sorry, the speech recognition service is unavailable.")
            return None


def main():
    st.title("Intento🤖")

    # Initialize conversation state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar Menu
    menu = ["Home 🏠", "Conversation History 📜", "About ℹ️"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home 🏠":
        st.write("Welcome to the Chatbot! Start by typing a message 😊")

        # User input area with fixed height
        user_input = st.text_input("You:", key="user_input")

        # Button for predefined responses
        col1, col2, col3 ,col4 = st.columns(4)
        with col1:
            if st.button("Speak to Me 🎤"):
                user_input = recognize_speech()
                if user_input:
                    st.session_state.messages.append({"role": "user", "text": user_input})
            
                    intent = predict_intent(user_input)
                    response = get_response(intent)
            
                    st.session_state.messages.append({"role": "bot", "text": response})
                    speak(response)  # Speak the bot response

        with col2:
            if st.button("hi 👋"):
                user_input = "Hello"
        with col3:
            if st.button("Ask About the Bot 🤖"):
                user_input = "What is your purpose?"
        with col4:
            if st.button("Goodbye 👋"):
                user_input = "Goodbye"

        if user_input:
            # Store user message in session state
            st.session_state.messages.append({"role": "user", "text": user_input})

            # Get the intent and response from the chatbot
            intent = predict_intent(user_input)
            response = get_response(intent)

            # Store chatbot response in session state
            st.session_state.messages.append({"role": "bot", "text": response})

        
            messages()


    # Conversation History Menu
    elif choice == "Conversation History 📜":
        st.header("Conversation History")
        # Display conversation history
        display_messages()

    elif choice == "About ℹ️":
        st.subheader("About the Project")
        st.write("""
        **Intento: Intent-Based Chatbot** is a conversational chatbot that leverages Natural Language Processing (NLP) techniques and a neural network model to understand and respond to user inputs based on predefined intents.
        The chatbot offers an interactive, user-friendly interface built with Streamlit and supports both text-based and voice-based interactions.
        """)
        st.subheader("Key Features")
        st.write("""
        1. **Intent Recognition**:
        - Uses labeled intents and patterns stored in a JSON file to train the model.
        - Implements TF-IDF Vectorization to convert user inputs into feature vectors.
        - Applies a neural network model with layers for intent classification.
        
        2. **Interactive UI**:
        - Built using Streamlit, offering an intuitive web interface.
        - Includes buttons for predefined responses and voice input recognition using SpeechRecognition.
        
        3. **Speech Interaction:**
        - Incorporates voice recognition with Google Speech API.
        -Responses are vocalized using the pyttsx3 library for a more interactive experience.
        
        4. **Customizable Responses:**
        - Dynamically selects appropriate responses from predefined options for each intent.
        
        5. **Conversation History:**
        - Stores and displays the entire conversation for review and context.

        """)
        st.subheader("Dataset:")
        st.write("""
        The dataset contains labeled intents and patterns:
        - **Intents:** Represent the goal of user inputs (e.g., "greeting", "help").
        - **Patterns:** Text inputs corresponding to each intent.
        - **Responses:** Possible chatbot replies for each intent.
        """)
        st.subheader("Conclusion:")
        st.write("""
        This chatbot demonstrates the application of NLP techniques and Streamlit for a functional, user-friendly chatbot. Future work may include deep learning models and expanded datasets.
        """)

    

    


if __name__ == '__main__':
    main()
