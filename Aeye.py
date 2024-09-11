import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
from PIL import Image


import urllib.request

# Function to convert text to speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)  # Open default camera (index 0)
    ret, frame = cap.read()  # Read a frame from the camera
    cap.release()  # Release the camera
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    return None

def capture_mobile_image(URL):

    # URL = "http://10.12.37.153:8080/shot.jpg"

    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)

    frame = cv2.imdecode(img_arr, -1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return img


# Function to recognize speech
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio = r.listen(source)

    # Using Google to recognize audio
    try:
        MyText = r.recognize_google(audio)
        MyText = MyText.lower()
        return MyText
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


genai.configure(api_key="AIzaSyBRumJNZQGgRNdUvAIHwl-C7bzUrGmOciA") #Enter your API key, PS: This key is depreciated:).

def get_gemini_response(input_text, image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    if input_text:
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text


def start_button():
    start = True

def stop_button():
    start = False

st.sidebar.header("Navigation")

# Sidebar navigation links with bullets
st.sidebar.markdown("- [GitHub Repo](https://github.com/samyjn/AEye.ai.git)")
st.sidebar.write("---")

# Connect with me section
st.sidebar.markdown("Connect with us:")
github_link = "[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat-square&logo=github)](https://github.com/samyjn)"
linkedin_link = "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com)"
email_link = "[![Email](https://img.shields.io/badge/Google-Mail-blue?style=flat-square&logo=gmail)](mailto:samyjn211@gmail.com)"

st.sidebar.markdown(github_link + " " + linkedin_link + " " + email_link)
st.sidebar.markdown("Created by Team AEye.AI from Graphic Era Deemed University")


def main():
    st.title("AEye.ai")
    st.write("AI Eyes in your service! Just Speak your question.")


    cam = st.checkbox('Select only if external camera source')
    if cam:
        URL = st.text_input('Give the Link to external camera source below')

    col1, col2 = st.columns([.5,1])
    with col1:
        start = st.button('Start', on_click=start_button)

    with col2:
        st.button('Stop', on_click=stop_button)


    if start:
        while True:

            input_text = recognize_speech()
            if input_text:
                st.write("You asked:", input_text)
                st.write("Capturing image...")

                if not cam:
                    image = capture_image()
                else:
                    image = capture_mobile_image(URL)

                if image is not None:
                    pil_image = Image.fromarray(image)
                    st.image(pil_image, caption='Captured Image', use_column_width=True)
                    
                    response = get_gemini_response(input_text, pil_image)
                
                    st.write("Response:", response)
                    SpeakText(response)
            #     else:
            #         st.warning("Failed to capture image. Please try again.")
            else:
                st.warning("Could not understand the speech. Please try again.")
        
    
if __name__ == "__main__":
    main()