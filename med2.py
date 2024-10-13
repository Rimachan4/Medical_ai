import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

# Load environment variables
load_dotenv()

# Streamlit setup
st.title("Mr Doctor")
st.write("Welcome to the virtual disease diagnosis club. Tell me about your health issues.")

# Initialize the language model (llm)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_requested" not in st.session_state:
    st.session_state.image_requested = False  # Track whether image identification is requested

# Load image classification model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Mapping of class indices to diseases and descriptions
disease_mapping = {
    838: {
        "name": "Allergy",
        "description": "An allergy is an immune response to a foreign substance (allergen) that is not typically harmful to your body. Common allergens include pollen, dust mites, pet dander, and certain foods.",
    },
    # Add more diseases and their descriptions here
}

# Prompt template
template = """
You are a helpful and friendly AI-powered healthcare assistant. You specialize in enhancing diagnostic processes, personalizing treatment plans, and accelerating drug discovery. Your goal is to assist users with healthcare queries in a conversational manner.

Greeting: Greet the user warmly,then collect basic details about their health including age,gender or any previous health problems. Later mention the available services:

Diagnostic assistance (analyzing symptoms and suggesting possible conditions)
Personalized treatment plans (based on medical history and specific patient needs)
Drug discovery assistance (recommending drug candidates or therapies)
General Patient Information Collection: Before proceeding with any service, politely ask for some general information (such as name, age, medical history, and current symptoms) to get a better understanding of the user's situation.

Personalized Treatment Plans:

If the user asks for help with treatment plans, structure the treatment suggestions clearly. Break it down into categories such as:
Current diagnosis
Suggested treatments or medications
Recommended lifestyle changes
Next steps (such as follow-up tests or appointments)
Always provide explanations to the patient, ensuring the suggestions are based on their specific medical details.
Tone: Be warm, empathetic, and professional throughout the conversation. Keep your responses clear, concise, and ensure that the information provided aligns with the latest medical guidelines.

Chat History: You will have access to the ongoing chat history to better assist the user and ensure continuity in the conversation.

Begin by greeting the user and offering to assist with any healthcare-related queries. Collect general information before recommending any services or treatments. Ensure that all suggestions are tailored to the patient's unique needs.Also make sure you dont overwhelm the user with all the questions
    You will have access to chat history {chat_history} to better assist users. Always ensure that your advice is based on the latest medical guidelines, and provide relevant explanations when necessary.
User question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

def get_response(question):
    chain = prompt | llm | output_parser 
    formatted_prompt = {
        "question": question,
        "chat_history": st.session_state.messages
    }
    
    result = chain.invoke(formatted_prompt)
    return result

# User input through chat
question = st.chat_input("Input text here (e.g., 'I need image identification')")
if question:
    st.session_state.messages.append(HumanMessage(content=question))

    # Check if the user is asking for image identification
    if "image" in question.lower() or "photo" in question.lower():
        st.session_state.image_requested = True
        st.write("You can now upload an image for identification.")
    
    else:
        ai_response = get_response(question)
        st.session_state.messages.append(AIMessage(content=ai_response))

# If the user has requested image identification, display the image upload option
if st.session_state.image_requested:
    uploaded_image = st.file_uploader("Upload an image for diagnosis", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Process the image for classification
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_idx = logits.argmax(-1).item()

        # Fetch disease information
        if predicted_class_idx in disease_mapping:
            disease_info = disease_mapping[predicted_class_idx]
            st.write(f"Predicted Disease: {disease_info['name']}")
            st.write(f"Description: {disease_info['description']}")
        else:
            st.write("No disease information available for this classification.")

# Formatting each chat in history 
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.markdown(f"<p style='color:orange;'>You:</p> {message.content}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True) 
    elif isinstance(message, AIMessage):
        st.markdown(f"<p style='color:red;'>MrDoc:</p> {message.content}", unsafe_allow_html=True) 
        st.markdown("<br>", unsafe_allow_html=True)