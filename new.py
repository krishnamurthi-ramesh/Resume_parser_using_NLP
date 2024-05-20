import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import os
from PyPDF2 import PdfReader
import io
import re
import base64
from PIL import Image
import google.generativeai as genai
from courses import ds_course, web_course, android_course, ios_course, uiux_course
import en_core_web_sm

# Function to read and preprocess data
def read_data():
    train_df = pd.read_csv("dataset/train.csv")
    resumes_path = "dataset/trainResumes/"
    resumes = []
    for filename in os.listdir(resumes_path):
        if filename.endswith(".pdf"):
            with open(os.path.join(resumes_path, filename), 'rb') as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                resumes.append(text)
    train_df['Resume'] = resumes
    return train_df

# Function to train the model
def train_model(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    model = LinearRegression()
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('model', model)
    ])
    pipeline.fit(data['Resume'], data['Match Percentage'])
    return pipeline

# Function to predict match percentage for uploaded resume
# kinda error handling 
def predict_match(model, uploaded_resume):
    match_percentage = model.predict([uploaded_resume])
    return match_percentage[0]

# Function to read default job description from PDF
def read_job_description():
    with open("dataset/Job description.pdf", "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract skills from job description
def extract_skills(job_description):
    # Regular expression pattern to match skills (you may need to adjust this based on your job descriptions)
    pattern = r"\b(?:python|java|sql|Machine learning|deep learning|data analysis|statistics|data visualization|natural language processing|computer vision|big data|cloud computing|database management|software development)\b"
    skills = re.findall(pattern, job_description, flags=re.IGNORECASE)
    return list(set(skills))  # Return unique skills

# Function to highlight missing skills
def highlight_missing_skills(job_skills, resume_text):
    # Convert both job skills and resume text to lowercase for case-insensitive matching
    job_skills_lower = [skill.lower() for skill in job_skills]
    resume_text_lower = resume_text.lower()
    missing_skills = [skill for skill in job_skills_lower if skill not in resume_text_lower]
    return missing_skills

# Function to convert PDF bytes to base64 data URI
def pdf_to_base64(pdf_bytes):
    return base64.b64encode(pdf_bytes).decode('utf-8')

# Function to perform data analysis
def perform_data_analysis(job_skills, resume_text):
    # Convert both job skills and resume text to lowercase for case-insensitive matching

    job_skills_lower = [skill.lower() for skill in job_skills]
    resume_text_lower = resume_text.lower()
    
    job_skills_freq = {skill: resume_text_lower.count(skill.lower()) for skill in job_skills_lower}
    job_skills_df = pd.DataFrame.from_dict(job_skills_freq, orient='index', columns=['Frequency'])
    job_skills_df = job_skills_df.sort_values(by='Frequency', ascending=False)

    # Plotting skills frequency
    plt.figure(figsize=(10, 6))
    plt.barh(job_skills_df.index, job_skills_df['Frequency'], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Skill')
    plt.title('Frequency of Skills in Resume')
    st.pyplot()

# Function to recommend courses based on predicted field and skills
    
def recommend_courses(predicted_field, skills):
    if predicted_field == 'Data Science':
        return ds_course
    elif predicted_field == 'Web Development':
        return web_course
    elif predicted_field == 'Android Development':
        return android_course
    elif predicted_field == 'IOS Development':
        return ios_course
    elif predicted_field == 'UI-UX Development':
        return uiux_course
    else:
        return []

# Function to display course recommendations
def display_course_recommendations(courses):
    st.subheader("**Recommended Courses & Certifications**")
    for course_name, course_link in courses:
        st.markdown(f"[{course_name}]({course_link})")

# Function to recommend additional skills based on job market demand
def recommend_additional_skills(resume_text): 
    # using English language model - SpaCy

    nlp = en_core_web_sm.load()

    # Process the resume text using SpaCy
    doc = nlp(resume_text)

    # Get the tokens (words) from the resume
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]

    # Example of recommended skills (you can replace with your own logic)
    recommended_skills = ["Cloud Computing", "DevOps", "Agile Methodologies", "Blockchain", "Cybersecurity"]

    return recommended_skills

#interaction with Gemini AI Assistant
def gemini_assistant():
    st.write("Integration with virtual Assistant:")
    st.write("Welcome to this Career_Path_Finder Dashboard. You can proceed by providing your Google API Key")
    
    with st.expander("Provide Your Google API Key"):
        google_api_key = st.text_input("Google API Key", key="google_api_key", type="password")
        
    if not google_api_key:
        st.info("Enter the Google API Key to continue")
        st.stop()
        
    genai.configure(api_key=google_api_key) # used to configure the API Key
    
    st.title("virtual Assistant")

    with st.sidebar:
        option = st.selectbox('Choose Your Model', ('gemini-pro', 'gemini-pro-vision'))

        if 'model' not in st.session_state or st.session_state.model != option:
            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.model = option

        st.write("Adjust Your Parameter Here:")
        temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        max_token = st.number_input("Maximum Output Token", min_value=0, value=100)
        gen_config = genai.types.GenerationConfig(max_output_tokens=max_token, temperature=temperature)

    st.divider()

    st.markdown("<span ><font size=1>Connect With Me</font></span>", unsafe_allow_html=True)
    "www.linkedin.com/in/krishna-murthi-7671a8258"
    "[GitHub](https://github.com/cornelliusyudhawijaya)"
    
    st.divider()
    
    upload_image = st.file_uploader("Upload Your Image Here", accept_multiple_files=False, type=['jpg', 'png'])
    
    if upload_image:
        image = Image.open(upload_image)

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if upload_image:
        if option == "gemini-pro":
            st.info("Please Switch to the Gemini Pro Vision")
            st.stop()
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response = st.session_state.chat.send_message([prompt, image], stream=True, generation_config=gen_config)
            response.resolve()
            msg = response.text

            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
            st.image(image, width=300)
            st.chat_message("assistant").write(msg)

    else:
        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            response = st.session_state.chat.send_message(prompt, stream=True, generation_config=gen_config)
            response.resolve()
            msg = response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

# Main function to run the Streamlit app
def main():
    st.image("Resized.jpg", use_column_width=True)
    st.title("Talent Fit: Job Matching")

    # Read data and train the model
    data = read_data()
    model = train_model(data)

    # Use default job description from PDF
    default_job_description = read_job_description()
    st.text_area("Default Job Description", default_job_description, height=300)

    # Extract skills from job description
    job_skills = extract_skills(default_job_description)
    st.text("Skills required for the job: {}".format(", ".join(job_skills)))

    # Upload resume
    uploaded_resume = st.file_uploader("Upload your resume (PDF file)", type="pdf")

    if uploaded_resume is not None:
        resume_text = ""
        with uploaded_resume:
            resume_bytes = uploaded_resume.read()
            resume_stream = io.BytesIO(resume_bytes)  # Create a file-like object
            reader = PdfReader(resume_stream)
            for page in reader.pages:
                resume_text += page.extract_text()
        match_percentage = predict_match(model, resume_text)

        # Convert resume PDF bytes to base64 data URI
        resume_base64 = pdf_to_base64(resume_bytes)
        st.markdown(f'<embed src="data:application/pdf;base64,{resume_base64}" width="800" height="1000" type="application/pdf">', unsafe_allow_html=True)

        # Highlight missing skills
        missing_skills = highlight_missing_skills(job_skills, resume_text)
        present_skills = [skill for skill in job_skills if skill.lower() in resume_text.lower()]

        # Perform data analysis
        perform_data_analysis(job_skills, resume_text)
        
        # Display Match Percentage with style
        st.write("Match Percentage:", match_percentage)
        
        # Display skills using columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Skills present in the resume")
            for skill in set(present_skills):
                st.info(skill)
        with col2:
            st.subheader("Missing skills in the resume")
            for skill in set(missing_skills):
                st.error(skill)
        
        # Recommend courses based on predicted field and skills
        courses = recommend_courses("Data Science", present_skills)  # You can change the predicted field as needed
        if courses:
            display_course_recommendations(courses)

        # Recommend additional skills based on job market demand
        additional_skills = recommend_additional_skills(resume_text)
        if additional_skills:
            st.subheader("**Recommended Additional Skills**")
            for skill in additional_skills:
                st.markdown(f"- {skill}")

        # Integrated Gemini AI Assistant
        gemini_assistant()

if __name__ == "__main__":
    main()