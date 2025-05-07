


    
    
    
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns

# Inject CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background-color: #fff;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_data(file_path=None):
    """Load the dataset from a default path or user-uploaded file."""
    try:
        if file_path:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv("news.csv")  # Replace with the actual path
        return df
    except Exception as e:
        st.error("Error loading data: " + str(e))
        return None

@st.cache_resource
def train_model(df):
    """Train the model and return the trained pipeline."""
    # Check dataset validity
    if "text" not in df.columns or "label" not in df.columns:
        st.error("Dataset must contain 'text' and 'label' columns.")
        return None

    # Split the data
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test

def main():
    """Main Streamlit app."""
    st.title("Fake News Detection")

    # Sidebar navigation
    menu = ["Home", "Fake News Detection", "Model Performance", "Upload Dataset", "About"]
    with st.sidebar:
        choice = option_menu(
            "News Detector",
            menu,
            icons=['house', 'search', 'chart-bar', 'upload', 'info-circle'],
            menu_icon="cast",
            default_index=0
        )

    if choice == "Home":
        st.markdown("## Welcome to the Fake News Detection App!")
        st.write("This app leverages machine learning to classify news as real or fake. Explore the features to get started.")
        st.markdown("""
        <div style='text-align: center;'>
            <h3>Explore Our Features:</h3>
            <ul style='list-style-type: none; padding: 0;'>
                <li style='margin: 10px 0;'><button id='traffic-button' style='padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Fake News Detection</button></li>
                <li style='margin: 10px 0;'><button id='data-button' style='padding: 10px 20px; font-size: 16px; background-color: #008CBA; color: white; border: none; border-radius: 5px;'>Model Performance</button></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <script>
        document.getElementById('traffic-button').onclick = function() {
            alert('Redirecting to Lane Detection...');
        };
        document.getElementById('data-button').onclick = function() {
            alert('Redirecting to Object Detection...');
        };
        </script>
        """, unsafe_allow_html=True)

    elif choice == "Fake News Detection":
        st.header("Enter a News Article:")
        news_text = st.text_area("Type or paste the news article text:")
        
        df = load_data()
        if df is not None:
            pipeline, _, _, _, _ = train_model(df)
            
            if st.button("Check"):
                if news_text.strip():
                    prediction = predict_news(news_text, pipeline)
                    st.subheader("Prediction:")
                    if prediction == "FAKE":
                        st.error("This news article is likely Fake.")
                    else:
                        st.success("This news article is likely Real.")
                else:
                    st.warning("Please enter a news article.")

    elif choice == "Model Performance":
        st.header("Model Performance Evaluation")
        df = load_data()
        if df is not None:
            pipeline, _, X_test, _, y_test = train_model(df)
            
            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
            
            # Classification report
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
            plt.title("Confusion Matrix")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            st.pyplot(fig)

    elif choice == "Upload Dataset":
        st.header("Upload Your Own Dataset")
        uploaded_file = st.file_uploader("Upload a CSV file with 'text' and 'label' columns:", type="csv")
        if uploaded_file:
            custom_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write(custom_df.head())
            pipeline, _, _, _, _ = train_model(custom_df)
            st.write("Model trained on uploaded dataset!")

    elif choice == "About":
        st.header("About")
        st.markdown("""
        Welcome to the **Fake News Detection App**! This app leverages cutting-edge machine learning technologies to help identify whether a news article is real or fake. Here's what makes this app special:
        """)

        st.markdown("### Explore Our Features:")
        tab1, tab2, tab3 = st.tabs(["Overview", "Technology", "How It Works"])

        with tab1:
            st.subheader("Overview")
            st.markdown("""
            - **Purpose**: Combat misinformation by providing an easy-to-use fake news detection tool.
            - **Audience**: Journalists, researchers, and anyone who consumes online content.
            - **Features**:
              1. Upload custom datasets for personalized predictions.
              2. Visualize model performance metrics like confusion matrices.
              3. Accurate predictions using a pre-trained machine learning pipeline.
            """)

        with tab2:
            st.subheader("Technology Stack")
            st.markdown("""
            This app is built using:
            - **Streamlit**: For creating the interactive web application.
            - **Python**: Core programming language for logic and data handling.
            - **Machine Learning**:
              - **Logistic Regression**: A powerful classification model.
              - **TF-IDF Vectorization**: Converts text into meaningful numerical data.
            - **Visualization Libraries**:
              - **Matplotlib** and **Seaborn** for graphs and heatmaps.
            """)

        with tab3:
            st.subheader("How It Works")
            st.markdown("""
            The process involves:
            1. **Preprocessing**: Cleaning and transforming raw text data.
            2. **Vectorization**: Using TF-IDF to convert text into numerical features.
            3. **Model Prediction**: Leveraging logistic regression for binary classification.
            4. **Evaluation**: Displaying performance metrics like accuracy and confusion matrix.
            """)

        st.markdown("### Frequently Asked Questions (FAQ)")
        faq_expander = st.expander("Why is detecting fake news important?")
        faq_expander.write("Misinformation can spread quickly and cause harm to individuals, communities, and societies. Tools like this help combat the spread of false information.")
        
        faq_expander = st.expander("Can I trust the model predictions?")
        faq_expander.write("While the model is trained on a high-quality dataset, no model is 100% accurate. Use the predictions as a guide, not an absolute truth.")
        
        faq_expander = st.expander("Can I use my own dataset?")
        faq_expander.write("Yes! Navigate to the 'Upload Dataset' section and upload your CSV file to train the model on your custom data.")

        st.markdown("""
        ---
        *Thank you for exploring the app! Feel free to reach out if you have suggestions or questions.*
        """)

def predict_news(news_text, pipeline):
    """Predict whether the given news article is real or fake."""
    prediction = pipeline.predict([news_text])[0]
    return prediction

if __name__ == "__main__":
    main()

    
    
    
    
    
   
    
    
    
    

    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

