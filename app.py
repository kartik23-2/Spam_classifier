import streamlit as st
import pickle
import sklearn 


@st.cache_resource
def load_model():
    """Loads the saved spam detection pipeline from a pickle file."""
    try:
        # Open the file 
        with open('spam_model.pkl', 'rb') as file:
            pipeline = pickle.load(file)
            return pipeline
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'spam_model.pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

spam_pipeline = load_model()

#  Create the Streamlit App UI 
st.set_page_config(page_title="Spam Detector", page_icon="🚫")
st.title("Spam Comment Detector")
st.write("Enter a comment below to check if it's spam or not.")

# Add a text area for user input
user_input = st.text_area("Your Comment:", height=150)

# Add a button to classify
if st.button("Classify Comment"):
    # Ensure the model is loaded and input is provided
    if spam_pipeline is not None and user_input.strip() != "":
        # Make Prediction 
        prediction = spam_pipeline.predict([user_input])
        probability = spam_pipeline.predict_proba([user_input])
        
        # Get the probability of the 'Spam' class (which is class 1)
        spam_prob = probability[0][1] # probability of class '1'

        # --- 4. Display the Result ---
        if prediction[0] == 1:
            st.error(f"This looks like Spam. (Confidence: {spam_prob*100:.2f}%)")
        else:
            st.success(f"This looks like a Normal Comment. (Confidence: {(1-spam_prob)*100:.2f}%)")
            
    elif user_input.strip() == "":
        st.warning("Please enter a comment to classify.")
