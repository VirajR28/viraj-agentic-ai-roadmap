import streamlit as st
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import clean_text

# Page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model
@st.cache_resource
def load_model():
    """Load the trained spam classifier model."""
    model_path = Path(__file__).parent.parent / 'models' / 'spam_classifier.joblib'
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        st.info("Please run the notebook to train the model first.")
        st.stop()
    return joblib.load(model_path)

# Page content
st.title("📧 Spam Classifier")
st.markdown("---")

st.write("""
This classifier uses **TF-IDF** and **Logistic Regression** to determine whether 
a message is spam or legitimate.

Enter a message below and click "Classify" to see the prediction.
""")

st.markdown("---")

# Load model
model = load_model()

# Input section
st.subheader("Enter your message:")
user_message = st.text_area(
    "Message text:",
    placeholder="Paste your message here...",
    height=120,
    label_visibility="collapsed"
)

# Classify button
col1, col2, col3 = st.columns(3)
with col2:
    classify_button = st.button("🔍 Classify", use_container_width=True)

st.markdown("---")

# Classification
if classify_button:
    if not user_message.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Preprocess
        cleaned_message = clean_text(user_message)
        
        # Predict
        prediction = model.predict([cleaned_message])[0]
        probabilities = model.predict_proba([cleaned_message])[0]
        
        # Get label and confidence
        if prediction == 1:
            label = "🚨 SPAM"
            confidence = probabilities[1]
            color = "red"
        else:
            label = "✓ NOT SPAM"
            confidence = probabilities[0]
            color = "green"
        
        # Display results
        st.markdown(f"### Result: {label}")
        
        # Confidence metric
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Model Probability",
                value=f"{confidence:.2%}"
            )
        
        with col2:
            st.metric(
                label="Confidence",
                value="High" if confidence > 0.8 else ("Medium" if confidence > 0.6 else "Low")
            )
        
        # Detailed probabilities
        st.markdown("---")
        st.subheader("Detailed Probabilities:")
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.write(f"**Not Spam:** {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
        
        with prob_col2:
            st.write(f"**Spam:** {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")
        
        # Visualization
        st.markdown("---")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ['Not Spam', 'Spam']
        probs = [probabilities[0] * 100, probabilities[1] * 100]
        colors_list = ['green', 'red']
        
        bars = ax.barh(categories, probs, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0, 100)
        ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 2, i, f'{prob:.1f}%', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
        
        # Additional info
        st.markdown("---")
        st.markdown("""
        ### ℹ️ About this Classifier
        
        - **Model:** Logistic Regression
        - **Features:** TF-IDF
        - **Training Data:** Small local demo dataset for learning and testing
        - **Note:** This is a simple educational project, not a production spam filter.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 12px; color: gray;'>
    <p>Spam Classifier v1.0 | Machine Learning Foundations Project</p>
</div>
""", unsafe_allow_html=True)
