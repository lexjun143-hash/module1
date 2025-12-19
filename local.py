# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
from dotenv import load_dotenv
import ollama

# Load environment variables
load_dotenv()

# Helper function to get dataset path
def get_dataset_path():
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the CSV file
    csv_path = os.path.join(current_dir,  "data", "customer_reviews.csv")
    return csv_path

# Function to get sentiment using local Gemma3 model
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        # Create the prompt
        prompt = f"""Classify the sentiment of the following review as exactly one word: Positive, Negative, or Neutral.
        
        Review: {text}
        
        Respond with only one word: Positive, Negative, or Neutral.
        """
        
        # Generate the response using local Gemma3 model
        response = ollama.chat(
            model='gemma3',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            options={
                'temperature': 0.1,  # Lower temperature for more deterministic outputs
            },
        )
        
        # Extract and clean the response
        sentiment = response['message']['content'].strip().capitalize()
        
        # Ensure the response is one of the expected values
        valid_sentiments = ["Positive", "Negative", "Neutral"]
        if sentiment not in valid_sentiments:
            # If the response is not exact, try to match the beginning
            for s in valid_sentiments:
                if sentiment.lower().startswith(s.lower()):
                    return s
            return "Neutral"
            
        return sentiment
    except Exception as e:
        st.error(f"Ollama error: {e}")
        return "Neutral"

def main():
    st.title("🔍 Local Gemma3 Sentiment Analysis")
    st.write("Analyze customer reviews with AI-powered sentiment analysis using a local Gemma3 model.")
    st.info("Running locally with Ollama - No internet connection required after model download.")

    # Layout two buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Load Dataset"):
            try:
                csv_path = get_dataset_path()
                df = pd.read_csv(csv_path)
                st.session_state["df"] = df.head(100)  # Load first 10 rows by default
                st.success("Dataset loaded successfully!")
            except FileNotFoundError:
                st.error("Dataset not found. Please check the file path.")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

    with col2:
        if st.button("🔍 Analyze Sentiment"):
            if "df" in st.session_state:
                try:
                    with st.spinner("Analyzing sentiment..."):
                        # Add a progress bar
                        progress_bar = st.progress(0)
                        total_rows = len(st.session_state["df"])
                        
                        # Process each row with progress update
                        sentiments = []
                        for i, text in enumerate(st.session_state["df"]["SUMMARY"]):
                            sentiments.append(get_sentiment(text))
                            progress_bar.progress((i + 1) / total_rows)
                        
                        st.session_state["df"]["Sentiment"] = sentiments
                        st.success("Sentiment analysis completed!")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
            else:
                st.warning("Please load the dataset first.")

    # Display the dataset if it exists
    if "df" in st.session_state:
        # Product filter dropdown
        st.subheader("🔍 Filter by Product")
        product = st.selectbox(
            "Choose a product", 
            ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique())
        )
        
        st.subheader(f"📁 Reviews for {product}")

        # Filter data based on product selection
        if product != "All Products":
            filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
        else:
            filtered_df = st.session_state["df"]
            
        st.dataframe(filtered_df)

        # Visualization using Plotly if sentiment analysis has been performed
        if "Sentiment" in st.session_state["df"].columns:
            st.subheader(f"📊 Sentiment Breakdown for {product}")
            
            # Create Plotly pie chart for sentiment distribution
            sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            # Define custom colors
            sentiment_colors = {
                'Negative': '#FF6B6B',  # Light red
                'Neutral': '#4ECDC4',   # Teal
                'Positive': '#77DD77'    # Light green
            }
            
            # Create pie chart
            fig = px.pie(
                sentiment_counts, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=sentiment_colors,
                hole=0.4
            )
            
            # Update layout
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#000000', width=1))
            )
            
            fig.update_layout(
                showlegend=False,
                margin=dict(t=0, b=0, l=0, r=0),
                height=400
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
