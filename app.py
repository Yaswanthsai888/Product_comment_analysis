from flask import Flask, render_template, request, jsonify
import googleapiclient
from groq import Groq
from googleapiclient.discovery import build
import langid
import re
import emoji
import random
import pandas as pd
from textblob import TextBlob
import os
import csv

# Initialize Flask app
app = Flask(__name__)

# Initialize the Groq client
groq_client = Groq(api_key="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")

# Initialize YouTube API
YOUTUBE_API_KEY = "AIzaSyCMdGdtmX2RcxBBKgldmfTelvk02CMQmu8"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Function to generate keywords using Groq API
def generate_keywords(product_name):
    completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "User will provide a product. Provide a concise list of relevant keywords."},
                  {"role": "user", "content": product_name}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )
    keywords = completion.choices[0].message.content.strip().split("\n")
    return keywords

# Function to fetch comments from YouTube based on search query and keywords
def fetch_comments(search_query, keywords):
    search_response = youtube.search().list(
        q=search_query,
        part="id,snippet",
        type="video",
        maxResults=5
    ).execute()

    video_ids = [item["id"]["videoId"] for item in search_response["items"]]
    all_comments = []

    for video_id in video_ids:
        try:
            # Fetch video details to check comment status
            video_response = youtube.videos().list(part="snippet,statistics", id=video_id).execute()

            # Check if comments are disabled
            if "commentCount" not in video_response["items"][0]["statistics"]:
                print(f"Comments disabled for video: {video_id}")
                continue  # Skip video if comments are disabled

            comments_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100
            ).execute()

            # Process comments if available
            for item in comments_response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                if any(keyword.lower() in comment.lower() for keyword in keywords):
                    all_comments.append(comment)

        except googleapiclient.errors.HttpError as error:
            # Handle specific errors
            if "commentsDisabled" in str(error):
                print(f"Comments are disabled for video {video_id}. Skipping...")
            else:
                print(f"An error occurred while fetching comments for video {video_id}: {error}")

    return all_comments


def preprocess_comments(comments):
    """
    Preprocess the comments by:
    - Keeping only English comments
    - Removing URLs
    - Removing Emojis
    - Removing special characters
    """
    def remove_urls(comment):
        """Remove URLs from the comment."""
        return re.sub(r'http\S+|www\S+', '', comment)

    def remove_emojis(comment):
        """Remove emojis from the comment."""
        return emoji.replace_emoji(comment, replace='')

    def remove_special_chars(comment):
        """Remove special characters except for spaces."""
        return re.sub(r'[^A-Za-z0-9\s]', '', comment)

    # Filter comments to keep only English ones
    english_comments = [comment for comment in comments if langid.classify(comment)[0] == 'en']

    # Clean each comment by removing URLs, emojis, and special characters
    cleaned_comments = []
    for comment in english_comments:
        comment = remove_urls(comment)  # Remove URLs
        comment = remove_emojis(comment)  # Remove Emojis
        comment = remove_special_chars(comment)  # Remove special characters
        cleaned_comments.append(comment)

    print("Preprocessed and Cleaned English Comments:", cleaned_comments)
    return cleaned_comments

# Function to remove short or irrelevant comments
def filter_short_comments(comments, min_length=5, keywords_to_remove=["aoa", "hi", "hello"]):
    """
    Remove short comments or those containing irrelevant words/phrases.
    """
    filtered_comments = []
    for comment in comments:
        # Remove comments that are too short or contain irrelevant keywords
        if len(comment) >= min_length and not any(keyword.lower() in comment.lower() for keyword in keywords_to_remove):
            filtered_comments.append(comment)
    print("Filtered Comments After Length and Keyword Removal:", filtered_comments)
    return filtered_comments

# Function to save comments to CSV
def save_comments_to_csv(comments, file_path):
    """
    Save filtered and cleaned comments to a CSV file.
    """
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Comment"])  # Write header
        for comment in comments:
            writer.writerow([comment])  # Write each filtered comment

# Function for sentiment analysis
def get_sentiment(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to detect questions
def is_question(comment):
    return '?' in comment

# Function to detect feelings/emotions (example: excited, angry, etc.)
def detect_feelings(comment):
    feelings_keywords = ['love', 'hate', 'excited', 'angry', 'happy', 'frustrated', 'sad', 'great', 'best', 'worst']
    for word in feelings_keywords:
        if word in comment.lower():
            return 'Feeling'
    return 'Other'

# Function to summarize comments using Groq API
def summarize_comments(comments_batch):
    """
    Summarize the given batch of comments using the Groq API.
    """
    summary = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a helpful assistant. Summarize the given batch of comments succinctly also try to eloborate more about those points by your understanding."},
                  {"role": "user", "content": "\n".join(comments_batch)}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )
    return summary.choices[0].message.content.strip()


def generate_answer(question):
    """
    Generate an answer to a question using the Groq API.
    """
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": "If there are any qustions in these summary try to answer them from your knowledge and try to make qustions and answers in short points you should print Qustion followed my answer keep in mind short points"},
                      {"role": "user", "content": question}],
            temperature=1,
            max_tokens=150,
            top_p=1,
            stream=False,
            stop=None
        )
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error while generating answer: {e}")
        return "Sorry, I couldn't answer that question."


@app.route('/')
def index():
    return render_template('index.html')  # Render the input page

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get product name from the form
    product_name = request.form['product_name']
    
    # Process the comments
    keywords = generate_keywords(product_name)
    comments = fetch_comments(product_name, keywords)
    preprocessed_comments = preprocess_comments(comments)
    filtered_comments = filter_short_comments(preprocessed_comments)
    
    # Sentiment analysis
    df = pd.DataFrame(filtered_comments, columns=['Comment'])
    df['Sentiment'] = df['Comment'].apply(get_sentiment)
    
    # Group comments by sentiment
    grouped_comments = df.groupby('Sentiment')['Comment'].apply(list).to_dict()

    # Summarize the comments
    positive_summary = summarize_comments(random.sample(grouped_comments.get('Positive', []), 15))
    negative_summary = summarize_comments(random.sample(grouped_comments.get('Negative', []), 15))
    neutral_summary = summarize_comments(random.sample(grouped_comments.get('Neutral', []), 15))

    summaries = {
        'Positive': positive_summary,
        'Negative': negative_summary,
        'Neutral': neutral_summary
    }

    # Pass summaries as questions to generate answers
    question_answers = {}
    for sentiment, summary in summaries.items():
        question = f"What do people think about the {product_name}'s {sentiment.lower()} feedback? Here's a summary: {summary}"
        answer = generate_answer(question)
        question_answers[answer] = None  # Store only the answer, not the question

    return render_template('results.html', summaries=summaries, question_answers=question_answers)



if __name__ == '__main__':
    app.run(debug=True)
