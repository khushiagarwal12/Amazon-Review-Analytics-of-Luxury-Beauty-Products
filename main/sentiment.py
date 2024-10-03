import json
import gzip
import nltk
import plotly.express as px
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Download NLTK data
nltk.download('vader_lexicon')

def load_data(file_path):
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def process_batch(batch, batch_number):
    sentiment_scores_list = []
    sid = SentimentIntensityAnalyzer()

    for i, review in enumerate(batch, start=1):
        review_text = review.get('reviewText', '')
        sentiment_scores = sid.polarity_scores(review_text)
        sentiment_scores_list.append(sentiment_scores)

    avg_scores = calculate_average(sentiment_scores_list)
    sentiment_statement = get_sentiment_statement(avg_scores['compound'])
    print(f"Batch {batch_number} Sentiment Statement: {sentiment_statement}\n", end="")

    visualize_sentiment_scores_plotly(avg_scores, batch_number)

    sentiment_labels = ["Positive" if review['overall'] >= 3.0 else "Negative" for review in batch]
    predicted_labels = ["Positive" if get_sentiment_statement(score['compound']) == 'Positive' else "Negative" for score in sentiment_scores_list]

    # Confusion Matrix and Metrics
    true_labels = sentiment_labels

    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="Positive")
    recall = recall_score(true_labels, predicted_labels, pos_label="Positive")
    f1 = f1_score(true_labels, predicted_labels, pos_label="Positive")

    print(f"Batch {batch_number}")
    print_confusion_matrix(cm)
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n\n\n\n\n")

def calculate_average(scores_list):
    num_reviews = len(scores_list)
    total_scores = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

    for scores in scores_list:
        for key in total_scores:
            total_scores[key] += scores[key]

    avg_scores = {key: total_scores[key] / num_reviews for key in total_scores}
    return avg_scores

def visualize_sentiment_scores_plotly(scores, batch_number):
    labels = list(scores.keys())
    values = list(scores.values())

    color_mapping = {'neg': 'rgba(255, 0, 0, 0.8)', 'neu': 'rgba(128, 128, 128, 0.8)',
                     'pos': 'rgba(0, 255, 0, 0.8)', 'compound': 'rgba(0, 0, 255, 0.8)'}

    fig = px.bar(x=labels, y=values, text=values, labels={'x': 'Sentiment Type', 'y': 'Score'},
                 title=f'Batch {batch_number} - Compound Sentiment Scores (Plotly)',
                 color=labels, color_discrete_map=color_mapping)

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 1])
    fig.show()

def get_sentiment_statement(compound_score, pos_threshold=0.75, neg_threshold=0.60, neu_lower=0.60, neu_upper=0.75):
    
    if compound_score >= pos_threshold:
        return "Positive"
    elif compound_score <= neg_threshold:
        return "Negative"
    elif neu_lower <= compound_score <= neu_upper:
        return "Neutral"
    else:
        return "Undetermined"

def print_confusion_matrix(cm):
    print("Confusion Matrix:")
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            label = ""
            if i == 0 and j == 0:
                label = "TP"
            elif i == 0 and j == 1:
                label = "FN"
            elif i == 1 and j == 0:
                label = "FP"
            elif i == 1 and j == 1:
                label = "TN"
            print(f"{cm[i][j]} {label}", end="\t")
        print()  # Move to the next line after each row

# Configurable parameters
file_path = 'Luxury_Beauty_5.json.gz'
batch_size = 100
max_batches = 10

# Load data
data = load_data(file_path)

# Batch processing
for batch_number in range(1, min(max_batches + 1, len(data) // batch_size + 1)):
    start_idx = (batch_number - 1) * batch_size
    end_idx = batch_number * batch_size
    batch = data[start_idx:end_idx]
    process_batch(batch, batch_number)
