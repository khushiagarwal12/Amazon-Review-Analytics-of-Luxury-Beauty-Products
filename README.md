# Amazon-Review-Analytics-of-Luxury-Beauty-Products
 Running `prediction_model.py`

# Steps to Run the Script

1. Install Required Packages:
   Ensure you have the necessary Python packages installed:
    pip install pandas matplotlib seaborn scikit-learn wordcloud
   

2. Prepare Data:
Make sure that the dataset `Luxury_Beauty_5.json.gz` is available in the same directory as the script or update the file path in the script accordingly.

3. Run the Script:
   Execute the script using Python:
   python prediction_model.py
   

# Expected Output

1. Best Ridge Alpha:
   - Prints the best alpha parameter found during hyperparameter tuning:
     
     Best Ridge Alpha: <value>
     

2. Ridge Mean Squared Error:
   - Prints the mean squared error of the Ridge regression model on the test set:
     
     Ridge Mean Squared Error: <value>
     

3. Linear Regression Mean Squared Error:
   - Prints the mean squared error of the Linear regression model on the test set:
     
     Linear Regression Mean Squared Error: <value>
     

4. Histograms and Word Cloud:
   - Histograms:
     A histogram comparing predicted ratings by Ridge and Linear Regression models against the actual ratings. This will be displayed as a visual plot using `matplotlib` and `seaborn`.

   - Word Cloud:
     A word cloud generated from the review texts. This will be displayed as a visual plot using `wordcloud`.

 Running `sentiment.py`

# Steps to Run the Script

1. Install Required Packages:
   Ensure you have the necessary Python packages installed:
      pip install nltk plotly scikit-learn
   

2. Download NLTK Data:
   The script includes code to download the required NLTK data, specifically the VADER lexicon.

3. Prepare Data:
   Make sure that the dataset `Luxury_Beauty_5.json.gz` is available in the same directory. 

4. Run the Script:
   Execute the script using Python:
   python sentiment.py
   

# Expected Output

1. Sentiment Analysis for Each Batch:
   - For each batch of reviews:
     - Prints the sentiment statement for the batch:
       
       Batch <number> Sentiment Statement: <Positive/Negative/Neutral/Undetermined>
       

     - Displays a bar chart of sentiment scores using Plotly. This chart will be a visual plot shown in a web browser or Jupyter notebook.

2. Confusion Matrix and Metrics:
   - Prints the confusion matrix and evaluation metrics (accuracy, precision, recall, F1 score) for each batch:
     
     Batch <number>
     Confusion Matrix:
     <TP> TP    <FN> FN
     <FP> FP    <TN> TN

     Metrics:
     Accuracy: <value>
     Precision: <value>
     Recall: <value>
     F1 Score: <value>
     
