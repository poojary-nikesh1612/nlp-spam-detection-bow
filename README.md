# NLP Spam Detection using Bag of Words

This is a mini NLP project that classifies messages as **Spam** or **Ham (Not Spam)** using **CountVectorizer (Bag of Words)** and **Multinomial Naive Bayes** from `scikit-learn`.

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Workflow

1. Load the dataset (`spam.csv`)
2. Convert labels (`spam = 1`, `ham = 0`)
3. Split data using `train_test_split()`
4. Convert text into numerical vectors using `CountVectorizer()`
5. Train the model using `MultinomialNB()`
6. Evaluate using `classification_report()`

## Project Structure

nlp-spam-detection-bow/
│
├── spam_detection.py
├── spam.csv
├── README.md
├── requirements.txt
└── .gitignore

## How to Run

```bash
pip install -r requirements.txt
python spam_detection.py
```

## Result

The model predicts whether a message is spam or ham using Bag of Words and Naive Bayes.

## Author

Built as part of learning Natural Language Processing (NLP).
