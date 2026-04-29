import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df=pd.read_csv('spam.csv')

# print(df.head())

df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)

# print(df.head())

X_train,X_test,y_train,y_test=train_test_split(df.Message,df.spam,test_size=0.2, random_state=42)

model=Pipeline(
    [('cv',CountVectorizer()),
     ('nb',MultinomialNB())]
)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

report=classification_report(y_test,y_pred)

print(report)


# emails = [
#     "Congratulations! You have won a free iPhone. Click here now to claim your prize.",
    
#     "Dear customer, your bank account has been temporarily suspended. Please verify your details immediately to avoid deactivation.",
    
#     "Limited time offer! Get 70% discount on all products. Buy now and save big.",
    
#     "Hi John, just reminding you about tomorrow's team meeting at 10 AM in the conference room.",
    
#     "Hello, your Amazon order has been shipped successfully and will arrive by Friday.",
    
#     "Can you please send me the project report before evening? I need to review it before submission.",
    
#     "URGENT: Your loan has been approved instantly. Submit your PAN details now to receive the amount.",
    
#     "Hey, are we still meeting for lunch today at 1 PM near the office?"
# ]

# print(model.predict(emails))