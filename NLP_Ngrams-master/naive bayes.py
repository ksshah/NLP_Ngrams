import pandas
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import json

def main():     
    #Read the config file to get the data
    with open("config.json") as json_data_file:
        data = json.load(json_data_file)
    
    #read the file names for the training set and test set
    truthful_file = open(data['files']['truthful_train'],"r")
    truthful_data=truthful_file.read()
    truth_training_data=truthful_data.splitlines()
    train_data_with_labels=[[truth_training_data[i],0] for i in range(0,len(truth_training_data))]
    
    deceptive_file = open(data['files']['deceptive_train'],"r")
    deceptive_data=deceptive_file.read()
    deceptive_training_data=deceptive_data.splitlines()
    for i in range(0,len(deceptive_training_data)):
        train_data_with_labels.append([deceptive_training_data[i],1])
    
    #For the purpose of parameter tuning, I am using the validation files. For the final testing pupose, we have used the test files  
    validation_file=open(data['files']['validation'],"r")
    validation_data=validation_file.read()
    validation_training_data=validation_data.splitlines()
    for i in range(0,len(validation_training_data)):
        test_data_with_labels=[[validation_training_data[i],0] for i in range(0,len(validation_training_data))]

    train_df=pandas.DataFrame(train_data_with_labels)
    train_df.columns = ['Text', 'Truth Value'] 

    test_df=pandas.DataFrame(test_data_with_labels)
    test_df.columns = ['Text', 'Truth Value']

    train_df['Text']=[entry.lower() for entry in train_df['Text']]
    test_df['Text']=[entry.lower() for entry in test_df['Text']]

    #Commented as only one approach is used for final calculation  
    '''
    cv=CountVectorizer(max_features=8000)
    Train_X_Tfidf=cv.fit_transform(train_df['Text'])
    Test_X_Tfidf= cv.transform(test_df['Text'])
    '''

    Tfidf_vect = TfidfVectorizer(max_features=8000)
    Tfidf_vect.fit(train_df['Text'])

    Train_X_Tfidf = Tfidf_vect.transform(train_df['Text'])
    Test_X_Tfidf= Tfidf_vect.transform(test_df['Text'])

    # fit the training dataset on the NB classifier
    Naive = MultinomialNB()
    Naive.fit(Train_X_Tfidf,train_df['Truth Value'])
    # predict the labels on validation dataset
    predictions_NB = Naive.predict(Test_X_Tfidf)
    print(predictions_NB)
    print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, test_df['Truth Value'])*100)

if __name__ == '__main__':
    main()