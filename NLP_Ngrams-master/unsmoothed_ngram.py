import math
from nltk.corpus import stopwords 
from string import punctuation
import re
import json

class Unsmoothed_NGram:
    
    def trainModel(self,corpora):
        #Open the file
        train_corpora=open(corpora,"r")
        
        #Convert all the data to lower case so that upper case doesn't have a different identuty from the lowercase words
        contents_train=train_corpora.read().lower()
        
        #remove non alphabetical characters from the content 
        contents_train = re.sub('[^A-Za-z]', ' ', contents_train)
        
        #Remove punctuations
        contents_train=contents_train.translate(str.maketrans('','',punctuation)).split()
        
        #Remove stop words from the content (eg. the, an, a)
        stop_words = set(stopwords.words('english')) 
        contents_train = [w for w in contents_train if not w in stop_words]
        
        #create a vocabulary of all the words in the corpora
        #Corpus length determines the length of the total words
        #Vocabulary is the list of unique words
        vocab=dict()
        corpus_length=0
        for word in contents_train:
            if word in vocab:
                vocab[word]+=1
        else:
            vocab[word]=1
        corpus_length+=1
        
        #Bigrams is the combination of 2 words. We count the occurrence of unique pairs of words and return them
        listOfTrainingBigrams = dict()
        
        for i in range(len(contents_train)-1):
            if (contents_train[i], contents_train[i+1]) in listOfTrainingBigrams.keys():
                listOfTrainingBigrams[(contents_train[i], contents_train[i + 1])] += 1
            else:
                listOfTrainingBigrams[(contents_train[i], contents_train[i + 1])] = 1
        
        return listOfTrainingBigrams,vocab,corpus_length
        
        
        
    def unsmoothed_ngram(self,n,listOfTrainingBigrams,vocab,corpus_length,test):
    
        #Pre-process the test data same as training data
        contents_test=test.lower() 
        
        contents_test = re.sub('[^A-Za-z]', ' ', contents_test)  
        contents_test=contents_test.translate(str.maketrans('','',punctuation)).split()
        stop_words = set(stopwords.words('english')) 
        contents_test = [w for w in contents_test if not w in stop_words]       
        
        
        #q here denotes the log of probabilities. Log so that we can handle underflow
        q=0
        
        #If we use unigram language modelling, we use n=1
        if n==1:
            for j in contents_test:
            #if the word is in the vocabulary, we calculate its probability as prob=(word_count/corpus_length) and take log 
            #to control the underflow.
                if j in vocab:
                    q += math.log((vocab[j]*1.0/corpus_length))
                #Since this is the unsmoothed version, if the word is not present in the vocabulary, we simply return 0 probability
                else:
                    return 0.0
                    break
            #Return antilog of finalprobabilityLog as probability is currently in the log form. 
            #The following is done because if we simply return q, we get 0 because of the extremely small value.
            return float(math.exp(q))
            #return float(math.exp(q/100)-math.exp(-100))
        
        #if we use bigram language modelling, we use n=2
        if n==2:
            #Create list of testing bigrams
            listOfTestingBigrams=dict()
        
            for i in range(len(contents_test)-1):
                if (contents_test[i], contents_test[i+1]) in listOfTestingBigrams:
                    listOfTestingBigrams[(contents_test[i], contents_test[i + 1])] += 1
                else:
                    listOfTestingBigrams[(contents_test[i], contents_test[i + 1])] = 1
            
            #To calculate the final probability
            finalprobabilityLog=0.0
            
            for bigram in listOfTestingBigrams:
                word1 = bigram[0]
            
                if((bigram in listOfTrainingBigrams) and (word1 in vocab)):
                    finalprobabilityLog += math.log(float(listOfTrainingBigrams[bigram] / vocab[word1]))
                else:
                    print(bigram)
                    return 0.0
                    break
            #Return antilog of finalprobabilityLog as probability is currently in the log form
            return float(math.exp(finalprobabilityLog))    
    
    def calculatePerplexity(self,p,n):
        if p==0:
            return math.inf
        return math.pow(float(1/p),float(1/n))
        
    def opininSpamClassifier(self,train_corpora_truthful,train_corpora_deceptive,test):
        #Open the file
        test_content=open(test,"r")
        #Since each review is seperated by \n, we use splitlines to get the individual reviews.
        test_con=test_content.read().splitlines()
        
        test_outputs=[]
        trainingBigramCounts_truthful, unigramCounts_truthful, corpus_length_truthful = self.trainModel(train_corpora_truthful)
        trainingBigramCounts_deceptive, unigramCounts_deceptive, corpus_length_deceptive = self.trainModel(train_corpora_deceptive)
        
        #perplexity is calculated as 1/probability. We calculate perplexity for both truthful and deceptive. 
        #Whichever has lower perplexity(higher probability), that label is tagged to the test sentence.
        for sentence in test_con:
            x=self.calculatePerplexity(self.unsmoothed_ngram(2,trainingBigramCounts_truthful,unigramCounts_truthful, corpus_length_truthful, sentence),2)
            y=self.calculatePerplexity(self.unsmoothed_ngram(2,trainingBigramCounts_deceptive,unigramCounts_deceptive,corpus_length_deceptive,sentence),2)
            print(x,y)
            if x<y:
                test_outputs.append(0)
            else:
                test_outputs.append(1)
        return test_outputs
        
    


def main():   
	#Read the config file to get the data
    with open("config.json") as json_data_file:
        data = json.load(json_data_file)
    
    #read the file names for the training set and test set
    train_corpora_truthful = data['files']['truthful_train']
    train_corpora_deceptive = data['files']['deceptive_train']
    test=data['files']['test']

    spamClassifier=Unsmoothed_NGram()
    
    #pass  the files to the spam classifier method
    test_outputs=spamClassifier.opininSpamClassifier(train_corpora_truthful,train_corpora_deceptive,test)
    
    print(test_outputs)
    
if __name__ == '__main__':
    main()

    
     
        


