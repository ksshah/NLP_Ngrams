import math
from nltk.corpus import stopwords 
from string import punctuation
import re
import json
#from sklearn.metrics import accuracy_score

class Smoothed_NGram:
    def trainModel(self,corpora):
        train_corpora=open(corpora,"r")
        contents_train=train_corpora.read().lower().split()
        
        vocab=dict()
        
        corpus_length=0
        for word in contents_train:
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word]=1
            corpus_length+=1
            
        #Unknown word handling for unigram
        oldKey=''
        oldCount=0
        
        for i,j in sorted(vocab.items(),key=lambda p:p[1]):
            oldKey=i
            oldCount=j
            break
        
        vocab['unk']=oldCount
        del vocab[oldKey]
            
        listOfTrainingBigrams = dict()
        
        for i in range(len(contents_train)-1):
            
            if (contents_train[i], contents_train[i+1]) in listOfTrainingBigrams.keys():
                listOfTrainingBigrams[(contents_train[i], contents_train[i + 1])] += 1
            else:
                listOfTrainingBigrams[(contents_train[i], contents_train[i + 1])] = 1
                
        return listOfTrainingBigrams,vocab,corpus_length

    #True corpus
    def smoothed_ngram(self,n,listOfTrainingBigrams,vocab,corpus_length,testSet):
        
        contents_test=testSet.lower().split()
        
        #alpha is the smoothing factor
        
        alpha=0.01
        
        len_vocab=len(vocab)
        q=0
        
        if n==1:
            for j in contents_test:
                if j not in vocab:
                    j='unk'
                q += math.log(float((vocab[j]+alpha)/(corpus_length+(alpha*len_vocab))))
            return float(math.exp(q/100)-math.exp(-100))
        
        if n==2:
            
            finalprobabilityLog=0.0
            listOfTestingBigrams=dict()
            
            for i in range(len(contents_test)-1):
                if (contents_test[i], contents_test[i+1]) in listOfTestingBigrams:
                    listOfTestingBigrams[(contents_test[i], contents_test[i + 1])] += 1
                else:
                    listOfTestingBigrams[(contents_test[i], contents_test[i + 1])] = 1
            
            for bigram in listOfTestingBigrams:
                word1 = bigram[0]
                
                if(bigram not in listOfTrainingBigrams):
                    listOfTrainingBigrams[bigram]=0
                if(word1 not in vocab):
                    vocab[word1]=0
                
                if((bigram in listOfTrainingBigrams) and (word1 in vocab)):
                    finalprobabilityLog += math.log(float((listOfTrainingBigrams[bigram]+alpha)/(vocab[word1]+(alpha*len_vocab))))
                else:
                    finalprobabilityLog += math.log(float(alpha / (vocab[word1]+(alpha*len_vocab))))
            #print(math.exp(finalprobabilityLog/100)-math.exp(-100))
            return float(math.exp(finalprobabilityLog/100)-math.exp(-100))
            
                

    def calculatePerplexity(self,p,n):
        if p==0:
            return math.inf
        return math.pow(float(1/p),float(1/n))        
          
    def opininSpamClassifier(self,train_corpora_truthful,train_corpora_deceptive,test_con):
        test_outputs=[]
        trainingBigramCounts_truthful, unigramCounts_truthful, corpus_length_truthful = self.trainModel(train_corpora_truthful)
        trainingBigramCounts_deceptive, unigramCounts_deceptive, corpus_length_deceptive = self.trainModel(train_corpora_deceptive)
        for sentence in test_con:
            x=self.calculatePerplexity(self.smoothed_ngram(1,trainingBigramCounts_truthful,unigramCounts_truthful, corpus_length_truthful, sentence),1)
            y=self.calculatePerplexity(self.smoothed_ngram(1,trainingBigramCounts_deceptive,unigramCounts_deceptive,corpus_length_deceptive,sentence),1)
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

    spamClassifier=Smoothed_NGram()
    
    #pass  the files to the spam classifier method
    test=open(data['files']['test'],"r")
    test_con=test.read().splitlines()
    
    test_outputs=spamClassifier.opininSpamClassifier(train_corpora_truthful,train_corpora_deceptive,test_con)
    
    print(test_outputs)
    
    
if __name__ == '__main__':
    main()


     
        

