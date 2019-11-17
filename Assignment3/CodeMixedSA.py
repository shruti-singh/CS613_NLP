import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
import numpy as np
import joblib
from joblib import dump, load

class TextCleaner:

    vowels = ['a', 'e', 'i', 'o', 'u']
    
    def clean_token(self, token, thresh=2):
        token_and_lang = token.rsplit("_", 1)
        token = token_and_lang[0]
        lang = token_and_lang[1]

        if len(token)>thresh:
            if lang == "Hin":
                token = self.clean_repeated_chars(token)
                token = self.remove_vowels(token)
            else:
                token = self.clean_repeated_chars(token)
        return token

    def clean_repeated_chars(self, string):
        return re.sub(r'(.)\1+', r'\1\1', string)     

    def remove_vowels(self, string):
        return ''.join([l for l in string.lower() if l not in self.vowels])


def prepare_train_data():

    tcleaner = TextCleaner()

    f_train = open("/home/shruti/Desktop/iitgn/courses/nlp/Assignment/A3/nlp assignment 3/train.txt", "r")
    lines = f_train.read().lower().split("\n")
    f_train.close()
    train_dict = {}
    sentiment = {"negative": 0, "neutral": 1, "positive": 2}
    tweet = []
    uid = None

    count = 0

    for line in lines:
        count += 1
        line = line.split("\t")
        if line[0].strip() == "meta":
            prev_uid = uid
            if prev_uid:
                train_dict[prev_uid]["text"] = " ".join(tweet)
            uid = line[1].strip()
            s = line[2].strip().lower()
            tweet = []
            if s in sentiment:
                train_dict[uid] = {"text": "", "senti": sentiment[s]}
            else:
                print("Look into this", uid)
                print(line)
        else:
            tok = tcleaner.clean_token(line[0].lower().strip()+"_"+line[1].lower().strip())
            # tok = line[0].lower().strip()
            tweet.append(tok)
        # if count > 30000:
        #     break
    train_dict[uid]["text"] = " ".join(tweet)

    print("Len of train: ", len(train_dict))
    with open("train_dict.pkl", "wb") as f:
        pickle.dump(train_dict, f)
    return train_dict


def prepare_test_data():
    
    tcleaner = TextCleaner()

    f_train = open("/home/shruti/Desktop/iitgn/courses/nlp/Assignment/A3/nlp assignment 3/test.txt", "r")
    lines = f_train.read().lower().split("\n")
    f_train.close()
    test_dict = {}
    sentiment = {"negative": 0, "neutral": 1, "positive": 2}
    tweet = []
    uid = None
    for line in lines:
        line = line.split("\t")
        if line[0].strip() == "meta":
            prev_uid = uid
            if prev_uid:
                test_dict[prev_uid]["text"] = " ".join(tweet)
            uid = line[1].strip()
            s = line[2].strip().lower()
            tweet = []
            if s in sentiment:
                test_dict[uid] = {"text": "", "senti": sentiment[s]}
            else:
                print("Looks into this", uid)
                print(line)
        else:
            tok = tcleaner.clean_token(line[0].lower().strip()+"_"+line[1].lower().strip())
            # tok = line[0].lower().strip()
            tweet.append(tok)
    test_dict[uid]["text"] = " ".join(tweet)

    with open("test_dict.pkl", "wb") as f:
        pickle.dump(test_dict, f)
    return test_dict

def parse_train_test_data():
    trd = prepare_train_data()
    ted = prepare_test_data()
    # with open('train_dict.pkl', 'rb') as f:
    #     trd = pickle.load(f)
    # with open('test_dict.pkl', 'rb') as f:
    #     ted = pickle.load(f)
    print("Data parsed....")
    return trd, ted


class LogisticRegressionModel:

    cvec = None
    tfidf_transf = None
    model = None

    def __init__(self):
        self.model = LogisticRegression()
        return

    def fit(self, trd):
        
        docs = []
        y_values = []
        for key in trd:
            docs.append(trd[key]["text"])
            y_values.append(trd[key]["senti"])

        # self.cvec = CountVectorizer(analyzer="char_wb")
        self.cvec = CountVectorizer()
        X = self.cvec.fit_transform(docs)
        df = pd.DataFrame(X.toarray(), columns=self.cvec.get_feature_names())

        self.tfidf_transf = TfidfTransformer()
        X_tfidf = self.tfidf_transf.fit_transform(df)

        y = np.array(y_values)

        self.model.fit(X_tfidf, y)
        y_pr = self.model.predict(X_tfidf)
        # print("ON TRAIN SET:")
        # print("Accuracy: ", accuracy_score(y, y_pr))
        # print("F1 score: ", f1_score(y, y_pr, average='macro'))

        joblib.dump(self.model, 'char_logistic_regression_model.joblib')
        joblib.dump(self.cvec, 'cvec.joblib')
        joblib.dump(self.tfidf_transf, 'tfidftrans.joblib')

        return

    def predict(self, ted):

        test_docs = []
        y_values_test = []

        for key in ted:
            test_docs.append(ted[key]["text"])
            y_values_test.append(ted[key]["senti"])

        X = self.cvec.transform(test_docs)
        df = pd.DataFrame(X.toarray(), columns=self.cvec.get_feature_names())
        X_tfidf_test = self.tfidf_transf.transform(df)

        y = np.array(y_values_test)

        y_pr = self.model.predict(X_tfidf_test)
        print("ON TEST SET:")
        print("Accuracy: ", accuracy_score(y, y_pr))
        print("F1 score: ", f1_score(y, y_pr, average='macro'))
        print(classification_report(y, y_pr))

        return


class SVMModel:

    model = None
    cvec = None
    tfidf_transf = None

    def __init__(self):
        self.model = SVC(kernel='linear')
        return
    
    def train(self, trd):
        docs = []
        y_values = []
        for key in trd:
            docs.append(trd[key]["text"])
            y_values.append(trd[key]["senti"])

        # self.cvec = CountVectorizer(analyzer="char_wb")
        self.cvec = CountVectorizer()
        X = self.cvec.fit_transform(docs)
        df = pd.DataFrame(X.toarray(), columns=self.cvec.get_feature_names())

        self.tfidf_transf = TfidfTransformer()
        X_tfidf = self.tfidf_transf.fit_transform(df)

        y = np.array(y_values)

        self.model.fit(X_tfidf, y)
        y_pr = self.model.predict(X_tfidf)

        # print("ON TRAIN SET:")
        # print("Accuracy: ", accuracy_score(y, y_pr))
        # print("F1 score: ", f1_score(y, y_pr, average='macro'))

        joblib.dump(self.model, 'char_svm_model.joblib')
        joblib.dump(self.cvec, 'cvec.joblib')
        joblib.dump(self.tfidf_transf, 'tfidftrans.joblib')

        return

    def predict(self, ted):

        test_docs = []
        y_values_test = []

        for key in ted:
            test_docs.append(ted[key]["text"])
            y_values_test.append(ted[key]["senti"])

        X = self.cvec.transform(test_docs)
        df = pd.DataFrame(X.toarray(), columns=self.cvec.get_feature_names())
        X_tfidf_test = self.tfidf_transf.transform(df)

        y = np.array(y_values_test)

        y_pr = self.model.predict(X_tfidf_test)

        print("ON TEST SET:")
        print("Accuracy: ", accuracy_score(y, y_pr))
        print("F1 score: ", f1_score(y, y_pr, average='macro'))
        print(classification_report(y, y_pr))

        return

if __name__ == "__main__":
    train, test = parse_train_test_data()

    lr_model = LogisticRegressionModel()
    lr_model.fit(train)
    lr_model.predict(test)

    svm_model = SVMModel()
    svm_model.train(train)
    svm_model.predict(test)