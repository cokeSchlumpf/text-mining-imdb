import json
import os
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.datasets import load_files
from typing import List
from nltk.corpus import stopwords
from nltk import pos_tag
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stopwords = stopwords.words('english')


def load_data(directory_path: str) -> pd.DataFrame:
    """
    Loads all text files from a provided directory into a Pandas Data Frame containing the text and the filename.

    :param directory_path: The directory to read.
    :return:
    """
    if not os.path.isdir(directory_path):
        print("Bloody idiot - Provide a directory!")

    result = pd.DataFrame(columns=['text', 'filename'])

    for filename in os.listdir(directory_path):
        path = os.path.join(directory_path, filename)

        if os.path.isdir(path):
            result = result.append(load_data(path), ignore_index=True, sort=False)
        if filename.endswith("txt"):
            with open(path) as f:
                text = f.read()
                current_df = pd.DataFrame({'text': [text], 'filename': path})
                result = result.append(current_df, ignore_index=True, sort=False)

    return result


def load_training() -> pd.DataFrame:
    positive = load_data('./data/train/pos')
    negative = load_data('./data/train/neg')
    positive['sentiment'] = 1
    negative['sentiment'] = 0

    return positive.append(negative, ignore_index=True, sort=False)


def prepare_text(s: str) -> str:
    # Remove all the special characters
    s = re.sub(r'\W', ' ', s)
    # remove all single characters
    s = re.sub(r'\s+[a-zA-Z]\s+', ' ', s)
    # Remove single characters from the start
    s = re.sub(r'\^[a-zA-Z]\s+', ' ', s)
    # Substituting multiple spaces with single space
    s = re.sub(r'\s+', ' ', s, flags=re.I)
    # Converting to Lowercase
    s = s.lower()
    # Lemmatization
    s = re.sub('<[^>]*>', '', s)
    s = re.sub(r'[^\w\s]', '', s)

    s = [w for w in word_tokenize(s) if len(w) > 2 and w not in stopwords]
    s = [WordNetLemmatizer().lemmatize(w) for w in s]

    s = s + [w[1] for w in pos_tag(s)]
    return s


def prepare_texts(documents: List[str]):


    vectorizer = CountVectorizer(max_features=1500, min_df=15, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(documents).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    return X


def random_forest(x_train, y_train, x_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(x_train, y_train)
    return clf.predict(x_test)


def metrics(y_test, y_pred) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    confusion_matrix_value = confusion_matrix(y_test, y_pred).toList()
    classification_report_value = classification_report(y_test, y_pred, output_dict=True)
    accuracy_score_value = accuracy_score(y_test, y_pred)

    print(type(confusion_matrix_value))
    print(type(classification_report_value))
    print(type(accuracy_score_value))

    return {
        'confusion_matrix': confusion_matrix_value,
        'classification_report': classification_report_value,
        'accuracy_score': accuracy_score_value
    }


def run():
    print('... loading data')
    training_set = load_files(r"../data/train", categories=['pos', 'neg'])
    test_set = load_files(r"../data/test", categories=['pos', 'neg'])
    documents_train, y_train = training_set.data, training_set.target
    documents_test, y_test = test_set.data, test_set.target

    print('... prepare')
    documents_train = [prepare_text(c.decode('utf-8')) for c in documents_train]
    documents_test = [prepare_text(c.decode('utf-8')) for c in documents_test]

    print('... feature extraction')
    x_train = prepare_texts(documents_train)
    x_test = prepare_texts(documents_test)

    file_x_train = open('x_train', 'rb')
    pickle.dump(x_train, file_x_train)
    file_x_test = open('x_test', 'rb')
    pickle.dump(x_test, file_x_test)


    #documents_train = pickle.load(file_documents_train)
    #documents_test = pickle.load(file_documents_test)

    print('... train and predict')
    y_pred = random_forest(x_train, y_train, x_test)
    m = metrics(y_test, y_pred)

    print('... done')
    print()

    print(json.dumps(m, indent=2))

    with open('model/metrics.json', 'w') as fp:
        json.dump(m, fp, indent=2)


if __name__ == '__main__':
    run()
