import json
import os
import re
import pandas as pd

from sklearn.datasets import load_files
from typing import List


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
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    # Remove HTML tags
    # clean = re.compile('<.*?>')
    # s = re.sub(clean, ' ', s)

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
    s = s.split()
    s = [stemmer.lemmatize(word) for word in s]
    s = ' '.join(s)

    return s


def prepare_texts(documents: List[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    from nltk.corpus import stopwords

    vectorizer = CountVectorizer(max_features=500, min_df=0, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(2,2))
    X = vectorizer.fit_transform(documents).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()

    return X


def random_forest(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    return metrics(y_test, y_pred)


def random_forest_split(x_train, y_train):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    return random_forest(x_train, y_train, x_test, y_test)


def logistic_regression(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression

    reg = LogisticRegression(solver='lbfgs', max_iter = 1000, multi_class='multinomial', random_state=0)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    return metrics(y_test, y_pred)


def svc(x_train, y_train, x_test, y_test):
    from sklearn.svm import SVC

    model = SVC(gamma='auto')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return metrics(y_test, y_pred)


def metrics(y_test, y_pred) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    confusion_matrix_value = confusion_matrix(y_test, y_pred).tolist()
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


def finalize_metrics(m):
    if not os.path.exists('model'):
        os.makedirs('model')

    print(json.dumps(m, indent=2))

    with open('model/metrics.json', 'w') as fp:
        json.dump(m, fp, indent=2)


def run():
    import nltk

    # download nltk data
    nltk.download('stopwords')
    nltk.download('wordnet')

    print('... loading data')
    training_set = load_files("./data/train", categories=['pos', 'neg'])
    test_set = load_files("./data/test", categories=['pos', 'neg'])
    documents_train, y_train = training_set.data, training_set.target
    documents_test, y_test = test_set.data, test_set.target

    print('... prepare')
    documents_train = [prepare_text(c.decode('utf-8')) for c in documents_train]
    documents_test = [prepare_text(c.decode('utf-8')) for c in documents_test]

    print('... feature extraction')
    x_train = prepare_texts(documents_train)
    x_test = prepare_texts(documents_test)

    print('... train and predict')
    finalize_metrics({
        #"rf_metrics": random_forest(x_train, y_train, x_test, y_test),
        #"rf_split_metrics": random_forest_split(x_train, y_train)
        "lr_metrics": logistic_regression(x_train, y_train, x_test, y_test)
        #"svc_metrics": svc(x_train, y_train, x_test, y_test)
    })

    print('... done')
    print()


if __name__ == '__main__':
    run()