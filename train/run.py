import json
import os
import re
import pandas as pd

from sklearn.datasets import load_files
from typing import List


REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def initialize_nltk():
    import nltk

    # download nltk data
    nltk.download('stopwords')
    nltk.download('wordnet')


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


def read_data(directory: str):
    data = load_files(directory, categories=['pos', 'neg'])
    documents, y = data.data, data.target
    documents = [prepare_text_simplified(c.decode('utf-8')) for c in documents]

    return documents, y


def prepare_text(s: str) -> str:
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()

    # Remove HTML tags
    # clean = re.compile('<.*?>')
    # s = re.sub(clean, ' ', s)

    # s = re.sub(REPLACE_NO_SPACE, '', s)
    # s = re.sub(REPLACE_WITH_SPACE, ' ', s)

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


def prepare_text_simplified(s: str) -> str:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer

    #english_stop_words = stopwords.words('english')
    english_stop_words = stop_words = ['in', 'of', 'at', 'a', 'the']
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # remove stopwords
    s = s.lower()
    s = re.sub(REPLACE_NO_SPACE, '', s)
    s = re.sub(REPLACE_WITH_SPACE, ' ', s)

    s = re.sub(r'\W', ' ', s)
    s = re.sub(r'\s+[a-zA-Z]\s+', ' ', s)
    s = re.sub(r'\^[a-zA-Z]\s+', ' ', s)
    s = re.sub(r'\s+', ' ', s, flags=re.I)

    #s = str.join(' ', s.split()[:150])
    s = ' '.join([word for word in s.split() if word not in english_stop_words])
    s = ' '.join([stemmer.stem(word) for word in s.split()])
    s = ' '.join([lemmatizer.lemmatize(word) for word in s.split()])
    #s = str.join(' ', s.split()[:100])

    return s


def prepare_texts(documents: List[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    from nltk.corpus import stopwords

    vectorizer = CountVectorizer(max_features=500, min_df=0, max_df=0.7, stop_words=stopwords.words('english'), ngram_range=(1,3))
    X = vectorizer.fit_transform(documents).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X)

    return X


def prepare_texts_binary(documents: List[str]):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    cv = CountVectorizer(max_features=500, binary=True, ngram_range=(1,3))
    cv.fit(documents)

    x = cv.transform(documents).toarray()
    tfidfconverter = TfidfTransformer()
    x = tfidfconverter.fit_transform(x)

    return x


def prepare_texts_tfidf(documents: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), binary=True)
    tfidf_vectorizer.fit(documents)
    x = tfidf_vectorizer.transform(documents)

    return x


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
    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75)

    reg = LogisticRegression(C=0.25, max_iter=500, solver='lbfgs', multi_class='multinomial', random_state=0)
    reg.fit(x_train, y_train)

    print('... training done.')

    y_val_pred = reg.predict(x_val)
    y_test_pred = reg.predict(x_test)

    return {
        'validation': metrics(y_val, y_val_pred),
        'test': metrics(y_test, y_test_pred)
    }


def logistic_regression_optimized(x_train, y_train, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75)

    param_grid = {
        'C': [0.25, 0.5, 0.75, 1],
        'max_iter': [500]
    }

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=0)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=10)

    print('... Logistic Regression with Grid Search')
    grid.fit(x_train, y_train)
    print(f"... Grid Search - Best parameters:\n{grid.best_params_}")

    y_val_pred = grid.best_estimator_.predict(x_val)
    y_test_pred = grid.best_estimator_.predict(x_test)

    return {
        'validation': metrics(y_val, y_val_pred),
        'test': metrics(y_test, y_test_pred)
    }


def svc(x_train, y_train, x_test, y_test):
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.75)

    model = LinearSVC(C=0.1)
    model.fit(x_train, y_train)

    print('... training done.')

    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    return {
        'validation': metrics(y_val, y_val_pred),
        'test': metrics(y_test, y_test_pred)
    }


def metrics(y_test, y_pred) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    confusion_matrix_value = confusion_matrix(y_test, y_pred).tolist()
    classification_report_value = classification_report(y_test, y_pred, output_dict=True)
    accuracy_score_value = accuracy_score(y_test, y_pred)

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
    initialize_nltk()
    print('... loading and prepare data')
    documents_train, y_train = read_data('./data/train')
    documents_test, y_test = read_data('./data/test')

    [print(f"{d}\n\n--\n\n") for d in documents_train[:3]]

    print('... feature extraction')
    x_all = prepare_texts_tfidf(documents_train + documents_test)
    x_train = x_all[:len(documents_train)]
    x_test = x_all[len(documents_train):]

    print('... train and predict')
    finalize_metrics({
        #"rf_metrics": random_forest(x_train, y_train, x_test, y_test),
        #"rf_split_metrics": random_forest_split(x_train, y_train)
        #"lr_metrics": logistic_regression(x_train, y_train, x_test, y_test),
        #"lr_optimized": logistic_regression_optimized(x_train, y_train, x_test, y_test),
        "svc_metrics": svc(x_train, y_train, x_test, y_test)
    })

    print('... done')
    print()


if __name__ == '__main__':
    run()