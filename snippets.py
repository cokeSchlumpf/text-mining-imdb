import pickle
import pandas as pd
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

connectionstring = "host=localhost port=5433 dbname=cbdb user=cb password=cemile"


def write_blob(object_ba, object_name, object_desc):
    ba = pickle.dumps(object_ba)
    conn = None
    try:
        conn = psycopg2.connect(connectionstring)
        cur = conn.cursor()
        cur.execute("INSERT INTO cbdata(object_ba, object_name, object_bezeichnung) VALUES(%s,%s,%s)",
                    (psycopg2.Binary(ba), object_name, object_desc))
        conn.commit()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def read_blob(object_name):
    conn = None
    try:
        conn = psycopg2.connect(connectionstring)
        cur = conn.cursor()
        cur.execute("SELECT object_ba FROM cbdata WHERE object_name = %s", (object_name,))
        blob = cur.fetchone()
        cur.close()
        return pickle.loads(blob[0])
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def check_pipelines():
    c = psycopg2.connect(connectionstring)
    df = pd.read_sql_query("select review, ispos from reviews", c)
    train, test = train_test_split(df, test_size=0.5)
    tfidf = TfidfVectorizer()
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
    parameters = {
        "clf__C": [1.0, 10.0]
    }
    grid = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    grid.fit(train['review'], train['ispos'])
    print("score = %3.2f" % (grid.score(test['review'], test['ispos'])))
    print(grid.best_params_)


def checkdb():
    c = None
    try:
        c = psycopg2.connect(connectionstring)
        cur = c.cursor()
        cur.execute('SELECT 1')
        ret = "Connected successfully to database"
    except psycopg2.Error as e:
        ret = "Exception on connection to database"
    finally:
        if c is not None:
            c.close()
    return ret


def currentmodelversion():
    conn = None
    try:
        conn = psycopg2.connect(connectionstring)
        cur = conn.cursor()
        cur.execute("select max(object_id) id from cbdata where object_name = 'logit-model' ")
        id = cur.fetchone()[0]
        cur.close()
        return "Model ID: " + str(id)
    except (Exception, psycopg2.DatabaseError) as error:
        return error
    finally:
        if conn is not None:
            conn.close()


def prepare_reg_model_1():
    c = psycopg2.connect(connectionstring)
    df = pd.read_sql_query("select review, ispos from reviews", c)
    tfidf = TfidfVectorizer()
    tfidf.fit(df['review'])

    train, test = train_test_split(df, test_size=0.5)

    x_trainv = tfidf.transform(train['review'])
    x_testv = tfidf.transform(test['review'])

    y_train = train['ispos']
    y_test = test['ispos']

    # train
    c_values = [0.001, 0.2, 0.7, 1.5, 10]
    solver_list = ['lbfgs', 'liblinear']

    for c in c_values:
        for s in solver_list:
            m = LogisticRegression(C=c, random_state=0, solver=s, n_jobs=-1)
            m.fit(x_trainv, y_train)
            m.predict(x_testv)
            score = m.score(x_testv, y_test)
            print(score)


def prepare_reg_model():
    c = None
    try:
        c = psycopg2.connect(connectionstring)
        train = pd.read_sql_query("select review, ispos from reviews where istest = 0", c)
        X_train = train['review']
        y_train = train['ispos']

        tfidf = TfidfVectorizer()
        X_train_v = tfidf.fit_transform(X_train)
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        clf.fit(X_train_v, y_train)

        write_blob(tfidf, 'tfidf', 'the object dump of the vectorizer')
        write_blob(clf, 'logit-model', 'the logistic regression model')
        ret = "Vectorizer and Model saved successfully! "
    except (Exception, psycopg2.DatabaseError) as error:
        ret = "Exception: " + error
    finally:
        if c is not None:
            c.close()
    return ret


def predict(review):
    try:
        tfidf = read_blob('tfidf')
        clf = read_blob('logit-model')
        if clf is None or tfidf is None:
            ret = 'Model / Vector not found!'
        else:
            ret = clf.predict(tfidf.transform([review]))
    except Exception as error:
        ret = "Prediction error " + error
    return "positive" if ret[0] == 1 else "negative"


if __name__ == '__main__':
    check_pipelines()