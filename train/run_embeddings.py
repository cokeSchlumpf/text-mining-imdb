from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding

from train.run import read_data

EMBEDDING_DIM = 190


def main():
    #
    #
    #
    print('... load and prepare data')
    documents_train, y_train = read_data('./data/train')
    documents_train, documents_validate, y_train, y_validate = train_test_split(documents_train, y_train, train_size=0.75)
    documents_test, y_test = read_data('./data/test')
    documents_all = documents_train + documents_test

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(documents_all)

    max_length = max([len(s.split()) for s in documents_all])
    vocab_size = len(tokenizer.word_index) + 1

    x_train_tokens = tokenizer.texts_to_sequences(documents_train)
    x_test_tokens = tokenizer.texts_to_sequences(documents_test)
    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding='post')
    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_length, padding='post')

    #
    #
    #
    print('... build model')

    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('... train model')
    model.fit()


if __name__ == '__main__':
    main()
