import os
import pandas as pd

from sklearn.datasets import load_files


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


if __name__ == '__main__':
    print(load_files('./data/foo', categories=['a', 'blums']))
    #df = load_training()
    #print(df.head())
