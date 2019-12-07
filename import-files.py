import os
import unicodecsv as csv
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer


def parse_folder(path):
    counter = 0
    ispos = path.endswith("pos")
    istest = path.split(os.sep)[-2] == 'test'
    stop_words = set(stopwords.words('english'))
    strispos = "pos" if ispos else "neg"
    stristest = "test" if istest else "train"
    filename = "../processed-data-" + strispos + "-" + stristest

    with open(filename, mode='wb') as datafile:
        datawriter = csv.writer(datafile)
        datawriter.writerow(["index", "review-raw", "review", "postags", "movieid", "rating"])

        for subdir, dirs, files in os.walk(path):
            for file in files:
                m = re.search('(.+)_(.+).txt', file)
                if m:
                    f = open(os.path.join(path, file), 'r', encoding="utf8")
                    s = f.read()
                    s1 = s.lower()
                    s1 = re.sub('<[^>]*>', '', s1)
                    s1 = re.sub(r'[^\w\s]', '', s1)

                    s1 = [w for w in word_tokenize(s1) if len(w) > 2 and w not in stop_words]
                    s1 = [WordNetLemmatizer().lemmatize(w) for w in s1]
                    pt = [w[1] for w in pos_tag(s1)]

                    s1 = ' '.join(s1)
                    pt = ' '.join(pt)

                    datawriter.writerow([counter, s, s1, pt, m.group(1), m.group(2)])

                    counter += 1
                    print(counter)


if __name__ == '__main__':
    p = r'../data'
    parse_folder(os.path.join(p, 'test', 'pos'))
    parse_folder(os.path.join(p, 'test', 'neg'))
    parse_folder(os.path.join(p, 'train', 'pos'))
    parse_folder(os.path.join(p, 'train', 'neg'))
