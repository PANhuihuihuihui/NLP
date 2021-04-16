import os
import pathlib
import sys
import re

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))


def read_samples(filename):
    """Read the data file and return a sample list.
    """
    samples = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            samples.append(line.strip())
    return samples


def write_samples(samples, file_path, opt='w'):
    """Write the samples into a file.
    """
    with open(file_path, opt, encoding='utf8') as file:
        for line in samples:
            file.write(line)
            file.write('\n')


def partition(samples,list_c):
    """Partition a whole sample set into training set, dev set and test set.

    """
    train, dev, test = [], [], []
    index = 0
    count = list_c[index]
    for sample in samples:
        if count == 0:
            index+=1
            count = list_c[index]
            print('train: ', len(train),"dev: ",len(dev),"test: ",len(test))
        else:
            count-=1
        if count >= int(list_c[index]*0.9):  # Test set size.
            test.append(sample)
        elif count >= int(list_c[index]*0.8):  # Dev set size.
            dev.append(sample)
        else:
            train.append(sample)
    write_samples(train, 'files/train.txt')
    write_samples(dev, 'files/dev.txt')
    write_samples(test, 'files/test.txt')


def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',str(text))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', str(text))


def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',str(text))

def remove_punctuation(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
def final_preprocess(text):
    text = text.replace('\\r', ' ')
    text = text.replace('\\"', ' ')
    text = text.replace('\\n', ' ')
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(e for e in text.split() if e.lower() not in stopwords)
    text = text.lower()
    ps = PorterStemmer()
    text = ps.stem(text)
    return text
def load_pretrain_emb(path):
    file_r = codecs.open(path, "rb", "utf-8")
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(" "))
    embedding = dict()
    line = file_r.readline()
    while line:
        items = line.split(" ")
        item = items[0]

        try:
            vec = np.array(items[1:], dtype="float32")
        except Exception:
            item = " "
            vec = np.array(items[2:], dtype="float32")

        embedding[item] = vec
        line = file_r.readline()
    return embedding, vec_dim
def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square