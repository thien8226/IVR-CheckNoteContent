import pandas as pd
import numpy as np
import regex as re
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer


# Feature "time_call"
def get_hour(date_time: str):
    time = date_time.split()[1]
    hour = time.split(":")[0]
    
    return float(hour)

# Feature "transcription"
uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
vowel_table = [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]
accent_table = ['', 'f', 's', 'r', 'x', 'j']

vowel_to_ids = {}
for i in range(len(vowel_table)):
    for j in range(len(vowel_table[i]) - 1):
        vowel_to_ids[vowel_table[i][j]] = (i, j)

def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

def convert_unicode(txt):
    dicchar = loaddicchar()

    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def word_accent_normalize(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    vowel_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = vowel_table[x][0]
        if not qu_or_gi or index != 1:
            vowel_index.append(index)
    if len(vowel_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = vowel_to_ids.get(chars[1])
                chars[1] = vowel_table[x][dau_cau]
            else:
                x, y = vowel_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = vowel_table[x][dau_cau]
                else:
                    chars[1] = vowel_table[5][dau_cau] if chars[1] == 'i' else vowel_table[9][dau_cau]
            return ''.join(chars)
        return word

    for index in vowel_index:
        x, y = vowel_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = vowel_table[x][dau_cau]
            # for index2 in vowel_index:
            #     if index2 != index:
            #         x, y = vowel_to_ids[chars[index]]
            #         chars[index2] = vowel_table[x][0]
            return ''.join(chars)

    if len(vowel_index) == 2:
        if vowel_index[-1] == len(chars) - 1:
            x, y = vowel_to_ids[chars[vowel_index[0]]]
            chars[vowel_index[0]] = vowel_table[x][dau_cau]
            # x, y = vowel_to_ids[chars[vowel_index[1]]]
            # chars[vowel_index[1]] = vowel_table[x][0]
        else:
            # x, y = vowel_to_ids[chars[vowel_index[0]]]
            # chars[vowel_index[0]] = vowel_table[x][0]
            x, y = vowel_to_ids[chars[vowel_index[1]]]
            chars[vowel_index[1]] = vowel_table[x][dau_cau]
    else:
        # x, y = vowel_to_ids[chars[vowel_index[0]]]
        # chars[vowel_index[0]] = vowel_table[x][0]
        x, y = vowel_to_ids[chars[vowel_index[1]]]
        chars[vowel_index[1]] = vowel_table[x][dau_cau]
        # x, y = vowel_to_ids[chars[vowel_index[2]]]
        # chars[vowel_index[2]] = vowel_table[x][0]
    return ''.join(chars)

def remove_and_tokenize(text: str) -> str:
    text = ' '.join(word for word in simple_preprocess(text))
    # text = word_tokenize(text, format='text')
    text = ViTokenizer.tokenize(text)

    return text

def preprocess_text(text: str) -> str:
    if type(text) == str:
        text = text.lower().strip()
        text = convert_unicode(text)
        text = accent_normalize(text)
        text = remove_and_tokenize(text)
        text = remove_stopwords(text)
    else:
        text = ""

    return text

def remove_stopwords(text):
    with open("../data/stopwords.txt") as f:
        stopwords = f.readlines()
    stopwords = [stopword.strip() for stopword in stopwords]

    return ' '.join(word for word in text.split() if word not in stopwords)

def is_valid_vietnam_word(word):
    chars = list(word)
    vowel_index = -1
    for index, char in enumerate(chars):
        x, y = vowel_to_ids.get(char, (-1, -1))
        if x != -1:
            if vowel_index == -1:
                vowel_index = index
            else:
                if index - vowel_index != 1:
                    return False
                vowel_index = index
    return True

def accent_normalize(sentence):
    # sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^\p{P}*)([p{L}.]*\p{L}+)(\p{P}*$)', r'\1/\2/\3', word).split('/')
        # print(cw)
        if len(cw) == 3:
            cw[1] = word_accent_normalize(cw[1])
        words[index] = ''.join(cw)
    return ' '.join(words)

def main(use_speech_features=False):
    # Loading data
    train_metadata = pd.read_csv("../data/official/train_with_silence.csv")
    test_metadata = pd.read_csv("../data/official/test_with_silence.csv")

    # Encode labels

    ## Note content
    note_content2label = {}
    num_note_contents = train_metadata["note_content"].nunique()
    note_contents = list(train_metadata["note_content"].value_counts().index)

    for i in range(num_note_contents):
        note_content2label[note_contents[i]] = num_note_contents - i - 1

    train_metadata["note_content"] = train_metadata["note_content"].apply(lambda x: note_content2label[x] if type(x) == str else x)
    test_metadata["note_content"] = test_metadata["note_content"].apply(lambda x: note_content2label[x] if type(x) == str else x)

    ## Mapping code
    mapping_code2label = {}
    num_mapping_codes = train_metadata["mapping_code"].nunique()
    mapping_codes = list(train_metadata["mapping_code"].value_counts().index)

    for i in range(num_mapping_codes):
        mapping_code2label[mapping_codes[i]] = num_mapping_codes - i - 1

    train_metadata["mapping_code"] = train_metadata["mapping_code"].apply(lambda x: mapping_code2label[x] if type(x) == str else x)
    test_metadata["mapping_code"] = test_metadata["mapping_code"].apply(lambda x: mapping_code2label[x] if type(x) == str else x)

    # Speech features
    if use_speech_features:
        train_metadata["transcription"] = train_metadata["transcription"].apply(lambda x: preprocess_text(x))
        test_metadata["transcription"] = test_metadata["transcription"].apply(lambda x: preprocess_text(x))

    # Save metadata
    train_metadata.to_csv("../data/processed_train_with_silence_no_error.csv", index=False)
    test_metadata.to_csv("../data/processed_test_with_silence_no_error.csv", index=False)


if __name__ == "__main__":
    main()
