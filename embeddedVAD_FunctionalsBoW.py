# coding: utf-8
import os
import pandas as pd
import numpy as np
import statistics
import math
import nltk
import shutil
import json
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import stats
from io import StringIO
from gensim.models import KeyedVectors
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from textblob import TextBlob
from textblob.exceptions import NotTranslated
from urllib.error import HTTPError
from deep_translator import GoogleTranslator


def convert_nan_inf(my_var):
    if (math.isnan(my_var)) or (my_var == np.inf):
        my_var = 0
    return my_var


def get_functionals(var_list, output_file):
    print('Extracting functionals ...')
    var_max         = max(var_list)
    var_min         = min(var_list)
    var_mean        = statistics.mean(var_list)
    var_quartile1   = np.percentile(var_list, 25)
    var_quartile3   = np.percentile(var_list, 75)
    var_std         = statistics.stdev(var_list)
    var_range       = var_max - var_min
    var_gmean       = stats.gmean(var_list)
    var_harmonic    = statistics.harmonic_mean(var_list)
    var_median      = statistics.median(var_list)
    var_mode        = stats.mode(var_list)[0][0]
    var_kurtosis    = stats.kurtosis(var_list)
    var_skewness    = stats.skew(var_list)
    var_variation   = stats.variation(var_list)
    var_variation   = convert_nan_inf(var_variation)
    var_variance    = statistics.variance(var_list)
    var_iqr         = stats.iqr(var_list)
    var_iqr         = convert_nan_inf(var_iqr)
    output_file[-1] = output_file[-1] + ',' + str(var_range) + ',' + str(var_std) + ',' + str(var_median) + ',' + str(var_mode) + ',' + str(var_kurtosis) + ',' + str(var_skewness) + ',' + str(var_variation) + ',' + str(var_variance) + ',' + str(var_iqr) + ',' + str(var_quartile1) + ',' + str(var_quartile3) + ',' + str(var_max) + ',' + str(var_min) + ',' + str(var_mean) + ',' + str(var_gmean) + ',' + str(var_harmonic)
    return output_file


def fill_out_vals(lexicon_words, w, lexicon_df, string_v, string_a, string_d, new_dic, valence, arousal, dominance, raw_data, song, word_original):
    i = lexicon_words.index(w.casefold())
    v = round(float(lexicon_df.iloc[i][string_v]), 3)
    a = round(float(lexicon_df.iloc[i][string_a]), 3)
    d = round(float(lexicon_df.iloc[i][string_d]), 3)

    new_dic[w.casefold()] = [v, a, d]
    print('Adding VAD for', w, ':',  v, ' ', a, ' ', d)
    if word_original is not None:
        new_dic[word_original.casefold()] = [v, a, d]
        print('Adding VAD for', word_original, ':',  v, ' ', a, ' ', d)

    valence.append(v)
    arousal.append(a)
    dominance.append(d)
    raw_data.append(str(song) + ',' + str(v) + ',' + str(a) + ',' + str(d))
    return valence, arousal, dominance, raw_data, new_dic


def get_english_word(word):
    idiom  = 'en'
    idioms = ['de', 'fr', 'it', 'es']
    try:
        idiom = detect(word)
    except LangDetectException:
        print(word + ': raised a LangDetectException ...')
        pass
    if idiom in idioms:
        print(word, ' is not an English word')
        try:
            analysis = TextBlob(word)
            word2    = analysis.translate(to='en')
            word     = str(word2)
            print('Lets consider ', word)
        except NotTranslated:
            print(word + ': raised a NotTranslatedException ...')
        except HTTPError:
            print('Too Many Requests error has been raised')
            translated = GoogleTranslator(source='auto', target='en').translate(word)
            word       = str(translated)
            print('Lets consider ', word)
    return word


def fill_vals_from_dic(new_dic, word, valence, arousal, dominance, raw_data, song, word_original):
    valence.append(new_dic[word.casefold()][0])
    arousal.append(new_dic[word.casefold()][1])
    dominance.append(new_dic[word.casefold()][2])
    raw_data.append(str(song) + ',' + str(new_dic[word.casefold()][0]) + ',' + str(new_dic[word.casefold()][1]) + ',' + str(new_dic[word.casefold()][2]))
    if word_original is not None:
        new_dic[word_original.casefold()] = [new_dic[word.casefold()][0], new_dic[word.casefold()][1], new_dic[word.casefold()][2]]
        print('Adding VAD for', word_original, ':',  new_dic[word.casefold()][0], ' ', new_dic[word.casefold()][1], ' ', new_dic[word.casefold()][2])
    return valence, arousal, dominance, raw_data


def get_VAD_scores_per_word(model, word, NRC_words, new_dic, valence, arousal, dominance, raw_data, song, skip_list, NRC_df):
    if word.casefold() in new_dic:
        valence, arousal, dominance, raw_data = fill_vals_from_dic(new_dic, word, valence, arousal, dominance, raw_data, song, word_original=None)
    elif word.casefold() in NRC_words:
        valence, arousal, dominance, raw_data, new_dic = fill_out_vals(NRC_words, word, NRC_df, 'Valence', 'Arousal', 'Dominance', new_dic, valence, arousal, dominance, raw_data, song, word_original=None)
    elif word.casefold() not in skip_list:
        if len(word) > 1:
            word = get_english_word(word.casefold())
            if word.casefold() in new_dic:
                valence, arousal, dominance, raw_data = fill_vals_from_dic(new_dic, word, valence, arousal, dominance, raw_data, song, word_original=None)
            elif word.casefold() in NRC_words:
                valence, arousal, dominance, raw_data, new_dic = fill_out_vals(NRC_words, word, NRC_df, 'Valence', 'Arousal', 'Dominance', new_dic, valence, arousal, dominance, raw_data, song, word_original=None)
            else:
                print(word.casefold(), 'not in Dictionary ...')
                match = False
                i     = 1
                count = 0
                while not match:
                    try:
                        next_similar = model.most_similar(positive=[word.casefold()], topn=i)
                        print('The most similar word in the embedding is', next_similar[i-1][0])
                        if next_similar[i-1][0].casefold() in new_dic:
                            print(next_similar[i-1][0], 'found on Dictionary!')
                            match = True
                            valence, arousal, dominance, raw_data = fill_vals_from_dic(new_dic, next_similar[i-1][0], valence, arousal, dominance, raw_data, song, word)
                        elif next_similar[i-1][0].casefold() in NRC_words:
                            print(next_similar[i-1][0], 'found on Dictionary!')
                            match = True
                            valence, arousal, dominance, raw_data, new_dic = fill_out_vals(NRC_words, next_similar[i-1][0], NRC_df, 'Valence', 'Arousal', 'Dominance', new_dic, valence, arousal, dominance, raw_data, song, word)
                        else:
                            print(next_similar[i-1][0], 'not in Dictionary :(')
                            match = False
                            i     = i + 1
                            count = count + 1
                            if count == 5:
                                match = True
                                skip_list.append(word.casefold())
                                print('Maximum number of attempts reached')
                            else:
                                print('Attempt ' + str(count) + ': Searching next ...')
                    except KeyError:
                        match = True
                        skip_list.append(word.casefold())
                        print(word, 'does not exist in the embedding')
        else:
            skip_list.append(word.casefold())
    return valence, arousal, dominance, skip_list, new_dic


def get_VAD_with_embeddings(model, instance_ID, tags_list, output_file, NRC_df, raw_data, song, skip_list, new_dic, NRC_words):
    lemmatizer = WordNetLemmatizer()
    valence    = []
    arousal    = []
    dominance  = []
    for tag in tags_list:
        if ' ' in tag:
            tokenizer  = nltk.RegexpTokenizer(r"\w+")
            words      = tokenizer.tokenize(tag)
            stop_words = set(stopwords.words('english'))
            s2         = [word for word in words if word not in stop_words]
            for index, word in enumerate(s2):
                w = lemmatizer.lemmatize(word)
                if not w.isdigit():
                    valence, arousal, dominance, skip_list, new_dic = get_VAD_scores_per_word(model, w, NRC_words, new_dic, valence, arousal, dominance, raw_data, song, skip_list, NRC_df)
        elif not tag.isdigit():
            tag = re.sub(r'[^a-z]', '', tag)
            w   = lemmatizer.lemmatize(tag)
            valence, arousal, dominance, skip_list, new_dic = get_VAD_scores_per_word(model, w, NRC_words, new_dic, valence, arousal, dominance, raw_data, song, skip_list, NRC_df)

    if len(valence) <= 1:
        output_file.append(instance_ID + ',0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0')
    else:
        output_file.append(instance_ID)
        output_file = get_functionals(valence, output_file)
        output_file = get_functionals(arousal, output_file)
        output_file = get_functionals(dominance, output_file)
    return output_file, new_dic


def import_data(data):
    """
    Taken from Github repo
    """
    cols        = ['track_id', 'track', 'artist_id', 'artist', 'user_id', 'tags']
    events_file = data + "LFM-2b_artist_track_LEs_no_empty_tags.txt"
    for_pd      = StringIO()
    with open(events_file) as file:
        for i, line in enumerate(file):
            line     = line.replace("|", " ")  # ensure no | is used
            new_line = re.sub(r'\t', '|', line.rstrip(), count=5)
            print(new_line, file=for_pd)
    for_pd.seek(0)
    df_events = pd.read_csv(for_pd, sep='|', names=cols, index_col=False)
    df_events.to_csv(data + "LFM-2b_transformed.csv")
    # df_events_sub = df_events.iloc[:10000,:]
    # df_events_sub.to_csv(data + "LFM-2b_transformed_test10000.csv")


def get_tags(data):
    """
    Taken from Github repo
    """
    df_events = pd.read_csv(data + "LFM-2b_transformed.csv", index_col=0)
    # df_events = pd.read_csv(data + "LFM-2b_transformed_test.csv", index_col=0)
    # df_events = pd.read_csv(data + "LFM-2b_transformed_test10.csv", index_col=0)
    # df_events = pd.read_csv(data + "LFM-2b_transformed_test10000.csv", index_col=0)
    df_events["just_tags"] = df_events["tags"].apply(lambda x: x.split("\t")[::2])
    return df_events


def get_VAD_functionals_raw(my_dir):
    embedding   = my_dir + '/GoogleNews-vectors-negative300.bin'  # If you do not have this, download it from here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    model       = KeyedVectors.load_word2vec_format(embedding, binary=True)
    output_file = ['File_name,V_range,V_std,V_median,V_mode,V_kurtosis,V_skewness,V_variation,V_variance,V_iqr,V_quartile1,V_quartile3,V_max,V_min,V_mean,V_gmean,V_harmonic,A_range,A_std,A_median,A_mode,A_kurtosis,A_skewness,A_variation,A_variance,A_iqr,A_quartile1,A_quartile3,A_max,A_min,A_mean,A_gmean,A_harmonic,D_range,D_std,D_median,D_mode,D_kurtosis,D_skewness,D_variation,D_variance,D_iqr,D_quartile1,D_quartile3,D_max,D_min,D_mean,D_gmean,D_harmonic']
    data        = my_dir + '/data/'
    raw_data    = ['File_name,Valence,Arousal,Dominance']
    skip_list   = []
    # import_data(data)

    df_events     = get_tags(data)
    track_id_list = df_events['track_id'].to_list()
    NRC           = my_dir + '/dictionaries/NRC-VAD-Lexicon.txt'
    NRC_df        = pd.read_csv(NRC, delimiter='\t')
    NRC_words     = NRC_df['Word'].tolist()
    new_dic       = {}

    if os.path.exists(my_dir + '/VAD_functionals.csv'):
        os.remove(my_dir + '/VAD_functionals.csv')
    if os.path.exists(my_dir + '/VAD_raw.csv'):
        os.remove(my_dir + '/VAD_raw.csv')
    for index, song in enumerate(track_id_list):
        print('Processing tags for', song, ' ...')
        tags_list   = df_events.loc[index, 'just_tags']
        output_file, new_dic = get_VAD_with_embeddings(model, song, tags_list, output_file, NRC_df, raw_data, song, skip_list, new_dic, NRC_words)

    f = open(my_dir + '/VAD_functionals.csv', 'w')
    for line in output_file:
        print(line, file=f)
    f.close()

    f = open(my_dir + '/VAD_raw.csv', 'w')
    for line in raw_data:
        print(line, file=f)
    f.close()

    with open('tagsVAD_dic.json', 'w') as f:
        json.dump(new_dic, f)
    # with open('new_dic.json') as f:
    #     new_dic = json.load(f)

    print(skip_list)


def get_BOW(my_dir):
    options = ' -writeName'
    print('Extracting BOW from VAD scores')
    file_train = my_dir + '/resources_BOW/VAD_tags_train.csv'
    file_test  = my_dir + '/resources_BOW/VAD_tags_test.csv'
    os.system('java -Xmx45g -jar openXBOW.jar -i ' + file_train + ' -o ' + my_dir + '/VAD_BoW_train.csv' + options + ' -B codebook_VAD -log -a 10 -standardizeInput')
    os.system('java -Xmx45g -jar openXBOW.jar -i ' + file_test + ' -o ' + my_dir + '/VAD_BoW_test.csv' + options + ' -b codebook_VAD')


def clean_BoW_csv(my_dir):
    for split in ['_train.csv', '_test.csv']:
        df     = pd.read_csv(my_dir + '/VAD_BoW' + split, header=None, sep=';')
        df_new = df.replace(r'\'', '', regex=True)
        df_new.to_csv(my_dir + '/VAD_BoW' + split, header=None, sep=',', index=False)


def get_resourcesBOW(my_dir):
    print('Getting train/test input for BoW')
    if not os.path.exists(my_dir + '/resources_BOW'):
        os.mkdir(my_dir + '/resources_BOW')
    else:
        shutil.rmtree(my_dir + '/resources_BOW')
        os.mkdir(my_dir + '/resources_BOW')

    df          = pd.read_csv(my_dir + '/VAD_raw.csv', sep=',')
    all_files   = df['File_name'].to_list()
    files_train = all_files[:int(len(all_files)*0.7)]
    files_test  = all_files[int(len(all_files)*0.8):]

    df1 = df[df['File_name'].isin(files_train)]
    df1.to_csv(my_dir + '/resources_BOW/VAD_tags_train.csv', sep=';', index=False)
    df2 = df[df['File_name'].isin(files_test)]
    df2.to_csv(my_dir + '/resources_BOW/VAD_tags_test.csv', sep=';', index=False)


if __name__ == '__main__':
    my_dir = os.getcwd()
    get_VAD_functionals_raw(my_dir)
    get_resourcesBOW(my_dir)
    get_BOW(my_dir)  # If you do not have openXBoW, you can get it here: https://github.com/openXBOW/openXBOW
    clean_BoW_csv(my_dir)
