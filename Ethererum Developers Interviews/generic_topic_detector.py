from gensim import corpora
import gensim
import pyLDAvis
import pyLDAvis.gensim
from features import preprocess
import warnings
warnings.filterwarnings("ignore")


def get_topic_by_lda(dictionary_list,number_topics=5, ldavis_url=None, ldavis_css_url=None):
    dictionary = corpora.Dictionary(dictionary_list)
    dictionary.filter_extremes(no_below=0, no_above=1.0)
    corpus = [dictionary.doc2bow(text) for text in dictionary_list]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=number_topics, id2word=dictionary, passes=20)
    data_prepared_object = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, n_jobs=1)
    formatted_html = pyLDAvis.prepared_data_to_html(data_prepared_object,
                                                    ldavis_url=ldavis_url,
                                                    ldavis_css_url=ldavis_css_url)
    return formatted_html


def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def clean_text(text_string):
    text_string = preprocess.text_clean_pipeline(text_string)
    return text_string


def get_transactional_list(list_of_strings,ngram_value=3):
    transaction_list = list([])
    for text_string in list_of_strings:
        if text_string != '': # and preprocess.is_ascii(text_string)
            cleaned_text_string = clean_text(text_string) #.encode('ascii','ignore')
            cleaned_ngram_tokens = find_ngrams(cleaned_text_string.split(), ngram_value)
            if len(cleaned_ngram_tokens) > 0:
                inner_gram_list = list([])
                for token in cleaned_ngram_tokens:
                    inner_gram_string = ''
                    for inner_ngram in token:
                        inner_gram_string = inner_gram_string + inner_ngram + ' '
                    inner_gram_list.append(inner_gram_string.strip())
                transaction_list.append(inner_gram_list)
    return transaction_list


def get_formatted_html_data(text_list,num_topics=5,ngram_value=3):
    formatted_transaction_list = get_transactional_list(text_list,ngram_value=ngram_value)
    return get_topic_by_lda(formatted_transaction_list,number_topics=num_topics)
