import pandas as pd
import generic_topic_detector

filepath = "C:\\Users\\rupachak\\Desktop\\Kaggle Data\\Ethereum Developer Interviews\\interview.csv"
interview_frame = pd.read_csv(filepath)
text_list = interview_frame['Who are you and what are you working on?'].values
text_list = list(map(lambda x:str(x),text_list))

lda_html = generic_topic_detector.get_formatted_html_data(text_list)
with open('eth_topics.html','w') as lda_topic:
    lda_topic.write(lda_html)

