import pandas as pd
import json
filename = 'yelp_academic_dataset_review.json'
f = json.loads(filename)
from json import JSONDecoder
JSONDecoder()
print f
review_frame = pd.read_json(filename,convert_axes=False,convert_dates=False)
print review_frame.columns
print review_frame.head(3)
review_frame = review_frame.head(1000)
review_frame.to_csv('yelp-sample.csv',index=False)