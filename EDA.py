import json
import pandas as pd
import numpy as np
import re
data = pd.read_csv('/home/pan/Downloads/cleanedTop30_1.csv')
file1 = open("review.json","w+",encoding='utf-8')
counter = len(data['Description'])
length_summary = []
for index,desc ,query in zip(data["index"],data['Description'],data['Query']):
    if isinstance(desc,str):
        desc = re.sub(r"[^a-zA-Z0-9.?,!:]"," ",desc)
        desc = re.sub(' +',' ',desc)
        query = re.sub(r"[^a-zA-Z0-9.?,!:]"," ",query)
        query = re.sub(' +',' ',query)
        if len(desc) < 40:
            print(desc)
            continue
        length_summary.append(len(desc))
    else:
        continue
    dic = {
        "review_id":str(index), 
        "user_id": query,
        "business_id": query,
        "stars": str(4),
        "date": "2016-03-09",
        "text": desc,
        "useful": str(0),
        "funny":str(0),
        "cool": str(0)
    }

    # dic = {
    #     "business_id": query,
    #     "categories": 
    #     [
    #         "Mexican",
    #         "Burgers",
    #         "Gastropubs"
    #     ],
    #     "city": "San Francisco"
        
    # }

    file1.write(json.dumps(dic))
    file1.write('\n')
    counter -=1
    # print(counter)
file1.close() 

df = pd.DataFrame(length_summary,columns=['len'])
print(df.info())
print(df.describe(percentiles=[.05,.25, .5, .75,.95]))


# None
#                 len
# count  72291.000000
# mean    1565.250183
# std      887.107365
# min        1.000000
# 25%     1003.500000
# 50%     1619.000000
# 75%     1856.000000
# max    18575.000000
# memory usage: 564.9 KB
# None
#                 len
# count  72291.000000
# mean    1565.250183
# std      887.107365
# min        1.000000
# 25%     1003.500000
# 50%     1619.000000
# 75%     1856.000000
# 95%     3760.000000
# max    18575.000000

# None
#                 len
# count  71927.000000
# mean    1573.121874
# std      882.402800
# min       40.000000
# 5%       385.000000
# 25%     1019.000000
# 50%     1619.000000
# 75%     1856.000000
# 95%     3760.000000
# max    18575.000000

#   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   len     71927 non-null  int64
# dtypes: int64(1)
# memory usage: 562.1 KB
# None
#                 len
# count  71927.000000
# mean    1573.121874
# std      882.402800
# min       40.000000
# 5%       385.000000
# 25%     1019.000000
# 50%     1619.000000
# 75%     1856.000000
# 95%     3760.000000
# max    18575.000000

# Saving processed splits
# Loading all reviews
# Filtering reviews longer than: 3760
# Total number of reviews before filtering: 71927
# Total number of reviews after filtering: 71927
# Filtering items with less than 50 reviews
# Total number of reviews after filtering: 71927
# Total number of items after filtering: 30
# Number of train reviews: 59516 / 57541
# Number of val reviews: 6258 / 7192
# Number of test reviews: 6153 / 7192
