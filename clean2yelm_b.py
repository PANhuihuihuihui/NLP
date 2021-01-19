import json
import pandas as pd
import numpy as np
import re
data = pd.read_csv('/home/pan/Downloads/cleanedTop30_1.csv')
file1 = open("business.json","w+",encoding='utf-8')

last_query = data['Query'][0]
last_query = re.sub(r"[^a-zA-Z0-9.?,!:]"," ",last_query)
last_query = re.sub(' +',' ',last_query)

for index,query in zip(data["index"],data['Query']):
    query = re.sub(r"[^a-zA-Z0-9.?,!:]"," ",query)
    query = re.sub(' +',' ',query)
    if last_query != query:
        last_query = query
        dic = {
            "business_id": query,
            "categories": 
            [
                query
            ],
            "city": "San Francisco"
            
        }
    else:
        continue

    file1.write(json.dumps(dic))
    file1.write('\n')
file1.close() 
