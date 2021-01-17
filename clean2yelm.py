import json
import pandas as pd
import numpy as np
data = pd.read_csv('/home/pan/Downloads/cleanedTop30_1.csv')
file1 = open("review.json","w+",encoding='utf-8')
counter = len(data['Description'])
for index,desc, query in zip(data["index"],data['Description'],data['Query']):
    
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
    file1.write(json.dumps(dic))
    file1.write('\n')
    counter -=1
    print(counter)
file1.close() 


"""
{
    // string, 22 character unique review id
    "review_id": "zdSx_SD6obEhz9VrW9uAWA",

    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // integer, star rating
    "stars": 4,

    // string, date formatted YYYY-MM-DD
    "date": "2016-03-09",

    // string, the review itself
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of drinks.",

    // integer, number of useful votes received
    "useful": 0,

    // integer, number of funny votes received
    "funny": 0,

    // integer, number of cool votes received
    "cool": 0
}
"""
