import json
import pandas as pd
import numpy as np
import re
data = pd.read_csv('/home/pan/Downloads/cleanedTop30_1.csv')
file1 = open("review.json","w+",encoding='utf-8')
counter = len(data['Description'])
for index,desc ,query in zip(data["index"],data['Description'],data['Query']):
    if isinstance(desc,str):
        desc = re.sub(r"[^a-zA-Z0-9.?,!:]"," ",desc)
        desc = re.sub(' +',' ',desc)
    else:
        desc = " "
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


{
    // string, 22 character unique string business id
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // string, the business's name
    "name": "Garaje",

    // string, the full address of the business
    "address": "475 3rd St",

    // string, the city
    "city": "San Francisco",

    // string, 2 character state code, if applicable
    "state": "CA",

    // string, the postal code
    "postal code": "94107",

    // float, latitude
    "latitude": 37.7817529521,

    // float, longitude
    "longitude": -122.39612197,

    // float, star rating, rounded to half-stars
    "stars": 4.5,

    // integer, number of reviews
    "review_count": 1198,

    // integer, 0 or 1 for closed or open, respectively
    "is_open": 1,

    // object, business attributes to values. note: some attribute values might be objects
    "attributes": {
        "RestaurantsTakeOut": true,
        "BusinessParking": {
            "garage": false,
            "street": true,
            "validated": false,
            "lot": false,
            "valet": false
        },
    },

    // an array of strings of business categories
    "categories": [
        "Mexican",
        "Burgers",
        "Gastropubs"
    ],

    // an object of key day to value hours, hours are using a 24hr clock
    "hours": {
        "Monday": "10:00-21:00",
        "Tuesday": "10:00-21:00",
        "Friday": "10:00-21:00",
        "Wednesday": "10:00-21:00",
        "Thursday": "10:00-21:00",
        "Sunday": "11:00-18:00",
        "Saturday": "10:00-21:00"
    }
}
"""

