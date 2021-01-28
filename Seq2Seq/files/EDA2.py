import pandas as pd
import numpy as np
import utils
from utils import *
def merge(data,values,counts):
    english = ['English Teacher Abroad ','Graduates: English Teacher Abroad (Conversational)','English Teacher Abroad', 'English Teacher Abroad (Conversational)','Graduates: English Teacher Abroad ','English Teacher Overseas']
    customer = ['Customer Service Associate ', 'Customer Service Associate - Part Time ','Customer Service Associate','Customer Service Associate - Part Time','Customer Service Associate - On Call ']
    code = ['Software Engineer','Software Developer']
    english_c = 0
    customer_c = 0
    code_c =0
    for i in english:
        data.title = data.title.replace(i,'English Teacher')
        english_c += counts[values.index(i)]
    for i in customer:
        data.title = data.title.replace(i,'Customer Service Associate')
        customer_c += counts[values.index(i)]
    for i in code:
        data.title = data.title.replace(i,'Software Engineer')
        code_c += counts[values.index(i)]
    print(english_c,customer_c,code_c)
    return data
def print_job_requirement(title,data):

    requirement = data[data["title"] == title]
    print(requirement['requirements'])

data = pd.read_csv('/home/pan/Downloads/fake_job_postings.csv')
tdata = data[data['fraudulent'] == 0]
df_no_none = tdata.dropna(subset=['description','requirements'])
# print(data.isna().sum())
# df_no_none['function'].value_counts()
# df_no_none[['function', 'industry','title','location']].groupby('function').agg(['count', 'size', 'nunique'])
# df_no_none['title'].value_counts()
# print(df_no_none['title'].value_counts())
values = df_no_none['title'].value_counts().keys().tolist()
counts = df_no_none['title'].value_counts().tolist()
df = merge(df_no_none,values,counts)

# requirement = df_no_none.loc[df_no_none['title'] == 'Customer Service Associate ','requirements']
#['English Teacher Abroad ', 'Customer Service Associate ', 'Graduates: English Teacher Abroad (Conversational)', 'English Teacher Abroad', 'English Teacher Abroad (Conversational)', 'Customer Service Associate - Part Time ', 'Account Manager', 'Software Engineer', 'Web Developer', 'Project Manager', 'Graduates: English Teacher Abroad ', 'Customer Service Associate', 'Product Manager', 'Marketing Manager', 'Senior Software Engineer', 'Customer Service Team Lead ', 'Sales Representative', 'iOS Developer', 'Office Manager', 'Web Designer', 'Account Executive', 'Customer Service Representative', 'Front End Developer', 'Contact Center Representatives', 'Software Developer', 'Administrative Assistant', 'Android Developer', 'Customer Service Technical Specialist ', 'Data Scientist', 'Java Developer', 'Business Analyst', 'Executive Assistant', 'PHP Developer', 'DevOps Engineer', 'English Teacher Overseas', 'Business Development Manager', 'Graphic Designer', 'Sales Director', 'Sales Manager', 'Marketing Representative', 'Collections Supervisor', 'Sales Executive', 'Data Analyst', 'Inside Sales Representative', 'UI/UX Designer', 'Community Manager', 'QA Engineer', 'Marketing Associate', 'Product Designer', 'Marketing Intern', 'Customer Service Associate - On Call ', 'Operations Manager', 'Agent-Inbound Sales Position', 'UX Designer', 'Customer Service Associate - Part Time', 'Physical Therapist', 'Systems Engineer', 'Head of Marketing', 'Senior Web Developer', 'Senior Java Developer', 'Front-end Developer', 'UI Designer', 'Office Administrator', 'Accountant', 'Customer Success Manager', 'Senior Developer', 'Digital Marketing Manager', 'Quality Assurance Engineer', 'Front-End Developer', 'Technical Support Engineer']
#[311, 145, 144, 95, 83, 76, 69, 64, 61, 57, 57, 43, 42, 42, 41, 40, 39, 39, 39, 38, 38, 32, 32, 31, 30, 28, 28, 26, 26, 24, 23, 23, 22, 22, 21, 21, 21, 21, 21, 20, 19, 18, 18, 18, 18, 18, 17, 17, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13]

values_c = df['title'].value_counts().keys().tolist()
counts_c = df['title'].value_counts().tolist()
df_100 = df[df['title'].isin(values_c[0:100])]
simple = set()
len_source_summary = []
len_traget_summary = []
df_100 = df_100.fillna(" ")
df_100 = df_100.sort_values(by='title', key=lambda col: col.str.lower())
counts_c = df_100['title'].value_counts().sort_index( key=lambda col: col.str.lower()).tolist()
print(counts_c)
for title,description,industry,function,requirements in zip(df_100["title"],df_100['description'],df_100['industry'],df_100["function"],df_100['requirements']):
    description = remove_URL(description)
    description = remove_emoji(description)
    description = remove_html(description)
    requirements = remove_URL(requirements)
    requirements = remove_emoji(requirements)
    requirements = remove_html(requirements)
    len_source_summary.append(len(''+title+industry+function+description))
    len_traget_summary.append(len(''+requirements))
    merge = ''+title+" "+industry+" "+function+" "+description+'<sep>'+requirements
    simple.add(merge)
write_samples(simple, "files/samples.txt")
partition(simple,counts_c)




