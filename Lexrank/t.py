from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from path import Path

documents = []
documents_dir = Path('/home/pan/Downloads/tech')

for file_path in documents_dir.files('*.txt'):
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        documents.append(fp.readlines())

lxr = LexRank(documents, stopwords=STOPWORDS['en'])

sentences = [
    'Experience with content management systems a major plus (any blogging counts!)',
    'Familiar with the Food52 editorial voice and aestheticLoves food, appreciates the importance of home cooking and cooking with the seasons.',
    'Meticulous editor,perfectionist, obsessive attention to detail,maddened by typos and broken links, delighted by finding and fixing them.'
    'Cheerful under pressure.',
    'Excellent communication skills.',
    'A+ multi-tasker and juggler of responsibilities big and small.',
    'Interested in and engaged with social media like Twitter, Facebook, and PinterestLoves .',
    'Problem-solving and collaborating to drive Food52 forward.'
    'Thinks big picture but pitches in on the nitty gritty of running a small company (dishes, shopping, administrative support).',
    'Comfortable with the realities of working for a startup: being on call on evenings and weekends, and working long hours.'
]

# get summary with classical LexRank algorithm
summary = lxr.get_summary(sentences, summary_size=2, threshold=.4)
print(summary)



# get summary with continuous LexRank
summary_cont = lxr.get_summary(sentences, threshold=None)
print(summary_cont)