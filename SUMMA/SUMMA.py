text_requirement = "Experience with content management systems a major plus (any blogging counts!). Familiar with the Food52 editorial voice and aestheticLoves food, appreciates the importance of home cooking and cooking with the seasons.\
    Meticulous editor,perfectionist, obsessive attention to detail,maddened by typos and broken links, delighted by finding and fixing them.\
    Cheerful under pressure.Excellent communication skills.\
    A+ multi-tasker and juggler of responsibilities big and small.\
    Interested in and engaged with social media like Twitter, Facebook, and PinterestLoves .\
    Problem-solving and collaborating to drive Food52 forward.\
    Thinks big picture but pitches in on the nitty gritty of running a small company (dishes, shopping, administrative support).\
    Comfortable with the realities of working for a startup: being on call on evenings and weekends, and working long hours."
text_des = "Food52, a fast-growing, James Beard Award-winning online food community and crowd-sourced and curated recipe hub, is currently interviewing full- and part-time unpaid interns to work in a small team of editors, executives, and developers in its New York City headquarters.Reproducing and/or repackaging existing Food52 content for a number of partner sites, such as Huffington Post, Yahoo, Buzzfeed, and more in their various content management systemsResearching blogs and websites for the Provisions by Food52 Affiliate ProgramAssisting in day-to-day affiliate program support, such as screening affiliates and assisting in any affiliate inquiriesSupporting with PR &amp; Events when neededHelping with office administrative work, such as filing, mailing, and preparing for meetingsWorking with developers to document bugs and suggest improvements to the siteSupporting the marketing and executive staff"
from summa import summarizer
print(summarizer.summarize(text_requirement,ratio=0.4))
print(summarizer.summarize(text_des))