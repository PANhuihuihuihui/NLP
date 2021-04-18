import re
import thulac

def skills(string,thu1,skill_list):
    nwant = [ "ns", "ni", "a","d","h","k","r","c","p","y","e","o","g","w","m","f","u"]
    wnwant = ["能力","年龄","岁","年","经验者","有","会","优先","熟悉","能够","学历","工作","相关","性别","男女","个","即可","能","性","经验","如","熟知","考虑","线"]
    tgtlist = re.split(r'[0-9]、',string[:-5])
    extract = []
    for index, string in enumerate(tgtlist):
        if index == 0:
            continue
        else:
            a = thu1.cut(string)
            tmp = []
            for elem , flage in a:
                if flage in ["w","c"] and elem != "、" and len(tmp) != 0:
                    if len(tmp) > 6:
                        continue
                    extract.append(" ".join(tmp))
                    tmp = []
                elif flage in nwant or elem in wnwant or elem.lower() not in skill_list:
                    continue
                else:
                    tmp.append(elem)
    return extract