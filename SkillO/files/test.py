import torch
import numpy as np
import random
from processdataall import *





"""
train:  8610 test:  2047
src_vocab:  14289 tar_vocab:  18712 skill_vocab 8506
704 421
"""

# This dataset is comming from SAMA 2020
# train = "/userhome/30/hjpan/NLP/SAMA/data/train.pt"
# test = "/userhome/30/hjpan/NLP/SAMA/data/test.pt"
# vocab_dir = "/userhome/30/hjpan/NLP/merge/data/vocab.pt"

train = "/home/phj/NLP/SAMA/dataset/train.pt"
test = "/home/phj/NLP/SAMA/dataset/test.pt"
vocab_dir = "/home/phj/NLP/SAMA/dataset/vocab.pt"

train_dataset = torch.load(train)
test_dataset = torch.load(test)
gVocab = torch.load(vocab_dir)

src_dic = gVocab.src_vocab
skill_dic = gVocab.skilltgt_vocab
tgt_dic = gVocab.tgt_vocab
skill_list = [str(elem).lower() for elem in skill_dic.word2id.keys()]

# import jieba
# import jieba.posseg as pseg
# jieba.enable_paddle()
string = "1 、中专 、 大专 以上 学历 ， 年龄 25 岁 以上 ， 机械 电子 类 专业 优先 ， 有 药品 、 保健品 、 食品 工厂 经历 优先 。  2 、 有 3 年 以上 生产 设备 制造 维护 保养 以及 维修 经验 （ 自动化 包装 设备 生产 制造 、 组装 、 调试 、 售后服务 等 经验者 优先 ) ， （ 熟悉 自动化 灌装机 、 背封 灌装机 、 三维 包装机 优先 ）  3 、 具有 电工 操作证 ， 能够 解决 设备 的 一般性 故障 ；  4 、 能够 读 懂 电气 图纸 、 根据 设备 结构图 ， 电子 图纸 分析 故障 。  5 、 能够 熟练 维修 机械 设备 ， 熟悉 机械 原理 及 电气 原理 ， 有 独立 处理 机械 设备 故障 能力 ， 会 电焊 或 氩弧 焊 <EOS>"
stringlist = string.split()
string = "1 、中专 、 大专 以上 学历 ， 年龄 25 岁 以上 ， 机械 电子 类 专业 优先 ， 有 药品 、 保健品 、 食品 工厂 经历 优先 。  2 、 有 3 年 以上 生产 设备 制造 维护 保养 以及 维修 经验 （ 自动化 包装 设备 生产 制造 、 组装 、 调试 、 售后服务 等 经验者 优先 ) ， （ 熟悉 自动化 灌装机 、 背封 灌装机 、 三维 包装机 优先 ）  3 、 具有 电工 操作证 ， 能够 解决 设备 的 一般性 故障 ；  4 、 能够 读 懂 电气 图纸 、 根据 设备 结构图 ， 电子 图纸 分析 故障 。  5 、 能够 熟练 维修 机械 设备 ， 熟悉 机械 原理 及 电气 原理 ， 有 独立 处理 机械 设备 故障 能力 ， 会 电焊 或 氩弧 焊 <EOS>".replace(" ","")
#string = "1 、 统招 本科 及 以上 学历 ， 5 年 以上 IM 开发 经验 ；  2 、 熟悉 主流 IM 产品 的 技术 架构 。 具备 高性能 ， 高 并发 ， 高 稳定性 系统 开发 和 调优 实际 工作 经验 ；  3 、 熟悉 IM 通讯 机制 及 常用 数据 传输 协议 ；  4 、 有 大型 IM 服务端 开发 经验 或 架构 设计 经验者 优先 。 <EOS>".replace(" ","")
string = "1 、 本科 及 以上 学历 ， 5 年 以上 零售 行业 销售 管理 工作 经验 ；  2 、 熟悉 零售 卖场 连锁 营运 管理 和 销售 管理 ， 了解 相关 的 法律 法规 ；  3 、 有 较 好 的 计划 组织 、 目标 管理 、 统筹 策划 和 商业 创新 能力 。 <EOS>".replace(" ","")
# string = "1 、 统招 全日制 大学 本科 及 以上 学历 。  2 、 3-5 年 以上 风险 计量 模型 开发 及 实施 经验 ， 从事 过 信用 评级 、 模型 开发 验证 实施 、 数据 质量 管理 、 风险 报告 、 风险 监测 、 组合 分析 等 方面 工作 。  3 、 精通 SAS 、 Python 等 工具 及 机器 学习 各 算法 ， 具备 较 强 的 数理 统计 相关 知识 背景 。  4 、 熟悉 信用卡 系统 、 数据 及 业务 管理 。  5 、 具备 良好 的 项目 管理 能力 ， 沟通 及 展现 能力 。 <EOS>".replace(" ","")
# words = pseg.cut(string,use_paddle=True) 

# for word, flag in words:
#     if word in stringlist and flag in ["n","nz","vn"]:
#         print('%s %s' % (word, flag))


# print(type(words))
# print(stringlist)


# data = train_dataset + test_dataset
# train = data[0:8000] 
# val = data[8000:8500] 
# test = data[8500:-1]
"""
n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名
m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
v/动词 a/形容词 d/副词 h/前接成分 k/后接成分 
i/习语 j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
e/叹词 o/拟声词 g/语素 w/标点 x/其它 
"""
import re
import thulac
thu1 = thulac.thulac(seg_only=False, model_path="/home/phj/software/models")  #设置模式为行分词模式
nwant = [ "ns", "ni", "a","d","h","k","r","c","p","y","e","o","g","w","m","f","u"]
wnwant = ["能力","年龄","岁","年","经验者","有","会","优先","熟悉","能够","学历","工作","相关","性别","男女","个","即可","能","性","如","熟知","考虑"]
tgtlist = re.split(r'[0-9]、',string[:-5])
extract = []
string_sapce = string = "1 、 本科 及 以上 学历 ， 5 年 以上 零售 行业 销售 管理 工作 经验 ；  2 、 熟悉 零售 卖场 连锁 营运 管理 和 销售 管理 ， 了解 相关 的 法律 法规 ；  3 、 有 较 好 的 计划 组织 、 目标 管理 、 统筹 策划 和 商业 创新 能力 。 <EOS>"
stringlist = string_sapce[:-5].split(" ")

print(tgtlist)
print(stringlist)
counter = 0
# stringlist.insert(0,"<nepod>")
# couter_str_len = 0
# for string in tgtlist:
#     couter_str_len += len(string)
stringlist.insert(0,"<nepod>")
for index, string in enumerate(tgtlist):
    if index == 0:
        counter+=2
        continue
    else:
        a = thu1.cut(string)
        tmp = []
        for elem , flage in a:
            # print("word: ",str(elem)," ",flage)
            if flage in ["w","c"] and elem != "、" and len(tmp) != 0:
                if len(tmp) > 6:
                    continue
                if flage == "c" and len(tmp) < 2:
                    continue
                counter = counter + stringlist[counter:].index(tmp[-1])
                sep_index = counter + stringlist[counter:].index(elem)
                if flage == "w" :
                    if sep_index + 3 >= len(stringlist):
                        stringlist = stringlist[:sep_index+1]+["<eowd>","<eopd>"]+stringlist[sep_index+1:]
                    else:
                        stringlist = stringlist[:sep_index+1]+["<eowd>","<nepod>"]+stringlist[sep_index+1:]
                if flage == "c":
                    stringlist = stringlist[:sep_index]+["<eowd>","<nepod>"]+stringlist[sep_index:]
                print("check: ",stringlist[counter],tmp[-1])
                extract.append(" ".join(tmp))
                tmp = []
            elif flage in nwant or elem in wnwant:
                continue
                print("word: ",str(elem)," ",flage)
            else:
                tmp.append(elem)
print(stringlist)
print(extract)
print("<SEP>".join(extract))
            
    



# nwant = [ "ns", "ni","v", "a","d","h","k","r","c","p","u","y","e","o","g","w","m"]
# for word, flag in a:
#     print('%s %s' % (word, flag))


"""
档案 管理 <SEP> 工程 管理 <SEP> 大专 <SEP> 档案 管理 <SEP> 办公 软件 <EOS>
档案 管理 工程 管理 相关 专业 大专 以上 学历 以上 相关 工作 经验 有 资料 员证 熟悉 档案 管理 办法 掌握 计算机 档案 管理 信息 系统 4 使用 办公 软件 有 建造 职称 优先 考虑 <EOS>
"""


"""
1 、 负责 仓库 的 管理 工作 ， 带领 团队 确保 收发货 、 出入库 、 整理 仓库 等 流程 工作 正常 进行 ， 顺应 天猫 、 淘宝 等 发货 物流 规则 ；  2 、 负责 日常 销售 订单 ， 货品 分拣 及 发货 安排 ， 核对 打印 的 快递 单 、 发货单 进行 配货 及 校验 ， 联系 物流 发货 ；  3 、 仓库 库位 巡视 、 补仓 ， ， 跟踪 库存 状况 ， 协调 客服部 处理 物流 售 后 工作 ；  4 、 管理 好 库存 ERP 系统 、 及时 更新 库存 ， 做好 每月 的 库存 盘点 工作 和 分析  5 、 执行 和 完善 仓库 的 规章制度 、 规范 作业 标准 及 流程 ， 提高 效率 ， 降低 成本 ；  6 、 科学 管理 货品 库位 ， 提出 改进 方案 ， 保证 仓库 的 使用率 ；  7 、 积极 配合 团队 工作 ， 合理 安排 好 仓管员 的 工作 ， 上传下达 ， 并 做好 现场 管理 ， 有效 配合 公司 整体 业务 运作 需求 。
<sep>淘宝 仓库 <SEP> 电子 商务 仓库 <SEP> 淘宝 发货 <SEP> 解决 问题 <SEP> 办公 管理 <EOS>
<sep>客户 ps 管理 文案 网络 直通车 问题 产品 时间 传播 意识 流程 规章 组织 审美 家居 创意 心理 撰写 计算机 渠道 沟通 网站 住宿 方案 责任心 架构 资源 仓库 营销 英语 力佳 高中 基础 思考 逻辑 图片 表达 平台 仓储 我们 计划 介绍 推广 创造性 办公 实施 策划 整体 玩法 语言 操作 网店 感知 品牌 危机 金蝶 oem 策略 团队者 社交 一起 阿里系 电脑 业务 应变 思维 能力 招聘 电子 口头 行业 技巧 执行 交易 京东店 管控 文字 组建 分析 市场 搜索引擎 工作 说服 精神 促销 压力 淘宝 设备 天猫 概念 页面 亲和力 京东 销售 运作 方式 处理 excel 商品 维护 总结 领悟 软件 团队 手法 执行力 公关 服务 商务 责任感 发货 紧急 学习 系统 数据 规则 风险 交流 美工 理解 设计 敏感度 运营 经管 广告 协调
<sep>1 、 中专 及 以上 学历 ， 23-38 岁 ；  2 、 1 年 以上 天猫 、 淘宝 仓库 等 电子 商务 仓库 工作 经验 ， 熟悉 淘宝 发货 、 出入库 流程 ；  3 、 较 强 的 解决 问题 和 沟通 的 能力 ， 熟练 使用 基本 的 办公 管理 软件 ；  4 、 具备 较 强 的 责任心 和 团队 精神 ， 能 吃苦耐劳 ， 能 承受 一定 工作 压力 。 <EOS>
"""

"""
1 、 负责 UPS 产品 国外 市场 产品 需求 分析 、 国外 业务 拓展 、 国外 业务 业绩 达成 ； 开拓 符合 公司 策略 方向 的 新 市场 、 新 客户 ， 挖掘 市场 商机 和 需求 并 向 内 转换 成 新 产品 及 订单 ；  2 、 按时 按质 完成 经营 指标 及 目标 ；  3 、 收集 并 充分 分析 市场 竞争 数据 ， 制定 竞争 策略 且 能 有效 执行 ；  4 、 开拓 并 深耕 客户 关系 ， 具有 良好 的 客户 关系 及 服务 意识 。
<sep>本科 <SEP> 
<sep>财务 预测 客户 管理 听说 问题 自动化 意识 审美 创意 渠道 沟通 谈判 英文 开拓 英语 方案 预算 规划 资源 营销 翻译 研究 空间 基础 洞察 综合 计划 关系 推广 办公 实施 酒店 策划 税务 品牌 标准 金蝶 策略 核算 提案 外贸 业务 思维 能力 招聘 技巧 执行 文字 观察 临场应变 分析 市场 职业 精神 建设 设备 复杂 项目 定位 奉献 会计 技能 决策 销售 运作 处理 维护 软件 执行力 商务 书面 学习 系统 战略 调查 交流 指导 开发 解决 运营 注册 广告 协调
<sep>1 、 本科 及 以上 学历 ， 电子 类 、 外贸 类 相关 专业 ；  2 、 5 年 及 以上 UPS 产品 外销 经验 ， 2 年 及 以上 外贸 区域 经理 或 同等 职位 经验 。  3 、 具有 良好 的 沟通 和 表达 能力 ， 具备 英语 四 级 及 以上 水平 ， 能 与 国外 客户 进行 日常 书面 及 口语 交流 ；  4 、 具有 较 强 的 业务 开拓 能力 ， 能 独立 开拓 新 客户 ， 维护 客户 基层 到 中 高层 人员 的 关系 。 <EOS>

"""
"""
[统招全日制大学本科及以上学历。', 
'3-5年以上风险计量模型开发及实施经验，从事过信用评级、模型开发验证实施、数据质量管理、风险报告、风险监测、组合分析等方面工作。', 
'精通SAS、Python等工具及机器学习各算法，具备较强的数理统计相关知识背景。', 
'熟悉信用卡系统、数据及业务管理。', 
'具备良好的项目管理能力，沟通及展现能力。']

统招 全日制 大学 本科<SEP>
风险 计量 模型 开发<SEP>实施<SEP>
精通 SAS Python 工具<SEP>
机器 学习 算法<SEP>
具备 数理 统计 知识 背景<SEP>
信用卡 系统 数据<SEP>
业务 管理<SEP>
具备 项目 管理<SEP>
沟通 展现
"""