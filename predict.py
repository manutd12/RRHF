import sys
import codecs
sys.path.append('..')
# from textgen import LlamaModel
from textgen import ChatGlmModel
import codecs
from collections import defaultdict
import re
import jieba
import random



def collect_antou_predict(predict_file, is_have_ori_title):
    '''
    利用暗投宽表数据，组装成模型预测需要的输入格式。 返回：预测数据 、每条数据原始的信息流标题/描述
    is_have_ori_title:输入数据 中是否包含原始标题/描述字段
    '''
    input_lines = []
    predict_sentences = []
    predict_sentences_map = {}
    ori_titles = []
    kk=0
    with codecs.open(predict_file) as f:
        for line in f.readlines():
                line_list = line.strip("\n").split("\t")
                if is_have_ori_title:
                    if len(line_list) !=8:
                        continue
                    industry_name_level1,industry_name_level2,group_name,corporation_name,subject,platform,title,description = line_list
                else:
                    if len(line_list) !=6:
                        print(len(line_list))
                        print(line_list)
                        continue
                    industry_name_level1,industry_name_level2,group_name,corporation_name,subject,platform = line_list
                    title,description = "",""
                if industry_name_level1 == "" :
                    continue

                subject = " ".join(set([item for item in subject.split("==") if item.strip()!=""]))
                platform = " ".join(set([item for item in platform.split("==") if item.strip()!=""])) if platform!="" else ""    
                op_group_name = platform if platform!="" else group_name # 客户简称          
                query1 = subject if subject!="" else platform
                query2 = subject+" "+platform if platform!="" else subject
                query_list = set()
                if  query1.strip() !="":
                    query_list.add(query1.strip())
                if query2.strip() !="":
                    query_list.add(query2.strip())

                kk+=1

           


                for query in query_list:
                    for flow in ["xq","sys"]:
                        input_promt = "对于给定的搜索query和广告主信息，生成高点击率的广告，每条广告需包含关键词、标题、描述，给定信息如下：\n" + "query："+query+"\n" + "流量："+flow+"\n" + "广告主行业："+industry_name_level1+"-"+industry_name_level2+"\n"+"集团名称："+group_name +"\n"+"客户简称："+op_group_name+"\n"+"公司名称："+corporation_name+"\n"
                        if input_promt in predict_sentences_map:
                            continue
                        input_lines.append("\t".join(line_list+[flow]))
                        predict_sentences.append(input_promt)
                        predict_sentences_map[input_promt] = 1
                        ori_titles.append([title,description])
    print("kk:{}".format(kk))
    return predict_sentences,ori_titles,input_lines



def save_test_res(data_type, predict_sentences, predict_res, ori_titles):
    with codecs.open(data_type+"_rrhf_predict_res_subject_"+str(args["temperature"])+"_"+str(args["top_p"])+"_"+str(args["repetition_penalty"])+".txt","w",encoding="utf-8") as f_save:
        for s, r, o in zip(predict_sentences, predict_res,ori_titles):
            f_save.write(s+"\n"+"生成结果为：\n"+r+"\n"+"原始标题描述为:\n"+o[0]+"\n"+o[1]+"\n\n")

def save_pre_res(data_type, predict_sentences, predict_res, ori_titles, input_lines):
    with codecs.open(data_type+"_rrhf_predict_res_subject_"+str(args["temperature"])+"_"+str(args["top_p"])+"_"+str(args["repetition_penalty"])+".txt","w",encoding="utf-8") as f_save:
        for s, r, o, l in zip(predict_sentences, predict_res,ori_titles, input_lines):
            f_save.write("input:"+"\t"+l+"\n"+r+"\n\n")



predict_rrhf = True

args={  "eval_batch_size": 16,
        "do_sample":True, "temperature":0.95, "top_p":0.7,"repetition_penalty":1.05,"top_k":50,"length_penalty":1.0,
        "num_beams":1, "num_return_sequences": 1  # num_beams=1, do_sample=True:使用随机采样； num_beams>1，do_sample=True：使用随机采样从num个beams中采样  num_return_sequences必须<=num_beams
                      } 

if predict_rrhf: #使用rrhf预测后的模型
    file_name = "rrhf"
    # model = LlamaModel("llama", "/apdcephfs/private_curvasong/ad_llm/llama_sogou_ad/output/merge_lora_llama", peft_name="/apdcephfs/private_curvasong/output/ad_llama_rrhf") # LLama
    
    # model = ChatGlmModel("chatglm", "/apdcephfs_cq3/share_2973545/mingyyi/chatglm_6b_lora_sogou_q2ad/merge_lora_chatglm", peft_name="/apdcephfs/private_curvasong/output/ad_glm_rrhf_ddp") # q2akt
    model = ChatGlmModel("chatglm", "/apdcephfs_cq3/share_2973545/mingyyi/chatglm_6b_lora_sogou_qa2kt_6epoch/merge_lora_chatglm", peft_name="/apdcephfs/private_curvasong/output/ad_glm_rrhf_ddp_datav3",args=args) # qa2kt
    
else:  #使用原始的底座模型
    file_name = "original"
    # model = LlamaModel("llama", "/apdcephfs_cq3/share_2973545/data/models/shibing624/chinese-alpaca-plus-7b-hf", peft_name="/apdcephfs/private_curvasong/ad_llm/llama_sogou_ad/output") # llama
    model = ChatGlmModel("chatglm", "/apdcephfs_cq3/share_2973545/data/models/THUDM-chatglm-6b", peft_name="/apdcephfs_cq3/share_2973545/mingyyi/chatglm_6b_lora_sogou_q2ad")



    
antou_predict_sentences, antou_ori_titles, input_lines = collect_antou_predict("/apdcephfs/private_curvasong/RRHF/data/暗投预测数据/predict_07_0203.csv",is_have_ori_title=False)
print("暗投预测数据量：{}".format(len(antou_predict_sentences)))

length = len(antou_predict_sentences)
antou_res = model.predict(antou_predict_sentences[:length//2])
save_pre_res("暗投_rrhf_v3_1",antou_predict_sentences[:length//2], antou_res, antou_ori_titles[:length//2], input_lines[:length//2])


antou_res = model.predict(antou_predict_sentences[length//2:])
save_pre_res("暗投_rrhf_v3_2",antou_predict_sentences[length//2:], antou_res, antou_ori_titles[length//2:], input_lines[length//2:])




