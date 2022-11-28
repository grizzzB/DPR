import warnings
warnings.simplefilter(action='ignore', category=Warning)

import elasticsearch
from elasticsearch import Elasticsearch, helpers, AsyncElasticsearch
from hjb_constant import QUERY
import numpy as np
import glob
import os
import json
import copy
import argparse
from tqdm import tqdm

len_dict = {
    "korquad1": 10645,
    "minds_news_mrc" : 47314,
    "book_mrc": 237500,
    "news_mrc": 125964,
    "admin_mrc": 75917
}

class ESModifier:
    def __init__(self, hosts: list = ["http://localhost:9200"]) -> None:
        self.es = Elasticsearch(hosts, timeout=3000)

    def search(self, ):
        pass

    def get_whole_docs(self, index: str):
        return( doc for doc in helpers.scan(self.es,
                                            index=index,
                                            scroll         = "1000m",
                                            raise_on_error = False,
                                            preserve_order = True,
                                            query=QUERY["all"]))


    def get_positives(self, answ:str, ques:str, index:str ):
        query = copy.copy(QUERY['positive'])
        ctxs= []

        new_q = query["query"]["bool"]["must"][0]["wildcard"]["context.keyword"].format(query=answ)
        query["query"]["bool"]["must"][0]["wildcard"]["context.keyword"] = new_q
        query["query"]["bool"]["must"][1]["multi_match"]["query"]=ques

        res = self.es.search(index=index, body=query)
        try:
            for doc in res['hits']['hits']:
                tmp = {}
                tmp['title'] = doc["_source"]["title"]
                tmp['text'] = doc["_source"]["context"]
                tmp['psg_id'] = doc["_id"]
                tmp['score'] = float(doc["_score"])
                tmp['title_score'] = 0
                ctxs.append(tmp)
        except:
            print(f"No res?? : {len(res['hits']['hits'])}")
            print(query)

        return ctxs


    def get_hard_negatives(self, answ:str, ques:str, index:str ):
        query = copy.copy(QUERY['hard_negative'])
        ctxs= []

        new_q = query["query"]["bool"]["must_not"][0]["wildcard"]["context.keyword"].format(query=answ)
        query["query"]["bool"]["must_not"][0]["wildcard"]["context.keyword"] = new_q
        query["query"]["bool"]["should"][0]["multi_match"]["query"]=ques

        res = self.es.search(index=index, body=query)
        try:
            for doc in res['hits']['hits']:
                tmp = {}
                tmp['title'] = doc["_source"]["title"]
                tmp['text'] = doc["_source"]["context"]
                tmp['psg_id'] = doc["_id"]
                tmp['score'] = float(doc["_score"])
                tmp['title_score'] = 0
                ctxs.append(tmp)
        except:
            print(f"No res?? : {len(res['hits']['hits'])}")
            print(query)

        return ctxs


    def update_es_document(self, index:str, doc_id:dict, offset:int, origin_id: str ):
        tmp_doc= {}
        tmp_doc["origin_id"] = origin_id
        tmp_doc["offset"] = str(offset)

        body = {
            "doc": tmp_doc
        }

        res = self.es.update(index=index, id=str(doc_id), body=body)
        return res


def split_passage(passage: str, max_len: int =120, min_len: int = 10):
    # whitespace 기준으로 길이 측정
    tokens = passage.split(" ")
    sub_passg = []
    offset_list = [0]
    for i in range(0, len(tokens), max_len):
        sub_passg.append(tokens[i:i+max_len])
        char_passg = offset_list[-1] + len(" ".join(sub_passg[-1]))
        offset_list.append(char_passg)

    return sub_passg, offset_list


def check_answer_valid(answ_start: int, answer:str, passg_offset: tuple):
    # 휴리스틱으로 offset(새로운 시작점) - answ_start_offset < 5 일떄 제거?
    # 건 수 체크.
    passage_range = range(*passg_offset)
    end = answ_start + len(answer)
    if answ_start in passage_range and end in passage_range:
        return True
    else:
        return False

def create_split_documets():
    # max_len으로 split 한다
    # full에서 offset 값을 char 단위로 저장
    # answer가 잘리거나 제외됐는지 체크
    # es.update.

    pass

def dump_json(jsonsl: list, out_path: str):
    with open(out_path, "w", encoding="utf8") as fw:
        json.dump(jsonsl, fw, indent=4,  
                              separators=(',', ': '))
    

def create_json_dataset(index:str, out_path: str):
    out_jsonsl = []
    cnt = 0
    es_obj = ESModifier()
    data_iter = es_obj.get_whole_docs(index=index)

    for doc in data_iter:
        answ_doc = {}
        positive_ctxs = []

        # add default answer from source data.
        answ_doc['title'] = doc["_source"]["title"]
        answ_doc['text'] = doc["_source"]["context"]
        answ_doc['psg_id'] = doc["_id"]
        answ_doc['score'] = 1000
        answ_doc['title_score'] = 1
        positive_ctxs.append(answ_doc)

        # additional positive and hord-negatives
        
        for qa in doc["_source"]["qas"]:
            out_json = {}
            ques = qa["question"]

            if type(qa["answers"]) is dict:
                qa["answers"] = [qa["answers"]]
            for a in qa["answers"]:
                hard_negative_ctxs = []
                answ = a['text']
                #answ_offset = a['answer_start']
                try:
                    add_positives = es_obj.get_positives(answ=answ, ques=ques, index=index)
                    hard_negative_ctxs = es_obj.get_hard_negatives(answ=answ, ques=ques, index=index)
                except elasticsearch.NotFoundError as e:
                    print(e)
                    eixt()

                out_json["dataset"] = index
                out_json["question"] = ques
                out_json["answers"] = answ
                out_json["positive_ctxs"] =positive_ctxs + add_positives
                out_json["hard_negative_ctxs"] = hard_negative_ctxs
                out_json["negative_ctxs"] =[]
                out_jsonsl.append(out_json)
            if len(out_jsonsl) % 10000 == 0:
                print(f"{index} is done for 10000 documents.")
                dump_json(out_jsonsl, os.path.join(out_path, f"{index}_{cnt}.json"))
                cnt += 1
                out_jsonsl = []
          

from multiprocessing import Process

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--indices', nargs="+",default=["korquad1"])
parser.add_argument('--output', type=str,   default="./")

if __name__ == '__main__':
    #indices=["korquad1"]

    args = parser.parse_args()
    print(f"argumets: {args}")
    indices=args.indices

    assert os.path.isdir(args.output), f"No valid path. {args.output}"

    p_list = []
    for idx in indices:
        out_p = f'./{idx}.json'
        p_list.append(Process(target = create_json_dataset, args=(idx, args.output)))
        p_list[-1].start()
    for p in p_list:
        p.join()

 