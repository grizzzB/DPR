import re

CALL_DB = False
# for Multi search API.
CHUNK_SIZE = 8000

EXPLAIN = False
DEFAULT = "default"
GROUP_TOPK = 1
TOPK = 3
OPERATOR = "or"

WORK = "/code"

# file path.
FILE_DIR = "/resources/files/tmp/"
DUMP_FILE_PATH = f"{WORK}/files/"
RESC_DIR = "/resources"

CONFIG_PATH = {"dir": "{resc_dir}/es_idx_config/{index}",
               "mapping": "{resc_dir}/es_idx_config/{index}/mapping.json",
               "setting": "{resc_dir}/es_idx_config/{index}/setting.json"}

DICT_PATH = {"dir": "{resc_dir}/dictionary/{index}",
             "stopwords": "dictionary/{index}/stopwords.txt",
             "stoptags": "{resc_dir}/dictionary/{index}/stoptags.txt",
             "synonyms": "dictionary/{index}/synonyms.txt",
             "user": "dictionary/{index}/userDictionary.txt"}
# dict type
RESPONSE = {
    "result": DEFAULT
}
# elasticsearch setting
SNAP_REPO_NAME = "snap_repo"
SNAPSHOT = {
    "type": "fs",
    "settings": {
        "compress": True,
        "location": "/usr/share/elasticsearch/snapshots"
    }
}
SNAP_INDEX = {
  "indices": DEFAULT,
  "ignore_unavailable": True,
  "include_global_state": True
}

SNAP_NAME_DATE = "%Y%m%dt%H%M"

WORD_WEIGHT_REGEX = re.compile(r"weight\(.*:.* in \d*\)")
S_FIELDS = ["answCtt", "quesCtt", "tagCtt"]
TEST_RES_FILE_HEADER = {
    "dict": ["No.", "Rank", "Test sentence", "Correction", "User answer", "Title",
             "Word", "Answer", "_id", "Score", "Word weights"],
    "set": ["No.", "Rank", "Test sentence", "Correction", "User answer", "Title",
            "Question", "Answer", "_id", "Score", "Word weights"]
}
# attr{number} == ner tag 라고 가정. * 새로 srchDtm 검색 요청 시간 추가. 나머진 색인시 들어있는 필드값
OUTPUT_FIELDS = ["idxId", "idxSeq", "chnlId", "ctgrId", "chnlNm", "ctgrNm", "answVw"
                 "quesNo", "quesCtt", "answNo", "answCtt", "answer_view", "tagCtt",
                 "attr1", "attr2", "attr3", "attr4", "attr5", "attr6", "regDtm"]
SEARCH_FIELDS = ["query", "search_type", "search_fields", "q_weight", "a_weight", "tag_weight",
                 "quesNo", "quesCtt", "answNo", "answCtt", "answer_view", "tagCtt", "chnl_id", "ctgr_id", 
                 "session_id", "turn", "criteria", "usr_answer", "nlp_intents","nlp_intents"]



Q_WEIGHTS = 2
A_WEIGHTS = 1
TAG_WEIGHTS = 1

QUERY_SEQ = {
    "ngram": [("ngram", "and")],
    "nori": [("nori", "or"), ("nori", "and")],
    "keyword": [("keyword", "or")]
}


QUERY = {
    "all": {
        "query": {
            "match_all": {}
        }
    },
    "nori": {
        "query": {
            "multi_match": {
                "query": DEFAULT,
                "operator": OPERATOR,
                "fields": [f"question.nori^{Q_WEIGHTS}", f"answer.nori^{A_WEIGHTS}",
                           f"word.nori^{Q_WEIGHTS}", f"tags.ngram_token^{TAG_WEIGHTS}"]
            }
        },
        "size": TOPK,
        "explain": EXPLAIN
    },
    "ngram": {
        "query": {
            "multi_match": {
                "query": DEFAULT,
                "operator": OPERATOR,
                "fields": [f"question.ngram^{Q_WEIGHTS}", f"answer.ngram^{A_WEIGHTS}",
                           f"word.ngram^{Q_WEIGHTS}", f"tags.ngram_token^{TAG_WEIGHTS}"]
            }
        },
        "size": TOPK,
        "explain": EXPLAIN
    },
    "keyword": {
        "query": {
            "multi_match": {
                "query": DEFAULT,
                "operator": OPERATOR,
                "fields": ["answer.keyword", "question.keyword", "tags.keyword", "word.keyword"]
            }
        },
        "size": TOPK,
        "explain": EXPLAIN
    },
    "sort":{
        "_score" : {"order":"desc"}
    },
    "aggs": {
        "by_ids": {
          "terms": {
            "field": "{pivot_id}.keyword",
            "size": TOPK,
            "order": {"max_score": "desc"}
          },
          "aggs": {
            "by_top_hit": {"top_hits": {"size": GROUP_TOPK}},
            "max_score": {"max": {"script": "_score"}}
          }
        }
    },
    "multi_queris": {
        "query": {
            "bool": {
                "should": [

                ]
            }
        },
        "size": TOPK,
        "explain": EXPLAIN
    },
    "positive":{
        "query": {
            "bool": {
                "must":[
                    {
                        "wildcard": {
                            "context.keyword": "*{query}*"
                        }
                    },
                    {
                        "multi_match": {
                            "query":  DEFAULT,
                            "operator": "or",
                            "fields": ["title.nori","title.ngram","context.nori"]
                        }
                    }
                ]
            }
        },
        "size": TOPK,
        "explain": EXPLAIN
    },
    "hard_negative": {
        "query": {
            "bool": {
                "must_not":[
                    {
                        "wildcard": {
                            "context.keyword": "*{query}*"
                        }
                    }
                ],
                "should": [
                    {
                        "multi_match": {
                            "query": DEFAULT,
                            "operator": "or",
                            "fields": ["title.nori","title.ngram","context.nori"]
                        }
                    }
                    
                ]
            }
        },
        "size": TOPK,
        "explain":EXPLAIN
    }
}
