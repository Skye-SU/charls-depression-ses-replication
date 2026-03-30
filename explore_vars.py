import pyreadstat
import pandas as pd
import json

BASE = "/Users/xiwen/Documents/Career_Job/大厂简历/RA投递材料包/定量统计/CHARLS"
found_vars = {}

def get_cols(filename, keyword_list):
    try:
        df, meta = pyreadstat.read_dta(f"{BASE}/{filename}.dta")
        matched = {k: v for k, v in meta.column_names_to_labels.items() 
                   if any(kw.lower() in str(v).lower() for kw in keyword_list) or any(kw.lower() in k.lower() for kw in keyword_list)}
        found_vars[filename.split('/')[0]] = matched
    except Exception as e:
        found_vars[filename.split('/')[0]] = str(e)

get_cols("demographic_background/demographic_background", ["age", "gender", "educ", "marital", "hukou", "rural", "birth", "ba0", "bd", "be"])
get_cols("health_status_and_functioning/health_status_and_functioning", ["ces", "depres", "sad", "hopeful", "pain", "chronic", "adl", "iadl", "child", "health", "fall", "memory", "know", "dc0", "da", "db"])
get_cols("household_income/household_income", ["expend", "pce", "consum", "income", "total"])
get_cols("biomarker/biomarker", ["leg", "knee", "height"])
get_cols("weight/weight", ["weight"])
get_cols("family_information/family_information", ["die", "death", "pass", "parent", "child", "spouse"])

with open("/Users/xiwen/.gemini/antigravity/scratch/intelligence/charls_replication/var_meta.json", "w") as f:
    json.dump(found_vars, f, indent=2, ensure_ascii=False)
