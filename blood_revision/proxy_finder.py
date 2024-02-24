import requests
from typing import List
import random
import os
import json
import pandas as pd
from io import StringIO

def find_proxies(
    missing_rsIDs: List[str], 
    LDLink_API_token: str,
    proxies_json_path: str = None, 
    valid_coords: List[str] = [],
    valid_rsIDs: List[str] = [],
    max_bp_dist: int = int(1e5),
    min_R2: float = 0.5):
    
    if len(valid_coords) + len(valid_rsIDs) == 0:
        raise "Shoult at leas provide one of valid_RSIDs or valid_coords"
    
    assert all([":" in c and "chr" == c[:3] for c in valid_coords]), "valid_coords items should be formatted like 'chr1:123'"
    
    if proxies_json_path is not None and os.path.exists(proxies_json_path):
        print("Loading pre-existing proxies")
        with open(proxies_json_path,"r") as infile:
            proxies = json.load(infile)
    else:
        proxies = {}
    with requests.Session() as s:
        random.shuffle(missing_rsIDs)
        for i, m in enumerate(missing_rsIDs):
            if m in proxies:
                continue
            if i % 1 == 0:
                print("%d (%d) / %d" % (i,len(proxies),len(missing_rsIDs)))
                
            # First, check if m is in 1000G ref panel. If not, stop here
            #if not "1000Genomes" in s.get("https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/%d" % int(m[2:])).text:
            #    print("%s is not in reference panel (NCBI)" % m)
            #    proxies[m] = {"error":"not in reference panel (NCBI)"}
            URL = "https://ldlink.nih.gov/LDlinkRest/ldproxy"
            PARAMS = {
                "var":m,
                "pop":"GBR",
                "r2_d":"r2",
                "window":max_bp_dist,
                "genome_build":"grch37",
                "token":LDLink_API_token,
            }
            try:
                #time.sleep()
                print("Sending request for %s..." % m)
                r = s.get(url = URL, params = PARAMS,timeout=12*60)
                
            except requests.Timeout as e:
                print("Timeout %s" % m)
                proxies[m] = {"error":"timeout..."}
                continue
            try:
                snp = pd.read_csv(StringIO(r.text),sep="\t")
                candidate_proxies = snp.loc[snp.RS_Number.isin(valid_rsIDs) | snp.Coord.isin(valid_coords)]
                candidate_proxies = candidate_proxies.loc[candidate_proxies.R2 >= min_R2]
                if candidate_proxies.shape[0] == 0:
                    print("No proxy for %s" % m)
                    proxies[m] = {"error":"No proxy found"}
                else:
                    proxies[m] = {
                        "rsID":candidate_proxies.RS_Number.iloc[0],
                        "coords":candidate_proxies.Coord.iloc[0],
                        "R2":candidate_proxies.R2.iloc[0]
                    }
                    print("Found proxy for %s : %s" % (m, candidate_proxies.RS_Number.iloc[0]))
            except Exception as e:
                if "not in 1000G reference panel" in r.text:
                    print("%s is not in reference panel" % m)
                    proxies[m] = {"error":"not in reference panel"}
                elif "Concurrent API" in r.text:
                    print("Concurrent API requests not allowes. Wait some time")
                    continue
                else:
                    try:
                        print(r.text)
                        proxies[m] = {"error":"other error : %s" % json.loads(r.text)["error"]}
                    except:
                        raise e
            if proxies_json_path is not None:
                with open(proxies_json_path, "w") as outfile:
                    json.dump(proxies, outfile)
        
    
    return proxies