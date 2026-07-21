#!/usr/bin/env python3
"""
Cache UniProt Sequences and KEGG SMILES for all Small Molecule Effector TFs.
"""

import os
import csv
import json
import urllib.request
import urllib.parse

CACHE_FILE = 'data/processed/cache_sequences_smiles.json'

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        fasta = resp.read().decode('utf-8')
        lines = fasta.split('\n')
        return ''.join(lines[1:]).strip()

def fetch_smiles_from_kegg(kegg_id):
    # 1. PubChem PUG REST substance by KEGG ID
    try:
        url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/KEGG/{kegg_id}/cids/JSON"
        req1 = urllib.request.Request(url1, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req1) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            cid = data['InformationList']['Information'][0]['CID'][0]
            
            url_s = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES,ConnectivitySMILES,IsomericSMILES/JSON"
            req_s = urllib.request.Request(url_s, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req_s) as resp_s:
                props = json.loads(resp_s.read().decode('utf-8'))['PropertyTable']['Properties'][0]
                smiles = props.get('SMILES') or props.get('IsomericSMILES') or props.get('ConnectivitySMILES')
                if smiles:
                    return smiles
    except Exception:
        pass

    # 2. KEGG REST API fallback
    try:
        url_kegg = f"https://rest.kegg.jp/get/{kegg_id}"
        req_k = urllib.request.Request(url_kegg, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_k) as resp:
            lines = resp.read().decode('utf-8').split('\n')
            for line in lines:
                if 'PubChem:' in line:
                    cid = line.split('PubChem:')[1].strip().split()[0]
                    url_s = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES,ConnectivitySMILES,IsomericSMILES/JSON"
                    req_s = urllib.request.Request(url_s, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req_s) as resp_s:
                        props = json.loads(resp_s.read().decode('utf-8'))['PropertyTable']['Properties'][0]
                        smiles = props.get('SMILES') or props.get('IsomericSMILES') or props.get('ConnectivitySMILES')
                        if smiles:
                            return smiles
    except Exception:
        pass

    return None

def build_cache(raw_csv='data/raw/tf_effectors_curated_2607.csv'):
    cache = {'uniprot': {}, 'kegg_smiles': {}}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    with open(raw_csv, 'r', encoding='utf-8') as f:
        raw_rows = list(csv.DictReader(f))

    small_mols = [r for r in raw_rows if r.get('Effector type', '').strip() == 'small molecule']

    tf_uniprot = {}
    lig_kegg = {}
    for r in small_mols:
        tf = r['Transcription factor'].strip()
        uid = r['Uniprot ID'].strip()
        lig = r['effector_name'].strip()
        kegg = r['kegg_id'].strip()
        if tf and uid:
            tf_uniprot[tf] = uid
        if lig and kegg:
            lig_kegg[lig] = kegg

    print(f"Resolving {len(tf_uniprot)} UniProt sequences...")
    for tf, uid in tf_uniprot.items():
        if uid not in cache['uniprot'] or not cache['uniprot'][uid]:
            try:
                seq = fetch_uniprot_sequence(uid)
                if seq:
                    cache['uniprot'][uid] = seq
            except Exception as e:
                print(f"Warning: UniProt fetch failed for {tf} ({uid}): {e}")

    print(f"Resolving {len(lig_kegg)} KEGG SMILES...")
    failed_kegg = []
    for lig, kegg in lig_kegg.items():
        if kegg not in cache['kegg_smiles'] or not cache['kegg_smiles'][kegg]:
            s = fetch_smiles_from_kegg(kegg)
            if s:
                cache['kegg_smiles'][kegg] = s
            else:
                failed_kegg.append((lig, kegg))

    os.makedirs(os.path.dirname(os.path.abspath(CACHE_FILE)), exist_ok=True)
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

    print(f"Cache complete! {len(cache['uniprot'])} UniProt sequences, {len(cache['kegg_smiles'])} KEGG SMILES cached.")
    if failed_kegg:
        print(f"Failed KEGG IDs: {failed_kegg}")

if __name__ == '__main__':
    build_cache()
