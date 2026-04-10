import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests

from src.preprocess.data_loader import get_label_names

names = get_label_names()

print("Looking up EuroVoc labels via SPARQL...\n")

# Batch query via EU SPARQL endpoint
SPARQL_URL = "https://publications.europa.eu/webapi/rdf/sparql"

# Build query for all 100 concepts
values = " ".join([f"<http://eurovoc.europa.eu/{cid}>" for cid in names])
query = f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?concept ?label WHERE {{
    VALUES ?concept {{ {values} }}
    ?concept skos:prefLabel ?label .
    FILTER(LANG(?label) = "en")
}}
"""

try:
    resp = requests.get(SPARQL_URL, params={"query": query, "format": "application/json"}, timeout=30)
    data = resp.json()
    
    # Parse results
    id_to_name = {}
    for row in data["results"]["bindings"]:
        uri = row["concept"]["value"]
        label = row["label"]["value"]
        cid = uri.split("/")[-1]
        id_to_name[cid] = label
    
    print(f"Found {len(id_to_name)} labels from SPARQL\n")
    
    label_map = {}
    for i, cid in enumerate(names):
        readable = id_to_name.get(cid, f"UNKNOWN_{cid}")
        label_map[str(i)] = {"id": cid, "name": readable}
        print(f"{i:3d}: {cid} -> {readable}")

except Exception as e:
    print(f"SPARQL failed: {e}")
    print("\nFalling back to manual lookup...")
    
    label_map = {}
    for i, cid in enumerate(names):
        url = f"https://publications.europa.eu/resource/authority/eurovoc/{cid}"
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                readable = str(data)[:80]
            else:
                readable = f"UNKNOWN_{cid}"
        except:
            readable = f"UNKNOWN_{cid}"
        
        label_map[str(i)] = {"id": cid, "name": readable}
        print(f"{i:3d}: {cid} -> {readable}")
        time.sleep(0.3)

# Save
os.makedirs("data/annotations", exist_ok=True)
with open("data/annotations/eurovoc_label_names.json", "w") as f:
    json.dump(label_map, f, indent=2)
print(f"\nSaved to data/annotations/eurovoc_label_names.json")