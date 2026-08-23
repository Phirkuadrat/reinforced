from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ane import recommender, df_all_recommendation_base, run_query
import uvicorn

app = FastAPI(title="REINFORCED Recommendation API")

# Allow CORS for Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/dosen")
def get_dosen_list():
    if df_all_recommendation_base is None:
        raise HTTPException(status_code=500, detail="Model rekomendasi belum siap atau gagal dimuat.")
    
    peneliti_raw = df_all_recommendation_base["Peneliti"].unique()
    
    dosen_list = []
    import re
    for p in peneliti_raw:
        match = re.search(r"^(.*?)\s+\(SINTA:\s*(\d+)\)", p)
        if match:
            dosen_list.append({
                "nama": match.group(1).strip(),
                "sinta_id": match.group(2).strip()
            })
            
    dosen_list.sort(key=lambda x: x["nama"])
    return {"status": "success", "data": dosen_list}

@app.get("/api/rekomendasi")
def get_recommendation(name: str, use_cascading: bool = True):
    try:
        # Panggil fungsi recommender bawaan ane.py
        df_result = recommender(name, use_cascading=use_cascading)
        
        if df_result.empty:
            return {"status": "success", "data": []}
        
        result_list = df_result.to_dict('records')
        
        # Ekstrak semua SINTA ID (target + semua rekomendasi)
        target_sinta = result_list[0]['SINTA_ID'] if len(result_list) > 0 else None
        rekom_sinta_list = [str(r['Rekomendasi_SINTA_ID']) for r in result_list]
        all_sinta = rekom_sinta_list + ([str(target_sinta)] if target_sinta else [])
        
        # Konversi ke string format untuk query Cypher (contoh: ['123', '456'])
        sinta_in_query = "[" + ", ".join([f"'{s}'" for s in set(all_sinta)]) + "]"
        
        # Query detail dosen ke Neo4j
        query = f"""
        MATCH (p:ns0__Person)
        WHERE p.ns0__hasSintaID IN {sinta_in_query}
        OPTIONAL MATCH (p)-[:ns0__hasPublication]->(pub:ns0__Publication)
        RETURN 
            p.ns0__hasSintaID AS sinta_id, 
            p AS properties, 
            collect(pub.ns0__hasTitle) AS publications
        """
        
        df_detail = run_query(query)
        
        detail_dict = {}
        for _, row in df_detail.iterrows():
            sid = str(row['sinta_id'])
            
            # Safely extract properties (dict) and publications (list)
            props = dict(row['properties']) if isinstance(row['properties'], dict) else {}
            pubs = list(row['publications']) if isinstance(row['publications'], list) else []
            
            # Hapus element list kosong
            pubs = [p for p in pubs if p]
            
            detail_dict[sid] = {
                "statistik": props,
                "publikasi": pubs
            }
            
        # Gabungkan data ke result_list
        for r in result_list:
            rsid = str(r['Rekomendasi_SINTA_ID'])
            if rsid in detail_dict:
                r['Detail_Statistik'] = detail_dict[rsid]['statistik']
                r['Detail_Publikasi'] = detail_dict[rsid]['publikasi']
                
        return {"status": "success", "data": result_list}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph")
def get_graph_data(target_name: str, rekom_names: str):
    try:
        names_list = [name.strip() for name in rekom_names.split(',') if name.strip()]
        if not names_list:
            return {"status": "success", "data": {"nodes": [], "edges": []}}
            
        # Bangun query UNION dari semua rekomendasi
        query_blocks = []
        for nama_rekom in names_list:
            block = f"""
            MATCH path = shortestPath(
                (p1:ns0__Person {{ns0__hasName: '{target_name}'}})-[:collaborateWith*1..10]-
                (p2:ns0__Person {{ns0__hasName: '{nama_rekom}'}})
            )
            RETURN nodes(path) AS nodes, relationships(path) AS rels
            """
            query_blocks.append(block)

        full_query = "\nUNION\n".join(query_blocks)
        result = run_query(full_query)
        
        nodes_dict = {}
        edges_list = []
        added_edges = set()
        
        if result.empty:
            # Fallback query: jika tidak ada path collaborateWith
            rekom_in_query = "[" + ", ".join([f"'{s}'" for s in set(names_list)]) + "]"
            fallback_query = f"""
            MATCH (target:ns0__Person {{ns0__hasName: '{target_name}'}})
            WITH target
            MATCH (rekom:ns0__Person)
            WHERE rekom.ns0__hasName IN {rekom_in_query}
            RETURN collect(DISTINCT target) AS target_nodes, collect(DISTINCT rekom) AS rekom_nodes
            """
            result = run_query(fallback_query)
            if not result.empty:
                target_nodes = result.iloc[0]["target_nodes"] if isinstance(result.iloc[0]["target_nodes"], list) else []
                rekom_nodes = result.iloc[0]["rekom_nodes"] if isinstance(result.iloc[0]["rekom_nodes"], list) else []
                
                t_id = ""
                # Extract Target
                if target_nodes and len(target_nodes) > 0:
                    t_node = target_nodes[0]
                    t_id = str(t_node.get("ns0__hasSintaID", target_name))
                    nodes_dict[t_id] = {
                        "id": t_id,
                        "label": t_node.get("ns0__hasName", target_name),
                        "group": "target"
                    }
                    
                # Extract Rekom
                for r_node in rekom_nodes:
                    r_id = str(r_node.get("ns0__hasSintaID", r_node.get("ns0__hasName", "")))
                    r_name = r_node.get("ns0__hasName", "")
                    nodes_dict[r_id] = {
                        "id": r_id,
                        "label": r_name,
                        "group": "recommendation"
                    }
                    if t_id:
                        edges_list.append({
                            "from": t_id,
                            "to": r_id,
                            "label": "recommended"
                        })
        else:
            for idx, row in result.iterrows():
                path_nodes = row["nodes"]
                path_rels = row["rels"]
                
                # Tambahkan node
                if isinstance(path_nodes, list):
                    for node in path_nodes:
                        name = node.get("ns0__hasName", "Unknown")
                        node_id = str(node.get("ns0__hasSintaID", name))
                        
                        group = "intermediate"
                        if name == target_name:
                            group = "target"
                        elif name in names_list:
                            group = "recommendation"
                            
                        if node_id not in nodes_dict:
                            nodes_dict[node_id] = {
                                "id": node_id,
                                "label": name,
                                "group": group
                            }
                            
                # Tambahkan edge (relationship)
                if isinstance(path_rels, list):
                    for rel in path_rels:
                        # Di neo4j driver, rel adalah Tuple (Node, Type, Node)
                        if isinstance(rel, tuple) and len(rel) == 3:
                            source_node, rel_type, target_node = rel
                            s_id = str(source_node.get("ns0__hasSintaID", source_node.get("ns0__hasName", "")))
                            t_id = str(target_node.get("ns0__hasSintaID", target_node.get("ns0__hasName", "")))
                            
                            edge_key = f"{s_id}-{t_id}-{rel_type}"
                            if edge_key not in added_edges:
                                edges_list.append({
                                    "from": s_id,
                                    "to": t_id,
                                    "label": rel_type
                                })
                                added_edges.add(edge_key)

        return {
            "status": "success",
            "data": {
                "nodes": list(nodes_dict.values()),
                "edges": edges_list
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
