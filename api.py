from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from ane import recommender, df_all_recommendation_base, run_query
import uvicorn
import pandas as pd
from collections import defaultdict

app = FastAPI(title="REINFORCED Recommendation API")

# Allow CORS for Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API untuk mengambil daftar seluruh dosen (beserta SINTA ID)
@app.get("/api/dosen")
def get_dosen_list():
    query = """
    MATCH (p:ns0__Person)
    WHERE p.ns0__hasName IS NOT NULL AND p.ns0__hasSintaID IS NOT NULL
    RETURN 
        p.ns0__hasName AS nama,
        p.ns0__hasSintaID AS sinta_id,
        p.ns0__hasDepartment AS departemen
    ORDER BY nama ASC
    """
    try:
        df = run_query(query)
        dosen_list = []
        for _, row in df.iterrows():
            dosen_list.append({
                "nama": row["nama"],
                "sinta_id": row["sinta_id"],
                "departemen": row["departemen"] if pd.notnull(row["departemen"]) else ""
            })
        return {"status": "success", "data": dosen_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j Error: {str(e)}")

# API untuk mengambil daftar publikasi dari satu dosen berdasarkan SINTA ID
@app.get("/api/publikasi")
def get_publikasi(sinta_id: str):
    query = f"""
    MATCH (p:ns0__Person {{ns0__hasSintaID: '{sinta_id}'}})-[:ns0__hasPublication]->(pub:ns0__Publication)
    RETURN pub.ns0__hasTitle AS judul
    """
    try:
        df_pub = run_query(query)
        if df_pub.empty:
            return {"status": "success", "sinta_id": sinta_id, "data": []}
        
        pubs = df_pub['judul'].tolist()
        return {"status": "success", "sinta_id": sinta_id, "data": pubs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                

        # ---------------------------------------------------------
        # GENERATE GRAPH DATA
        # ---------------------------------------------------------
        target_name_graph = name
        names_list = [r['Rekomendasi_Nama'] for r in result_list]
        
        nodes_dict = {}
        edges_list = []
        
        if names_list:
            query_blocks = []
            for nama_rekom in names_list:
                block = f"""
                MATCH path = shortestPath(
                    (p1:ns0__Person {{ns0__hasName: '{target_name_graph}'}})-[:collaborateWith*1..10]-
                    (p2:ns0__Person {{ns0__hasName: '{nama_rekom}'}})
                )
                RETURN nodes(path) AS nodes, relationships(path) AS rels
                """
                query_blocks.append(block)

            full_query = "\nUNION\n".join(query_blocks)
            result = run_query(full_query)
            
            added_edges = set()
            
            if result.empty:
                # Fallback query
                rekom_in_query = "[" + ", ".join([f"'{s}'" for s in set(names_list)]) + "]"
                fallback_query = f"""
                MATCH (target:ns0__Person {{ns0__hasName: '{target_name_graph}'}})
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
                    if target_nodes and len(target_nodes) > 0:
                        t_node = target_nodes[0]
                        t_id = str(t_node.get("ns0__hasSintaID", target_name_graph))
                        nodes_dict[t_id] = {
                            "id": t_id,
                            "label": t_node.get("ns0__hasName", target_name_graph),
                            "group": "target"
                        }
                        
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
                    
                    if isinstance(path_nodes, list):
                        for node in path_nodes:
                            n_name = node.get("ns0__hasName", "Unknown")
                            node_id = str(node.get("ns0__hasSintaID", n_name))
                            
                            group = "intermediate"
                            if n_name.lower() == target_name_graph.lower():
                                group = "target"
                            elif n_name in names_list:
                                group = "recommendation"
                                
                            if node_id not in nodes_dict:
                                nodes_dict[node_id] = {
                                    "id": node_id,
                                    "label": n_name,
                                    "group": group
                                }
                                
                    if isinstance(path_rels, list):
                        for rel in path_rels:
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
            "data": result_list,
            "graph": {
                "nodes": list(nodes_dict.values()),
                "edges": edges_list
            }
        }

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# ---------------------------------------------------------
# API untuk Menyimpan Penilaian (Evaluasi) dari Frontend
# ---------------------------------------------------------
class EvaluationItem(BaseModel):
    nama_rekomendasi: str
    nilai: int

class EvaluationRequest(BaseModel):
    target_name: str
    evaluations: List[EvaluationItem]
    komentar: Optional[str] = None
    metode: Optional[str] = "Cascading Hybrid"


# ---------------------------------------------------------
# ENDPOINT BARU UNTUK LARAVEL (Dosen, Evaluasi, Departemen, Jaringan)
# ---------------------------------------------------------

@app.get("/api/dosen/detail")
def get_dosen_detail(sinta_id: str):
    query = f"""
    MATCH (p:ns0__Person {{ns0__hasSintaID: '{sinta_id}'}})
    RETURN
        p.ns0__hasSintaID        AS hasSintaID,
        p.ns0__hasName           AS hasName,
        p.ns0__hasDepartment     AS hasDepartment,
        toInteger(p.ns0__hasAcademicAge)              AS hasAcademicAge,
        toInteger(p.ns0__hasCollaborator)             AS hasCollaborator,
        toFloat(p.ns0__hasAverageCitationScholar)     AS hasAverageCitationScholar,
        toFloat(p.ns0__hasAverageCitationScopus)      AS hasAverageCitationScopus,
        toFloat(p.ns0__hasAverageCitationWos)         AS hasAverageCitationWos,
        toInteger(p.ns0__hasHIndexScholar)            AS hasHIndexScholar,
        toInteger(p.ns0__hasHIndexScopus)             AS hasHIndexScopus,
        toInteger(p.ns0__hasHIndexWos)                AS hasHIndexWos,
        toInteger(p.ns0__hasPublicationScholar)       AS hasPublicationScholar,
        toInteger(p.ns0__hasPublicationScopus)        AS hasPublicationScopus,
        toInteger(p.ns0__hasPublicationWos)           AS hasPublicationWos
    """
    try:
        df = run_query(query)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Dosen dengan SINTA ID {sinta_id} tidak ditemukan.")
        row = df.iloc[0].where(df.iloc[0].notna(), other=None).to_dict()
        return {"status": "success", "data": row}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluasi")
def get_evaluasi(use_cascading: bool = True):
    if df_all_recommendation_base is None:
        raise HTTPException(status_code=500, detail="Model rekomendasi belum siap atau gagal dimuat.")
    try:
        from ane import evaluation
        result = evaluation(use_cascading=use_cascading)
        result["metode"] = "Cascading Hybrid" if use_cascading else "Standard ANE"
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/departemen")
def get_departemen():
    query = """
    MATCH (p:ns0__Person)
    WHERE p.ns0__hasDepartment IS NOT NULL
    RETURN DISTINCT p.ns0__hasDepartment AS departemen
    ORDER BY departemen
    """
    try:
        df = run_query(query)
        if df.empty:
            return {"status": "success", "data": []}
        return {"status": "success", "data": df["departemen"].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jaringan/full")
def get_full_graph(departemen: Optional[str] = None):
    if departemen and departemen.strip():
        filter_clause = f"WHERE p1.ns0__hasDepartment = '{departemen}' AND p2.ns0__hasDepartment = '{departemen}'"
    else:
        filter_clause = ""
    query = f"""
    MATCH (p1:ns0__Person)-[:collaborateWith]->(p2:ns0__Person)
    {filter_clause}
    RETURN
        p1.ns0__hasSintaID    AS from_sinta_id,
        p1.ns0__hasName       AS from_name,
        p1.ns0__hasDepartment AS from_dept,
        p2.ns0__hasSintaID    AS to_sinta_id,
        p2.ns0__hasName       AS to_name,
        p2.ns0__hasDepartment AS to_dept
    """
    try:
        df = run_query(query)
        nodes_dict = {}
        edges = []
        for _, row in df.iterrows():
            for sid_key, name_key, dept_key in [
                ("from_sinta_id", "from_name", "from_dept"),
                ("to_sinta_id", "to_name", "to_dept")
            ]:
                sid = str(row[sid_key])
                if sid not in nodes_dict:
                    nodes_dict[sid] = {
                        "id": sid,
                        "label": row[name_key],
                        "group": "connector",
                        "department": row[dept_key]
                    }
            edges.append({
                "from": str(row["from_sinta_id"]),
                "to": str(row["to_sinta_id"]),
                "label": "collaborateWith"
            })
        return {
            "status": "success",
            "data": {
                "nodes": list(nodes_dict.values()),
                "edges": edges
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



import sqlite3
import os

DB_PATH = os.path.join("database", "evaluasi.db")

@app.on_event("startup")
def startup_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS penilaian_user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_name TEXT NOT NULL,
            rekom_name TEXT NOT NULL,
            score INTEGER NOT NULL,
            komentar TEXT,
            metode TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

@app.post("/api/penilaian")
def submit_penilaian(request: EvaluationRequest):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        komentar = request.komentar.strip() if request.komentar else ""
        target_name = request.target_name.strip()
        
        # Cek apakah target_name, rekom_name, dan metode ini spesifik sudah dinilai
        for eval_item in request.evaluations:
            rekom_name = eval_item.nama_rekomendasi.strip()
            cursor.execute(
                "SELECT COUNT(*) FROM penilaian_user WHERE target_name = ? AND rekom_name = ? AND metode = ?", 
                (target_name, rekom_name, request.metode)
            )
            if cursor.fetchone()[0] > 0:
                conn.close()
                raise HTTPException(
                    status_code=400, 
                    detail=f"Rekomendasi '{rekom_name}' menggunakan metode '{request.metode}' sudah pernah dinilai."
                )
        
        # Jika lolos cek, masukkan ke database
        for eval_item in request.evaluations:
            cursor.execute(
                "INSERT INTO penilaian_user (target_name, rekom_name, score, komentar, metode) VALUES (?, ?, ?, ?, ?)",
                (target_name, eval_item.nama_rekomendasi.strip(), eval_item.nilai, komentar, request.metode)
            )
            
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Penilaian berhasil disimpan ke database."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rekap-penilaian")
def get_rekap_penilaian():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT target_name FROM penilaian_user ORDER BY created_at DESC")
        targets = cursor.fetchall()
        result = []
        for t in targets:
            target_name = t["target_name"]
            cursor.execute("""
                SELECT rekom_name, score, komentar, metode
                FROM penilaian_user
                WHERE target_name = ?
                ORDER BY id ASC
            """, (target_name,))
            rows = cursor.fetchall()
            if not rows:
                continue
            rekomendasi = [
                {
                    "nama": r["rekom_name"], 
                    "score": r["score"], 
                    "komentar": r["komentar"] or "",
                    "metode": r["metode"] or "Cascading Hybrid"
                }
                for r in rows
            ]
            rata = round(sum(r["score"] for r in rows) / len(rows), 1)
            result.append({
                "target_name": target_name.upper(),
                "rata_rata": rata,
                "rekomendasi": rekomendasi
            })
        conn.close()
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# API statistik ringkas untuk dashboard
# ---------------------------------------------------------
@app.get("/api/stats")
def get_stats():
    try:
        # Total publikasi (jumlah semua publikasi Scholar dari semua dosen)
        q_pub = """
        MATCH (p:ns0__Person)
        RETURN sum(toInteger(p.ns0__hasPublicationScholar)) AS totalPublikasi
        """
        # Total relasi kolaborasi unik
        q_rel = """
        MATCH ()-[:collaborateWith]->()
        RETURN count(*) AS totalRelasi
        """
        # Total departemen
        q_dept = """
        MATCH (p:ns0__Person)
        RETURN count(DISTINCT p.ns0__hasDepartment) AS totalDepartemen
        """
        df_pub = run_query(q_pub)
        df_rel = run_query(q_rel)
        df_dept = run_query(q_dept)
        return {
            "status": "success",
            "data": {
                "totalPublikasi": int(df_pub["totalPublikasi"].iloc[0] or 0),
                "totalRelasi":    int(df_rel["totalRelasi"].iloc[0] or 0),
                "totalDepartemen": int(df_dept["totalDepartemen"].iloc[0] or 0),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


# ==========================================
# FUNGSI GRAF UNTUK STREAMLIT (EX-GRAPH.PY)
# ==========================================
import streamlit.components.v1 as components
from pyvis.network import Network
import streamlit as st

# API/Fungsi untuk memvisualisasikan seluruh graf kolaborasi (digunakan oleh antarmuka Streamlit)
def show_collaboration_graph():
    try:
        query = """
        MATCH (p1:ns0__Person)-[:collaborateWith]->(p2:ns0__Person)
        RETURN p1, p2
        """
        df_result = run_query(query)

        if df_result.empty:
            st.warning("Tidak ada data kolaborasi ditemukan.")
            return

        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        net.force_atlas_2based()

        added_nodes = set()

        for _, row in df_result.iterrows():
            p1 = row["p1"]
            p2 = row["p2"]

            name1 = p1.get("ns0__hasName", "Unknown 1")
            id1 = p1.get("ns0__hasSintaID", "id1")

            name2 = p2.get("ns0__hasName", "Unknown 2")
            id2 = p2.get("ns0__hasSintaID", "id2")

            label1 = f"{name1}\n(SINTA: {id1})"
            label2 = f"{name2}\n(SINTA: {id2})"

            if id1 not in added_nodes:
                net.add_node(id1, label=label1, title=label1, color="#3794ff")
                added_nodes.add(id1)

            if id2 not in added_nodes:
                net.add_node(id2, label=label2, title=label2, color="#3794ff")
                added_nodes.add(id2)

            net.add_edge(id1, id2, title="collaborateWith", color="gray")

        net.save_graph("graph_collaboration.html")
        with open("graph_collaboration.html", "r", encoding="utf-8") as f:
            html = f.read()
            components.html(html, height=650, scrolling=True)

    except Exception as e:
        st.error(f"Gagal menampilkan graf kolaborasi: {e}")
        


def visualize_recommendation_paths(target_name, rekom_names):
    # Bangun query UNION dari semua rekomendasi
    print(rekom_names)
    query_blocks = []
    for nama_rekom in rekom_names:
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
    
    if result.empty:
        fallback_query = f"""
        MATCH (target:ns0__Person {{ns0__hasName: '{target_name}'}})
        WITH target
        MATCH (rekom:ns0__Person)
        WHERE rekom.ns0__hasName IN {rekom_names}
        RETURN collect(DISTINCT target) AS target_nodes, collect(DISTINCT rekom) AS rekom_nodes
        """
        result = run_query(fallback_query)

        if not result.empty:
            target_nodes = result.iloc[0]["target_nodes"]  # list of dicts
            rekom_nodes = result.iloc[0]["rekom_nodes"]    # list of dicts

            nodes = target_nodes + rekom_nodes

            # Buat relasi programatikal: target → setiap rekomendasi
            rels = []
            for rekom_node in rekom_nodes:
                rels.append((target_nodes[0], "recommended", rekom_node))

            # Ganti struktur result menjadi DataFrame dengan kolom 'nodes' dan 'rels'
            result = pd.DataFrame([{
                "nodes": nodes,
                "rels": rels
            }])

    net = Network(height="650px", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()
    added_nodes = set()
    
    for idx, row in result.iterrows():
        nodes = row["nodes"]
        rels = row["rels"]

        # Tambahkan node
        for node in nodes:
            label = node.get("ns0__hasName", "Unknown")
            node_id = label  # Gunakan nama sebagai ID
            # 🔷 Tentukan warna berdasarkan jenis node
            if label == target_name:
                color = "#1f77b4"  # Biru → target utama
            elif label in rekom_names:
                color = "#ff9800"  # Oranye → hasil rekomendasi
            else:
                color = "#4caf50"  # Hijau → penghubung biasa
            
            if node_id not in added_nodes:
                net.add_node(node_id, label=label, title=label, color=color)
                added_nodes.add(node_id)
                
        edge_map = defaultdict(set)
        # Tambahkan edge dari relasi
        for rel in rels:
            if isinstance(rel, tuple) and len(rel) == 3:
                source_node, rel_type, target_node = rel
                source_label = source_node.get("ns0__hasName", "Unknown")
                target_label = target_node.get("ns0__hasName", "Unknown")
                edge_map[(source_label, target_label)].add(rel_type)

        # Tambah relasi tambahan dari rekomendasi
        for rekom_name in rekom_names:
            if rekom_name != target_name and rekom_name in added_nodes:
                edge_map[(target_name, rekom_name)].add("recommended")

        # Tambahkan ke graf
        for (src, tgt), rel_types in edge_map.items():
            # Gabungkan label
            label = ", ".join(rel_types)
            color = "gray" if "recommended" in rel_types else "black"
            net.add_edge(src, tgt, label=label, title=label, color=color, width=2 if "recommended" in rel_types else 1)
              
    
    net.save_graph("graph_recommendation_path.html")
    with open("graph_recommendation_path.html", "r", encoding="utf-8") as f:
        html = f.read()
        st.components.v1.html(html, height=600, scrolling=False)
        
    st.markdown("""
    <ul style="list-style: none; padding-left: 0;">
      <li>
        <span style="display: inline-block; width: 12px; height: 12px; background-color: #1f77b4; margin-right: 10px;"></span>
        <strong>Peneliti Target</strong>
      </li>
      <li>
        <span style="display: inline-block; width: 12px; height: 12px; background-color: #ff7f0e; margin-right: 10px;"></span>
        <strong>Peneliti Rekomendasi</strong>
      </li>
      <li>
        <span style="display: inline-block; width: 12px; height: 12px; background-color: #2ca02c; margin-right: 10px;"></span>
        <strong>Peneliti Penghubung (bukan rekomendasi)</strong>
      </li>
    </ul>

    <ul style="list-style: none; padding-left: 0;">
      <li>
        <span style="display: inline-block; width: 12px; height: 2px; background-color: black; margin-right: 10px;"></span>
        <strong>Relasi collaborateWith (pernah berkolaborasi)</strong>
      </li>
      <li>
        <span style="display: inline-block; width: 12px; height: 2px; background-color: gray; margin-right: 10px;"></span>
        <strong>Relasi recommended (direkomendasikan)</strong>
      </li>
    </ul>
    <hr>
    """, unsafe_allow_html=True)
    