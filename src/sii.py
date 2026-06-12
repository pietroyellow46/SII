import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import community as community_louvain
from sentence_transformers import SentenceTransformer
import weaviate
import logging
from pyvis.network import Network
import re
import os

# Funzione di sicurezza per castare a int
def safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

logging.basicConfig(level=logging.INFO)


# Caricamento dati
cartella_script = os.path.dirname(os.path.abspath(__file__))
percorso_file = os.path.join(cartella_script, "..", "data", "dataset_nba_completo.json")

df = pd.read_json(percorso_file)
colonna_nome = 'Player' if 'Player' in df.columns else ('Nome' if 'Nome' in df.columns else 'player')

# Analisi di rete
logging.info("Calcolo delle community con Louvain")

colonne_stat = ['G', 'PTS', 'TRB', 'AST', 'FG%', 'FG3%', 'FT%', 'eFG%', 'PER', 'WS']
for col in colonne_stat:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

scaler = StandardScaler()
stat_scalate = scaler.fit_transform(df[colonne_stat])

pesi_stat = {
    'G': 0.2, 'PTS': 1.2, 'TRB': 2.0, 'AST': 2.0, 
    'FG%': 1.0, 'FG3%': 1.8, 'FT%': 0.5, 'eFG%': 1.2, 
    'PER': 0.8, 'WS': 1.0     
}

array_pesi_stat = np.array([pesi_stat[col] for col in colonne_stat])
stat_scalate_pesate = stat_scalate * array_pesi_stat

df_ruoli_binari = pd.get_dummies(df['Ruolo_Specifico'])

peso_ruolo = 2.5
matrice_ruoli_pesati = df_ruoli_binari.values * peso_ruolo

matrice_caratteristiche_totale = np.hstack((stat_scalate_pesate, matrice_ruoli_pesati))
matrice_similarita = cosine_similarity(matrice_caratteristiche_totale)

G = nx.Graph()
soglia = 0.85
indici_i, indici_j = np.where((matrice_similarita > soglia) & np.triu(np.ones(matrice_similarita.shape, dtype=bool), k=1))
archi = zip(indici_i, indici_j, matrice_similarita[indici_i, indici_j])
G.add_weighted_edges_from(archi)

partizioni = community_louvain.best_partition(G, weight='weight', random_state=42)
df['Community_ID'] = df.index.map(partizioni)
df['Community_ID'] = df['Community_ID'].fillna(-1).astype(int)

# Esportazione Gephi
logging.info("Preparazione del grafo per l'esportazione su Gephi...")

attributi_nodi = {}
for nodo in G.nodes():
    attributi_nodi[nodo] = {
        'Label': str(df.loc[nodo, colonna_nome]),
        'Community': int(df.loc[nodo, 'Community_ID']),
        'Punti': float(df.loc[nodo, 'PTS'])
    }

nx.set_node_attributes(G, attributi_nodi)

cartella_script = os.path.dirname(os.path.abspath(__file__))
nome_file_grafo = os.path.join(cartella_script, "..", "data", "exports", "graph_nba.graphml")
nx.write_graphml(G, nome_file_grafo)
logging.info(f"Grafo salvato con successo in: {nome_file_grafo}")

# Generazione narrativa e vettorizzazione
def genera_narrativa(r):
    draft_info = f"Scelto al draft dai {r.get('Draft Team', 'Nessuna squadra')} nel {r.get('Draft Year', 'Sconosciuto')}" if pd.notna(r.get('Draft Team')) else "Undrafted"
    
    fg3 = float(r.get('FG3%', 0.0))
    if fg3 >= 36.0:
        frase_triple = f"È un eccellente tiratore perimetrale, convertendo il {fg3}% da tre punti."
    elif fg3 > 0.0 and fg3 < 36.0:
        frase_triple = f"Tira da tre punti con il {fg3}%."
    else:
        frase_triple = "Non è un tiratore da tre punti (0.0%)."

    return re.sub(' +', ' ', (
        f"{r[colonna_nome]} è un giocatore di basket che gioca come {r['Position']}. "
        f"Nato nel {r.get('Born', 'data sconosciuta')}, proviene da {r['College']} e {r['High School']}. "
        f"{draft_info} ed ha debuttato nel {r.get('Debut', 'Sconosciuto')}. "
        f"Fisicamente misura {r['Altezza']} cm per {r['Peso']} kg. "
        f"Statistiche di carriera: ha giocato {r.get('G', 0)} partite, mantenendo medie di {r.get('PTS', 0)} punti, "
        f"{r.get('TRB', 0)} rimbalzi e {r.get('AST', 0)} assist. "
        f"Tira con il {r.get('FG%', 0)}% dal campo. {frase_triple} "
        f"La sua efficienza effettiva (eFG%) è del {r.get('eFG%', 0)}%. Ha un PER di {r.get('PER', 0)}."
    ))

df['Summary'] = df.apply(genera_narrativa, axis=1)

logging.info("Generazione vettori semantici...")
modello = SentenceTransformer('all-MiniLM-L6-v2')
vettori = modello.encode(df['Summary'].tolist(), batch_size=128, show_progress_bar=True)

# Caricamento Weaviate
logging.info("Connessione a Weaviate ed importazione")
client = weaviate.connect_to_local()
nome_collezione = "GiocatoriNBA"

if client.collections.exists(nome_collezione):
    client.collections.delete(nome_collezione)

client.collections.create(nome_collezione)
collezione = client.collections.get(nome_collezione)

with collezione.batch.dynamic() as batch:
    for i, riga in df.iterrows():
        batch.add_object(
            properties={
                "player": str(riga[colonna_nome]), 
                "url": str(riga.get('URL', '')),
                "summary": str(riga['Summary']),
                "community_id": int(riga['Community_ID']),
                
                # METADATI AGGIUNTIVI PER LA RICERCA IBRIDA E I FILTRI
                "ruolo_specifico": str(riga.get('Ruolo_Specifico', 'Altro')),
                "peso": safe_int(riga.get('Peso')),
                "born": safe_int(riga.get('Born')),
                "debut": safe_int(riga.get('Debut')),
                "draft_team": str(riga.get('Draft Team', 'Undrafted')),
                
                # STATISTICHE CHIAVE
                "pts": float(riga.get('PTS', 0.0)),
                "trb": float(riga.get('TRB', 0.0)),
                "ast": float(riga.get('AST', 0.0)),
                "fg3_pct": float(riga.get('FG3%', 0.0)),
                "efg_pct": float(riga.get('eFG%', 0.0))
            },
            vector=vettori[i].tolist()
        )

logging.info(f"Pipeline completata con successo! Inseriti {len(df)} record.")
client.close()

# Louvain
cluster_validi = df[df['Community_ID'] != -1]
numero_cluster = cluster_validi['Community_ID'].nunique()

print(f"\nAnalisi community (Louvain)")
print(f"Totale cluster trovati: {numero_cluster}\n")

conteggio_cluster = cluster_validi['Community_ID'].value_counts().sort_index()

for c_id, conteggio in conteggio_cluster.items():
    giocatori_nel_cluster = df[df['Community_ID'] == c_id][colonna_nome].tolist()
    
    print(f"  COMMUNITY {c_id} | Numero di giocatori: {conteggio}")
    esempi = ", ".join(giocatori_nel_cluster[:10])
    print(f"   Esempi: {esempi}")
    if conteggio > 10:
        print(f"   ...e altri {conteggio - 10} atleti.\n")
