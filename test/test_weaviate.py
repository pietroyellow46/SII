import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

# Disattiva i log di SentenceTransformer
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def formatta_risultato(oggetto):
    props = oggetto.properties
    nome = props.get('player', 'Sconosciuto')
    ruolo = props.get('ruolo_specifico', 'N/D')
    peso = props.get('peso', 'N/D')
    community = props.get('community_id', 'N/D')
    distanza = f"{oggetto.metadata.distance:.4f}" if oggetto.metadata and oggetto.metadata.distance is not None else "N/A"
    return f"{nome:<22} | Ruolo: {ruolo:<15} | Peso: {peso}kg | Comm: {community:<2} | Distanza: {distanza}"

# Inizializzazione modello (stesso usato per vettorizzare item)
print("Caricamento del modello per vettorizzare le query...")
modello = SentenceTransformer('all-MiniLM-L6-v2') 

# Connessione a Weaviate
print("Connessione a Weaviate in corso...\n")
client = weaviate.connect_to_local()

try:
    collezione = client.collections.get("GiocatoriNBA")
    
    # TEST 1: Ricerca Semantica Pura (Testo -> Vettore -> Weaviate)
    query_1 = "Centro dominante a rimbalzo, forte fisicamente"
    print(f"[TEST 1] RICERCA SEMANTICA (near_vector)")
    print(f"Query: '{query_1}'")
    
    # Trasformi la frase in un vettore matematico
    vettore_query_1 = modello.encode(query_1).tolist()
    
    # usi near_vector passando i numeri
    res_semantica = collezione.query.near_vector(
        near_vector=vettore_query_1,
        limit=4,
        return_metadata=MetadataQuery(distance=True)
    )
    for o in res_semantica.objects:
        print("  ->", formatta_risultato(o))

    # TEST 2: Vettoriale + Filtri Strutturati
    # Cerchi tiratori meno di 100kg
    query_2 = "Giocatore perimetrale con un ottimo tiro da tre punti"
    print(f"\n[TEST 2] VETTORIALE + FILTRO RIGIDO")
    print(f"Query: '{query_2}' | Filtro: Peso > 100kg e Nati dopo il 1990")
    
    vettore_query_2 = modello.encode(query_2).tolist()
    
    res_filtrata = collezione.query.near_vector(
        near_vector=vettore_query_2,
        limit=4,
        filters=(
            Filter.by_property("peso").greater_than(100) &
            Filter.by_property("born").greater_than(1990)
        ),
        return_metadata=MetadataQuery(distance=True)
    )
    for o in res_filtrata.objects:
        print("  ->", formatta_risultato(o))

    # TEST 3: Giocatori simili (near_object) all'interno del Grafo
    pivot = "LeBron James"
    target_community = 10 # community data da Louvian
    print(f"\n[TEST 3] GRAFO + VETTORI: Simili a '{pivot}' (solo nella Community {target_community})")
    
    # Trovi prima l'oggetto LeBron
    res_pivot = collezione.query.fetch_objects(filters=Filter.by_property("player").equal(pivot), limit=1)
    
    if res_pivot.objects:
        obj_pivot = res_pivot.objects[0]
        vicini = collezione.query.near_object(
            near_object=obj_pivot.uuid,
            limit=4,
            filters=Filter.by_property("community_id").equal(target_community),
            return_metadata=MetadataQuery(distance=True)
        )
        for o in vicini.objects:
            if o.properties.get('player') != pivot:
                print("  ->", formatta_risultato(o))

    # TEST 4: Database Relazionale Classico
    print(f"\n[TEST 4] RICERCA SQL-LIKE (Senza vettori, solo matematica)")
    print("Obiettivo: Point Guard, Età > 2000, eFG% > 50%")
    
    res_sql = collezione.query.fetch_objects(
        filters=(
            Filter.by_property("ruolo_specifico").equal("Point Guard") &
            Filter.by_property("born").greater_or_equal(2000) &
            Filter.by_property("efg_pct").greater_than(50.0)
        ),
        limit=4
    )
    for o in res_sql.objects:
        pts = o.properties.get('pts', 0)
        efg = o.properties.get('efg_pct', 0)
        print(f"  -> {o.properties.get('player'):<22} | Born: {o.properties.get('born')} | PTS: {pts} | eFG%: {efg}%")

finally:
    client.close()
    print("\nConnessione a Weaviate chiusa correttamente")