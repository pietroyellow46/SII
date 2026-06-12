import os
import warnings
import transformers
import logging
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
from dotenv import load_dotenv

# Spegne i warning di Python
warnings.filterwarnings("ignore")

# Spegne le variabili d'ambiente
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" 
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"

# Spegne barra animata (tqdm)
try:
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except ImportError:
    pass

# Spegne log rimanenti
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# silenziatore ufficiale
transformers.logging.set_verbosity_error()


# Configurazione GROQ
load_dotenv()
CHIAVE_GROQ = os.getenv("GROQ_API_KEY")

if not CHIAVE_GROQ:
    raise ValueError(" Chiave API non trovata! Assicurati di avere il file .env configurato correttamente.")

client_ai = Groq(api_key=CHIAVE_GROQ)


# LLM PARSER
def analizza_query_con_llm(domanda_utente, verbose=False):
    if verbose:
        print(" Llama-3 sta analizzando la richiesta...")
    
    prompt_di_sistema = """
    Sei un assistente per un database NBA vettoriale. 
    Estrai i parametri dalla richiesta dell'utente e rispondi SOLO con un oggetto JSON, senza markdown e senza spiegazioni.
    
    REGOLA FONDAMENTALE (CONSERVATIVE FILTERING):
    Se l'utente NON menziona in modo esplicito un parametro (peso, anno di debutto o un ruolo esatto), NON provare a indovinarlo o a dedurlo. Nel dubbio, usa sempre 'null'.
    
    GLOSSARIO DEI RUOLI (ITALIANO -> INGLESE):
    Se l'utente specifica un ruolo in italiano, DEVI tradurlo usando ESATTAMENTE questa mappatura:
    - Playmaker / Play -> "Point Guard"
    - Guardia / Guardia Tiratrice -> "Shooting Guard"
    - Ala Piccola -> "Small Forward"
    - Ala Forte / Ala Grande -> "Power Forward"
    - Centro / Pivot / Lungo -> "Center"
    
    Struttura JSON obbligatoria:
    {
        "semantic_query": "Testo descrittivo (es. dominante a rimbalzo). Escludi da qui il nome del ruolo se lo hai già mappato sotto.",
        "ruolo_specifico": "Point Guard|Shooting Guard|Small Forward|Power Forward|Center|Combo Guard|Wing (SG/SF)|Combo Forward|Big (PF/C) oppure null se incerto",
        "peso_minimo": numero intero o null se non specificato,
        "anno_debutto_minimo": numero intero o null se non specificato
    }
    """
    
    risposta = client_ai.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt_di_sistema},
            {"role": "user", "content": domanda_utente}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    contenuto = risposta.choices[0].message.content
    return json.loads(contenuto)


# MOTORE DI RICERCA IBRIDO
def formatta_risultato(oggetto):
    props = oggetto.properties
    
    nome = props.get('player', 'Sconosciuto')
    ruolo = props.get('ruolo_specifico', 'N/D')
    peso = props.get('peso', 'N/D')
    altezza = props.get('altezza', 'N/D')
    debutto = props.get('debut', 'N/D')
    exp = props.get('experience', 'N/D')
    
    return f"{nome:<22} | Ruolo: {ruolo:<15} | Peso: {peso}kg | Altezza: {altezza} | Debutto: {debutto} | Exp: {exp}"

def esegui_ricerca(domanda_utente, verbose=False):
    try:
        parametri = analizza_query_con_llm(domanda_utente, verbose)
        if verbose:
            print(f" JSON Estratto:\n{json.dumps(parametri, indent=2)}\n")
    except Exception as e:
        print(f" Errore nell'analisi LLM: {e}")
        return

    modello_vettori = SentenceTransformer('all-MiniLM-L6-v2')
    testo_semantico = parametri.get("semantic_query", "")
    vettore_query = modello_vettori.encode(testo_semantico, show_progress_bar=False).tolist() if testo_semantico else None

    # Costruzione filtri Weaviate
    filtri_weaviate = []
    if parametri.get("peso_minimo"):
        filtri_weaviate.append(Filter.by_property("peso").greater_or_equal(parametri["peso_minimo"]))
    if parametri.get("anno_debutto_minimo"):
        filtri_weaviate.append(Filter.by_property("debut").greater_or_equal(parametri["anno_debutto_minimo"]))
    if parametri.get("ruolo_specifico"):
        filtri_weaviate.append(Filter.by_property("ruolo_specifico").equal(parametri["ruolo_specifico"]))

    filtro_finale = None
    if filtri_weaviate:
        filtro_finale = filtri_weaviate[0]
        for f in filtri_weaviate[1:]:
            filtro_finale = filtro_finale & f

    if verbose:
        print(" Ricerca in Weaviate in corso...\n")
        
    client_db = weaviate.connect_to_local()
    
    try:
        collezione = client_db.collections.get("GiocatoriNBA")
        
        if vettore_query:
            risultati = collezione.query.near_vector(
                near_vector=vettore_query,
                limit=5,
                filters=filtro_finale
            )
        else:
            risultati = collezione.query.fetch_objects(
                limit=5,
                filters=filtro_finale
            )
            
        print("\n🏀 RISULTATI TROVATI:")
        if not risultati.objects:
            print("  Nessun giocatore trovato con questi parametri rigidi.")
        else:
            for o in risultati.objects:
                print("  ->", formatta_risultato(o))
                
    finally:
        client_db.close()


# RICERCA GIOCATORI SIMILI
def cerca_giocatori_fuzzy(testo_nome, limite=5):
    client_db = weaviate.connect_to_local()
    try:
        collezione = client_db.collections.get("GiocatoriNBA")

        # Inizializzia il modello
        modello_vettori = SentenceTransformer('all-MiniLM-L6-v2')

        # Vettorizzia
        vettore = modello_vettori.encode(testo_nome, show_progress_bar=False).tolist()
        
        risultati = collezione.query.near_vector(
            near_vector=vettore,
            limit=limite
        )
        
        giocatori = []
        for obj in risultati.objects:
            giocatori.append({
                "uuid": obj.uuid,
                "nome": obj.properties.get("player", "Sconosciuto"),
                "ruolo": obj.properties.get("ruolo_specifico", "N/D"),
                "community_id": obj.properties.get("community_id")
            })
        return giocatori
    finally:
        client_db.close()

def cerca_simili_combinato(uuid_target, nome_target, target_community, limite=5):
    client_db = weaviate.connect_to_local()
    try:
        collezione = client_db.collections.get("GiocatoriNBA")
        
        # Ricerca GLOBALE (Pura somiglianza vettoriale)
        res_globale = collezione.query.near_object(
            near_object=uuid_target,
            limit=limite + 1
        )
        
        # Ricerca COMMUNITY (Vincolata al grafo di Louvain)
        res_community = collezione.query.near_object(
            near_object=uuid_target,
            limit=limite + 1,
            filters=Filter.by_property("community_id").equal(target_community)
        )
        
        # Dizionario per unire i risultati e trovare le sovrapposizioni (Match Perfetti)
        giocatori_uniti = {}
        
        # Inserisci risultati globali
        count_glob = 0
        for obj in res_globale.objects:
            if obj.properties.get("player") != nome_target:
                giocatori_uniti[obj.uuid] = {"oggetto": obj, "etichetta": "🌐 GLOBALE"}
                count_glob += 1
                if count_glob == limite: break
                
        # Inserisci risultati della community
        count_comm = 0
        for obj in res_community.objects:
            if obj.properties.get("player") != nome_target:
                if obj.uuid in giocatori_uniti:
                    giocatori_uniti[obj.uuid]["etichetta"] = " PERFETTO (Globale + Grafo)"
                else:
                    giocatori_uniti[obj.uuid] = {"oggetto": obj, "etichetta": f"🏘️ COMMUNITY {target_community}"}
                count_comm += 1
                if count_comm == limite: break
                
        return list(giocatori_uniti.values())
    finally:
        client_db.close()

# TESTIAMO IL BOT
if __name__ == "__main__":
    query_utente = "Trovami un'ala forte dominante a rimbalzo che pesa più di 100kg e ha debuttato dopo il 2020"
    print(f" Utente: '{query_utente}'\n" + "-"*50)
    esegui_ricerca(query_utente, verbose=True)