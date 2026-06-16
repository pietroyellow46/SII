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

# Silenziamento
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" 
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    from tqdm import tqdm
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
except ImportError:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()

# Configurazione Groq
load_dotenv()
CHIAVE_GROQ = os.getenv("GROQ_API_KEY")

if not CHIAVE_GROQ:
    raise ValueError("❌ Chiave API non trovata! Assicurati di avere il file .env configurato correttamente.")

client_ai = Groq(api_key=CHIAVE_GROQ)

# LLM parser
def analizza_query_con_llm(domanda_utente, verbose=False):
    if verbose:
        print("🧠 Llama-3 sta analizzando la richiesta...")
    
    prompt_di_sistema = """
    Sei un assistente per un database NBA vettoriale. 
    Estrai i parametri dalla richiesta dell'utente e rispondi SOLO con un oggetto JSON valido, senza markdown (es. niente ```json) e senza spiegazioni.
    
    REGOLA FONDAMENTALE (CONSERVATIVE FILTERING):
    Se l'utente NON menziona in modo esplicito un parametro, NON provare a indovinarlo. Nel dubbio, usa sempre 'null'.
    Fai attenzione alla logica temporale e numerica: 
    - "maggiore di", "dopo il", "almeno" -> usa i campi _min.
    - "minore di", "prima del", "massimo" -> usa i campi _max.
    
    GLOSSARIO DEI RUOLI (ITALIANO -> INGLESE):
    - Playmaker / Play -> "Point Guard"
    - Guardia / Guardia Tiratrice -> "Shooting Guard"
    - Ala Piccola -> "Small Forward"
    - Ala Forte / Ala Grande -> "Power Forward"
    - Centro / Pivot / Lungo -> "Center"

    [VINCOLO OBBLIGATORIO SUL RUOLO]
    Il database contiene SOLO i seguenti 6 valori normalizzati per la proprietà 'ruolo_specifico'. NON inventare mai nomi di ruoli che non sono in questo elenco:
    1. "Point Guard"
    2. "Shooting Guard"
    3. "Small Forward"
    4. "Power Forward"
    5. "Center"
    6. "Big (PF/C)"
    Se l'utente usa sinonimi, mappali correttamente. Se è ambiguo tra due ruoli simili, preferisci categorie ampie come "Big (PF/C)".
    
    Struttura JSON obbligatoria (se un dato non è richiesto, imposta il valore su null):
    {
        "semantic_query": "Testo descrittivo per la ricerca vettoriale (escludi ruoli e numeri stringenti).",
        "ruolo_specifico": "Point Guard|Shooting Guard|Small Forward|Power Forward|Center|Big (PF/C) oppure null",
        "squadra_draft": "Solo la keyword della squadra che lo ha draftato (es. 'Lakers' invece di 'Los Angeles Lakers') oppure null",
        "peso_min": "numero intero (PESO in kg) o null",
        "peso_max": "numero intero (PESO in kg) o null",
        "altezza_min": "numero intero (ALTEZZA in cm) o null",
        "altezza_max": "numero intero (ALTEZZA in cm) o null",
        "anno_nascita_min": "anno in formato intero (es. 1990) o null",
        "anno_nascita_max": "anno in formato intero o null",
        "anno_debutto_min": "anno in formato intero (DEBUTTO/DRAFT) o null",
        "anno_debutto_max": "anno in formato intero o null",
        "experience_min": "numero intero (ANNI DI ESPERIENZA/CARRIERA) o null",
        "experience_max": "numero intero o null",
        "pts_min": "numero decimale (PUNTI SEGNATI) o null",
        "pts_max": "numero decimale o null",
        "trb_min": "numero decimale (RIMBALZI TOTALI) o null",
        "trb_max": "numero decimale o null",
        "ast_min": "numero decimale (ASSIST) o null",
        "ast_max": "numero decimale o null",
        "fg3_pct_min": "numero decimale tra 0 e 1 (PERCENTUALE DA 3 PUNTI, es. 0.38 per 38%) o null",
        "fg3_pct_max": "numero decimale tra 0 e 1 o null",
        "efg_pct_min": "numero decimale tra 0 e 1 (EFFICIENZA AL TIRO / eFG%) o null",
        "efg_pct_max": "numero decimale tra 0 e 1 o null"
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



# Funzione formattazione terminale
def formatta_risultato(oggetto):
    props = oggetto.properties
    
    # Dati Anagrafici
    nome = props.get('player', 'Sconosciuto')
    ruolo = props.get('ruolo_specifico', 'N/D')
    altezza_val = props.get('altezza', 0)
    altezza_str = f"{altezza_val}cm" if altezza_val > 0 else "N/D"
    peso = props.get('peso', 'N/D')
    
    # Dati di Carriera
    nato = props.get('born', 'N/D')
    debutto = props.get('debut', 'N/D')
    exp = props.get('experience', 'N/D')
    draft = props.get('draft_team', 'N/D')
    
    # Statistiche Avanzate
    pts = props.get('pts', 0.0)
    trb = props.get('trb', 0.0)
    ast = props.get('ast', 0.0)
    fg3 = props.get('fg3_pct', 0.0)
    efg = props.get('efg_pct', 0.0)
    
    # Grafo
    community = props.get('community_id', 'N/D')
    
    # Composizione della scheda testuale su più righe
    riga1 = f"👤 {nome:<22} | 🏀 {ruolo:<15} | ⚖️ {altezza_str} {peso}kg"
    riga2 = f"   📅 Nato: {nato} | Deb: {debutto} | Exp: {exp} | Draft: {draft}"
    riga3 = f"   📊 STATS: {pts} PTS | {trb} REB | {ast} AST | 3P%: {fg3}% | eFG%: {efg}%"
    riga4 = f"   🏘️ Community Louvain: {community}"
    
    return f"{riga1}\n{riga2}\n{riga3}\n{riga4}\n"

def esegui_ricerca(domanda_utente, verbose=False, return_params=False):
    try:
        parametri = analizza_query_con_llm(domanda_utente, verbose)
        if verbose:
            print(f"📄 JSON Estratto:\n{json.dumps(parametri, indent=2)}\n")
    except Exception as e:
        print(f"❌ Errore nell'analisi LLM: {e}")
        return (None, None) if return_params else None

    modello_vettori = SentenceTransformer('all-MiniLM-L6-v2')
    testo_semantico = parametri.get("semantic_query", "")
    vettore_query = modello_vettori.encode(testo_semantico, show_progress_bar=False).tolist() if testo_semantico else None

    # filtri weaviate
    filtri_weaviate = []

    def aggiungi_filtro_numerico(chiave_min, chiave_max, colonna_weaviate):
        val_min = parametri.get(chiave_min)
        val_max = parametri.get(chiave_max)
        
        # Lista delle colonne che nel DB sono 0-100 ma l'LLM estrae 0-1
        colonne_percentuali = ["fg3_pct", "efg_pct"]
        
        # Moltiplichi per 100 solo se la colonna è una percentuale
        moltiplicatore = 100 if colonna_weaviate in colonne_percentuali else 1
        
        if val_min is not None:
            val_min_finale = val_min * moltiplicatore
            filtri_weaviate.append(Filter.by_property(colonna_weaviate).greater_or_equal(val_min_finale))
            
        if val_max is not None:
            val_max_finale = val_max * moltiplicatore
            filtri_weaviate.append(Filter.by_property(colonna_weaviate).less_or_equal(val_max_finale))

    # Filtri matematici (Min e Max)
    aggiungi_filtro_numerico("peso_min", "peso_max", "peso")
    aggiungi_filtro_numerico("altezza_min", "altezza_max", "altezza")
    aggiungi_filtro_numerico("anno_nascita_min", "anno_nascita_max", "born")
    aggiungi_filtro_numerico("anno_debutto_min", "anno_debutto_max", "debut")
    aggiungi_filtro_numerico("pts_min", "pts_max", "pts")
    aggiungi_filtro_numerico("trb_min", "trb_max", "trb")
    aggiungi_filtro_numerico("ast_min", "ast_max", "ast")
    aggiungi_filtro_numerico("fg3_pct_min", "fg3_pct_max", "fg3_pct")
    aggiungi_filtro_numerico("efg_pct_min", "efg_pct_max", "efg_pct")
    aggiungi_filtro_numerico("experience_min", "experience_max", "experience")

    # Filtri di Testo (Ruolo e Squadra)
    if parametri.get("ruolo_specifico"):
        filtri_weaviate.append(Filter.by_property("ruolo_specifico").equal(parametri["ruolo_specifico"]))
        
    if parametri.get("squadra_draft"):
        filtri_weaviate.append(Filter.by_property("draft_team").like(f"*{parametri['squadra_draft']}*"))

    # Filtri insieme con la logica AND (&)
    filtro_finale = Filter.all_of(filtri_weaviate) if filtri_weaviate else None

    if verbose:
        print("🔍 Ricerca in Weaviate in corso...\n")
        
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
            
        if return_params:
            return risultati, parametri
        return risultati
        
    except Exception as e:
        print(f"❌ Errore DB Weaviate: {e}")
        return (None, parametri) if return_params else None
    finally:
        client_db.close()


# Ricerca giocatori simili
def cerca_giocatori_fuzzy(testo_nome, limite=5):
    client_db = weaviate.connect_to_local()
    try:
        collezione = client_db.collections.get("GiocatoriNBA")
        modello_vettori = SentenceTransformer('all-MiniLM-L6-v2')
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
        
        res_globale = collezione.query.near_object(
            near_object=uuid_target,
            limit=limite + 1
        )
        
        res_community = collezione.query.near_object(
            near_object=uuid_target,
            limit=limite + 1,
            filters=Filter.by_property("community_id").equal(target_community)
        )
        
        giocatori_uniti = {}
        count_glob = 0
        for obj in res_globale.objects:
            if obj.properties.get("player") != nome_target:
                giocatori_uniti[obj.uuid] = {"oggetto": obj, "etichetta": "🌐 GLOBALE"}
                count_glob += 1
                if count_glob == limite: break
                
        count_comm = 0
        for obj in res_community.objects:
            if obj.properties.get("player") != nome_target:
                if obj.uuid in giocatori_uniti:
                    giocatori_uniti[obj.uuid]["etichetta"] = "⭐ PERFETTO (Globale + Grafo)"
                else:
                    giocatori_uniti[obj.uuid] = {"oggetto": obj, "etichetta": f"🏘️ COMMUNITY {target_community}"}
                count_comm += 1
                if count_comm == limite: break
                
        return list(giocatori_uniti.values())
    finally:
        client_db.close()

if __name__ == "__main__":
    query_utente = "Cerco una guardia tiratrice (Shooting Guard) alta tra i 190cm e i 195cm, che tira con più del 40% da tre punti e che è stata scelta dopo il 2020 scelto dai lakers"
    print(f"\n👤 Utente: '{query_utente}'\n")

    risultati, parametri = esegui_ricerca(query_utente, return_params=True)

    # stampa JSON estratto
    print("📄 Parametri estratti dall'LLM:")
    if parametri:
        print(json.dumps(parametri, indent=4))
    else:
        print("Nessun parametro estratto (errore LLM).")
    print("\n🏀 RISULTATI TROVATI:")

    # se risultati è None o non ha la proprietà 'objects', diventa una lista vuota []
    giocatori = getattr(risultati, 'objects', []) if risultati else []

    # ciclo for parte solo se ci sono giocatori. Se la lista è vuota, viene ignorato.
    for obj in giocatori:
        print("  ->", formatta_risultato(obj))

    # unico controllo per avvisare se la lista è vuota
    if not giocatori:
        print("  Nessun giocatore trovato o errore nella ricerca.")