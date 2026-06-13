import streamlit as st
import weaviate
from weaviate.classes.query import Filter
import json
from sentence_transformers import SentenceTransformer

# "motore" del progetto
from query_LLM import (
    analizza_query_con_llm, 
    cerca_giocatori_fuzzy, 
    cerca_simili_combinato
)

# Configurazione pagina e cache
st.set_page_config(page_title="NBA Search Engine", page_icon="🏀", layout="wide")

# cache di Streamlit per non ricaricare il modello sempre
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

modello_vettori = load_encoder()

# Funzione di stampa della card del giocatore
def draw_player_card(props, etichetta=None):
    with st.container():
        
        # Estrai dati
        nome = props.get("player", "N/D")
        ruolo = props.get("ruolo_specifico", "N/D")
        altezza = props.get('altezza', 'N/D')
        peso = props.get('peso', 'N/D')
        debutto = props.get("debut", "N/D")
        exp = props.get("experience", "N/D")
        
        col1, col2, col3, col4 = st.columns([2.5, 2, 2, 2.5])
        
        with col1:
            st.markdown(f"👤 **{nome}**")
        with col2:
            st.markdown(f"🏀 {ruolo}")
        with col3:
            st.markdown(f"⚖️ {altezza} cm | {peso} kg")
        with col4:
            st.markdown(f"📅 Deb: {debutto} | Exp: {exp}")
            
        st.divider()

# Interfaccia web
st.sidebar.title("🏀 Menu NBA")
st.sidebar.markdown("Scegli la modalità di esplorazione:")
scelta_menu = st.sidebar.radio(
    "Menu di navigazione", 
    ["🔍 Ricerca Intelligente", "👥 Giocatori Simili"], 
    label_visibility="collapsed"
)

if scelta_menu == "🔍 Ricerca Intelligente":
    st.title("Ricerca Intelligente (LLM + Vettori)")
    st.markdown("Chiedi a Llama-3 di trovare giocatori con parametri complessi.")
    
    query = st.chat_input("Es: Trovami un'ala forte dominante a rimbalzo oltre i 100kg...")
    
    if query:
        st.chat_message("user").write(query)
        
        with st.spinner("Llama-3 sta ragionando sui filtri..."):
            try:
                parametri = analizza_query_con_llm(query, verbose=False)
                
                # JSON estratto box a tendina
                with st.expander("📄 Visualizza il JSON estratto dall'LLM"):
                    st.json(parametri)
                    
                # logica di ricerca
                testo_semantico = parametri.get("semantic_query", "")
                vettore_query = modello_vettori.encode(testo_semantico, show_progress_bar=False).tolist() if testo_semantico else None
                
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

                # Query a Weaviate
                client_db = weaviate.connect_to_local()
                try:
                    collezione = client_db.collections.get("GiocatoriNBA")
                    if vettore_query:
                        risultati = collezione.query.near_vector(near_vector=vettore_query, limit=5, filters=filtro_finale)
                    else:
                        risultati = collezione.query.fetch_objects(limit=5, filters=filtro_finale)
                    
                    st.success("Ricerca completata!")
                    
                    if not risultati.objects:
                        st.warning("Nessun giocatore trovato con questi parametri rigidi.")
                    else:
                        for obj in risultati.objects:
                            draw_player_card(obj.properties)
                            
                finally:
                    client_db.close()
            except Exception as e:
                st.error(f"Errore durante l'elaborazione: {e}")

elif scelta_menu == "👥 Giocatori Simili":
    st.title("Raccomandazione Ibrida")
    st.markdown("Unisce la similarità globale alla rete di community di Louvain.")
    
    # Ricerca del giocatore
    nome_target = st.text_input("Inserisci il nome di un giocatore (es. 'LeBron Iames'):")
    
    if nome_target:
        giocatori_trovati = cerca_giocatori_fuzzy(nome_target)
        
        if not giocatori_trovati:
            st.warning("Nessun giocatore trovato.")
        else:
            # Selezione menu a tendina
            opzioni = {f"{g['nome']} ({g['ruolo']})": g for g in giocatori_trovati}
            scelta = st.selectbox("Quale intendevi?", list(opzioni.keys()))
            
            giocatore_scelto = opzioni[scelta]
            
            # Bottone per avviare l'analisi
            if st.button("Trova Simili", type="primary"):
                with st.spinner(f"Analisi Ibrida in corso per {giocatore_scelto['nome']}..."):
                    simili_misti = cerca_simili_combinato(
                        giocatore_scelto['uuid'], 
                        giocatore_scelto['nome'], 
                        giocatore_scelto['community_id']
                    )
                    
                    st.subheader(f"I giocatori più simili a {giocatore_scelto['nome']}")
                    for item in simili_misti:
                        draw_player_card(item["oggetto"].properties, etichetta=item["etichetta"])