import os
import sys
import subprocess
import time

try:
    from query_LLM import esegui_ricerca, cerca_giocatori_fuzzy, cerca_simili_combinato, formatta_risultato
except ImportError:
    print(" Impossibile importare 'query_LLM.py'. Assicurati di essere nella stessa cartella.")

def menu_ricerca():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Modalita ricerca intelligente")
    print("Scrivi 'q', 'esci' o 'exit' per tornare al menu principale.")
    
    while True:
        query = input("\nQuery: ")
        
        if query.lower().strip() in ['q', 'esci', 'exit', 'quit']:
            break
        
        if not query.strip():
            continue
            
        esegui_ricerca(query)

def menu_simili():
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(" TROVA GIOCATORI SIMILI")
        print(" Scrivi 'q' per tornare al menu principale\n")

        query_nome = input("\nInserisci il nome del giocatore: ").strip()
        if query_nome.lower() in ['q', 'esci', 'exit', 'quit']:
            break
        if not query_nome:
            continue
        
        print(" Cerco nel database...")
        giocatori_trovati = cerca_giocatori_fuzzy(query_nome)
        
        if not giocatori_trovati:
            print(" Nessun giocatore trovato.")
            input("\nPremi Invio per riprovare...")
            continue
        
        print("\n🏀 Ho trovato questi giocatori. Quale vuoi?")
        for i, g in enumerate(giocatori_trovati, 1):
            print(f"  {i}) {g['nome']} ({g['ruolo']} - Community {g['community_id']})")
        
        scelta = input("\n Seleziona un numero (1-5) oppure premi Invio per cercare un altro nome: ").strip()
        
        if scelta.isdigit() and 1 <= int(scelta) <= len(giocatori_trovati):
            scelta_idx = int(scelta) - 1
            gioc = giocatori_trovati[scelta_idx]
            
            print(f"\n Analisi Ibrida (Vettori + Grafo) per {gioc['nome']}...")
            
            simili_misti = cerca_simili_combinato(gioc['uuid'], gioc['nome'], gioc['community_id'])
            
            print("\n RISULTATI DELL'ANALISI:")
            for item in simili_misti:
                obj = item["oggetto"]
                print(f"  -> {formatta_risultato(obj)}")
                
            input("\nPremi Invio per fare una nuova ricerca...")
        else:
            continue

def reinizializza_database():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Pipeline di inizializzazione")
    
    # Calcola dinamicamente dove si trova sii.py
    cartella_corrente = os.path.dirname(os.path.abspath(__file__))
    script_sii = os.path.join(cartella_corrente, "sii.py")
    
    if not os.path.exists(script_sii):
        print(f" Errore: Il file {script_sii} non è stato trovato.")
        input("\nPremi Invio per tornare al menu...")
        return

    try:
        subprocess.run([sys.executable, script_sii], check=True)
        print("\n Inizializzazione completata con successo! Il Database è aggiornato.")
    except subprocess.CalledProcessError as e:
        print(f"\n Errore durante l'esecuzione: {e}")
    except KeyboardInterrupt:
        print("\n Operazione interrotta dall'utente.")
        
    input("\nPremi Invio per tornare al menu...")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(" NBA Search Engine")
        print("1) Ricerca Intelligente")
        print("2) Trova Giocatori Simili")
        print("3) Re-inizializza Database")
        print("4) Esci")
        
        scelta = input(" Scegli un'opzione: ").strip()
        
        if scelta == '1':
            menu_ricerca()
        elif scelta == '2':
            menu_simili()
        elif scelta == '3':
            conferma = input("\n Questa operazione ricreerà i vettori e le community. Procedere? (s/n): ")
            if conferma.lower() == 's':
                reinizializza_database()
        elif scelta == '4':
            print("\n Arrivederci!\n")
            sys.exit(0)
        else:
            print("\n Scelta non valida.")
            time.sleep(1)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()