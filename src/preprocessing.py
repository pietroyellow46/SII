
import pandas as pd
import numpy as np
import os

# Funzioni di supporto
def mappa_ruolo_specifico(posizione_list):
    if not isinstance(posizione_list, list):
        return 'Altro'
    
    ruoli = {r.lower().strip() for r in posizione_list}
    
    # Ibridi
    if 'point guard' in ruoli and 'shooting guard' in ruoli: return 'Combo Guard'
    if 'shooting guard' in ruoli and 'small forward' in ruoli: return 'Wing (SG/SF)'
    if 'small forward' in ruoli and 'power forward' in ruoli: return 'Combo Forward'
    if 'center' in ruoli and 'power forward' in ruoli: return 'Big (PF/C)'
    
    # Puri
    if 'point guard' in ruoli:     return 'Point Guard'
    if 'shooting guard' in ruoli:   return 'Shooting Guard'
    if 'small forward' in ruoli:    return 'Small Forward'
    if 'power forward' in ruoli:    return 'Power Forward'
    if 'center' in ruoli:           return 'Center'
    
    # Fallback
    if 'guard' in ruoli:            return 'Combo Guard'
    if 'forward' in ruoli:          return 'Combo Forward'
    return 'Altro'

# Caricamento dati
print("Inizio Preprocessing dei dati NBA...")

cartella_script = os.path.dirname(os.path.abspath(__file__))
nome_file_pulito = os.path.join(cartella_script, "..", "data", "dataset_nba_completo.json")
df = pd.read_json(nome_file_pulito)

# Feature engineering ruoli
print("\nCreazione dei ruoli specifici in corso...")
df['Ruolo_Specifico'] = df['Position'].apply(mappa_ruolo_specifico)


# Data cleaning peso
print("\nPulizia e imputazione del Peso in corso...")

# Estrazione del numero dalla stringa (es. "104kg" -> 104.0)
df['Peso'] = df['Peso'].astype(str).str.extract(r'(\d+)').astype(float)

# Imputazione Condizionata: Riempi NaN con la media del loro ruolo specifico
df['Peso'] = df.groupby('Ruolo_Specifico')['Peso'].transform(lambda x: x.fillna(x.mean()))

# Arrotondi a numero intero
df['Peso'] = df['Peso'].round().astype(int)

# Verifica al volo per assicurarci che i nulli siano spariti
nulli_peso = df['Peso'].isnull().sum()
print(f"   -> Valori nulli rimanenti nel Peso: {nulli_peso}")


# Data cleaning percetuali (eFG% e FG3%)
print("\nCorrezione storica delle percentuali di tiro...")

# FG% come numero
df['FG%'] = pd.to_numeric(df['FG%'], errors='coerce').fillna(0.0)

# Correzione FG3% (Tiro da 3 punti)
# Se manca perché non esisteva (pre-1979) o perché il giocatore non ha mai tirato, metti 0.0
df['FG3%'] = pd.to_numeric(df['FG3%'], errors='coerce').fillna(0.0)

# Correzione eFG% (Effective Field Goal Percentage)
# Se l'eFG% manca, significa che non c'erano tiri da 3. 
# In questo caso, l'eFG% equivale matematicamente al normale FG%.
df['eFG%'] = pd.to_numeric(df['eFG%'], errors='coerce').fillna(df['FG%'])

# Verifica
nulli_fg3 = df['FG3%'].isnull().sum()
nulli_efg = df['eFG%'].isnull().sum()
print(f"   -> Valori nulli rimanenti: FG3%={nulli_fg3}, eFG%={nulli_efg}")


# Data cleaning anno di nascita (born)
print("\nCorrezione della feature 'Born'")

colonna_nome = 'Player' if 'Player' in df.columns else 'player'

# Estrazione l'anno di nascita sovrascrivendo direttamente la feature originale 'Born'
df['Born'] = df['Born'].astype(str).str.extract(r'(\d{4})').astype(float)

# Imputazione Euristica: Anno di Debutto - 22 anni
df['Born'] = df['Born'].fillna(df['Debut'] - 22)

# Conversione in numero intero per pulizia (es. 1998 invece di 1998.0)
df['Born'] = df['Born'].astype('Int64')

# Verifica
nulli_nascita = df['Born'].isnull().sum()
print(f"   -> Valori nulli rimanenti in Born: {nulli_nascita}")


# Data cleaning Draft e College
df['Draft Team'] = df['Draft Team'].fillna("Undrafted")
df['Draft Year'] = df['Draft Year'].fillna("Undrafted")
df['College'] = df['College'].fillna("Non frequentato")

# Esportazione nel nuovo file JSON finale
cartella_script = os.path.dirname(os.path.abspath(__file__))
nome_file_pulito = os.path.join(cartella_script, "..", "data", "dataset_nba_completo.json")
df.to_json(nome_file_pulito, orient="records", indent=4, force_ascii=False)
print(f"\nDati puliti salvati con successo")