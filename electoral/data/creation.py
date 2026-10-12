#Made by Roberto Gaziano
#Last update 07/12/2026 (MM/DD/YYYY)

# Importing libraries
import pandas as pd
import geopandas as gpd

#################
### FUNCTIONS ###
#################

def filtering(data, cols_to_keep, new_names, cols_to_delete = None):

    ''' 
    Elements needed:
    - data: data to subset;
    - cols_to_keep: array of strings the column to keep,
    - new_names: array of new names;
    - cols_to_delete: similar to cols_to_keep, it tells the function which columns to delete.
    '''

    if cols_to_delete == None:
        cols_selected = data[cols_to_keep]

    else:
        cols_selected = data.drop(columns = cols_to_delete)

    cols_selected.columns = new_names

    return cols_selected


#Note [!]: This is the only function in which I've used AI.
def clean_string_column(series):
    """
    Helper function to normalize strings: 
    Converts to uppercase, removes accents (e.g., À -> A), 
    replaces special characters with spaces, 
    and removes extra whitespaces.
    """
    # 1. Convert to uppercase (e.g., "Forlì" -> "FORLÌ")
    s = series.str.upper()

    # 2. Remove accents/diacritics using Unicode normalization
    # NFKD separates characters from their accents, ascii ignores the detached accents
    s = s.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    # 3. Replace special characters (/, -, ., ', etc.) with a single space
    # Note: L'Aquila -> L AQUILA. Cantu' -> CANTU (since the ' becomes a space, stripped later)
    s = s.str.replace(r'[\/\-\.\'\,]', ' ', regex=True)

    # 4. Replace multiple spaces with a single space and strip edges
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()

    return s

## LOAD CONFIG ##
def dataset_creation(config: dict) -> gpd.GeoDataFrame:



    source = config["data"]["raw_elections_dir"]
    crs = config["data"]["crs"]
    output = config["data"]["geopackage_path"]

    ########################
    ### Election Results ###
    ########################

    #We import the first dataset from the folder
    data = pd.read_csv(f"{source}/Camera_Italia_LivComune.csv", sep = ";")


    # Extract the unique turnout totals per municipality first
    Astensione = data[["COMUNE", "CIRC-REG", "COLLPLURI", "COLLUNINOM", "VOTANTITOT", "ELETTORITOT"]].copy()

    #Drop duplicates
    Astensione = Astensione.drop_duplicates(subset=["COMUNE", "CIRC-REG", "COLLPLURI", "COLLUNINOM"])[["COMUNE", "CIRC-REG", "COLLPLURI", "COLLUNINOM", "VOTANTITOT", "ELETTORITOT"]].copy()

    #Clean Regions names
    Astensione["CIRC-REG"] = Astensione["CIRC-REG"].str.replace(r'\s+\d+$', '', regex=True)

    #Groupping municipalities
    Astensione = Astensione.groupby(["CIRC-REG", "COMUNE"])[["VOTANTITOT", "ELETTORITOT"]].sum().reset_index()

    #Drop duplicates
    Astensione = Astensione.drop_duplicates(subset=["COMUNE"])[["COMUNE", "VOTANTITOT", "ELETTORITOT"]].copy()

    #Computing the ratio
    Astensione["ASTENSIONE_RATIO"] = 1 - Astensione["VOTANTITOT"].div(Astensione["ELETTORITOT"])


    data = data.pivot_table(
        index = ["CIRC-REG", "COMUNE"],
        columns = "DESCRLISTA",
        values = "VOTILISTA",
        aggfunc="sum",
        fill_value=0 # Fills missing values with 0
    ).reset_index()

    columns = ['+EUROPA', 'ALLEANZA VERDI E SINISTRA',
        "ALTERNATIVA PER L'ITALIA - NO GREEN PASS",
        'AZIONE - ITALIA VIVA - CALENDA', 'FORZA DEL POPOLO', 'FORZA ITALIA',
        "FRATELLI D'ITALIA CON GIORGIA MELONI", 'FREE',
        'IMPEGNO CIVICO LUIGI DI MAIO - CENTRO DEMOCRATICO',
            "ITALEXIT PER L'ITALIA", 'ITALIA SOVRANA E POPOLARE',
        'LEGA PER SALVINI PREMIER', 'MASTELLA NOI DI CENTRO EUROPEISTI',
        'MOVIMENTO 5 STELLE', 'NOI MODERATI/LUPI - TOTI - BRUGNARO - UDC',
        'PARTITO ANIMALISTA - UCDL - 10 VOLTE MEGLIO',
        'PARTITO COMUNISTA ITALIANO', 'PARTITO DELLA FOLLIA CREATIVA',
        'PARTITO DEMOCRATICO - ITALIA DEMOCRATICA E PROGRESSISTA',
        'SUD CHIAMA NORD', 'SUDTIROLER VOLKSPARTEI (SVP) - PATT',
        'UNIONE POPOLARE CON DE MAGISTRIS', 'VITA']

    new_names = ["+E_RATIO", "AVS_RATIO", "NOVAX_RATIO", "TPOLO_RATIO", "FDP_RATIO", "FI_RATIO", "FDI_RATIO", "FREE_RATIO", "DI_MAIO_RATIO", "ITALEXIT_RATIO",
                "ISP_RATIO", "LEGA_RATIO", "MASTELLA_RATIO", "M5S_RATIO", "UDC_RATIO", "PA_RATIO", "PCI_RATIO", "PFC_RATIO", "PD_RATIO", "SCN_RATIO", "SVP_RATIO", "UP_RATIO", "VITA_RATIO"]

    tot = data[columns].sum(axis=1)

    data[new_names] = data[columns].div(tot, axis=0)

    data = pd.merge(data, Astensione[["COMUNE", "VOTANTITOT", "ELETTORITOT", "ASTENSIONE_RATIO"]], on="COMUNE", how="left")
    data["COMUNE"] = clean_string_column(data["COMUNE"])



    #########################
    ### Demographic Infos ###
    #########################

    Demographic = pd.read_csv(f"{source}/POSAS_2022_it_Comuni.csv",
                            sep = ";",
                            skiprows = 1)

    Union_status = Demographic[Demographic["Età"] == 999]

    Age = Demographic[Demographic["Età"] != 999]
    Age = Age.pivot(index = "Codice comune", columns = "Età", values = "Totale")



    Union_status = filtering(Union_status, cols_to_keep = None, cols_to_delete = ["Età"], 
                            new_names = ["Codice comune", "COMUNE", "CELIBI_MASCHI_F", "CONIUGATI_MASCHI_F", "DIVORZIATI_MASCHI_F", "VEDOVI_MASCHI_F", "UNIONI_CIV_MASCHI_F",
                    "UNIONE_DOPO_DIVORZIO_MASCHI_F","UNIONE_DOPO_MORTE_MASCHI_F", "TOTALE_MASCHI_F", "CELIBI_FEMMINE_F", "CONIUGATI_FEMMINE_F",  "DIVORZIATI_FEMMINE_F",
                    "VEDOVI_FEMMINE_F", "UNIONI_CIV_FEMMINE_F", "UNIONE_DOPO_DIVORZIO_FEMMINE_F", "UNIONE_DOPO_MORTE_FEMMINE_F", "TOTALE_FEMMINE_F", "TOTALE"])

    Demographic_new = pd.merge(Union_status, Age, on = "Codice comune")
    Demographic_new["COMUNE"] = clean_string_column(Demographic_new["COMUNE"])

    dataset = pd.merge(data, Demographic_new, on = "COMUNE")


    ##############
    ### INCOME ###
    ##############

    Income = pd.read_csv(f"{source}/Redditi_e_principali_variabili_IRPEF_su_base_comunale_CSV_2021.csv", 
                    sep=";", encoding='latin1')

    # Keeping the exact Italian column strings from the raw CSV
    income_cols = ["Denominazione Comune", "Sigla Provincia", "Regione", "Numero contribuenti", 
        "Reddito da fabbricati - Frequenza", "Reddito da fabbricati - Ammontare in euro",
        "Reddito da lavoro dipendente e assimilati - Frequenza", "Reddito da lavoro dipendente e assimilati - Ammontare in euro", 
        "Reddito da pensione - Frequenza", "Reddito da pensione - Ammontare in euro", 
        "Reddito da lavoro autonomo (comprensivo dei valori nulli) - Frequenza",
        "Reddito da lavoro autonomo (comprensivo dei valori nulli) - Ammontare in euro",
        "Reddito di spettanza dell'imprenditore in contabilita' ordinaria  (comprensivo dei valori nulli) - Frequenza", 
        "Reddito di spettanza dell'imprenditore in contabilita' ordinaria  (comprensivo dei valori nulli) - Ammontare in euro",
        "Reddito di spettanza dell'imprenditore in contabilita' semplificata (comprensivo dei valori nulli) - Frequenza", 
        "Reddito di spettanza dell'imprenditore in contabilita' semplificata (comprensivo dei valori nulli) - Ammontare in euro",
        "Reddito da partecipazione  (comprensivo dei valori nulli) - Frequenza", 
        "Reddito da partecipazione  (comprensivo dei valori nulli) - Ammontare in euro"
    ]

    income_new_names = [
        "COMUNE", "prov_acr", "REGIONE", "NUMERO_CONTRIBUENTI", "R_FABBRICATI_F", "R_FABBRICATI_E",
        "R_LAVORO_DIPENDENTE_F", "R_LAVORO_DIPENDENTE_E", "R_PENSIONE_F", "R_PENSIONE_E", 
        "R_LAVORO_AUTONOMO_F", "R_LAVORO_AUTONOMO_E", "R_IMPRENDITORE_ORD_F", "R_IMPRENDITORE_ORD_E",
        "R_IMPRENDITORE_SEMPL_F", "R_IMPRENDITORE_SEMPL_E", "R_PARTECIPAZIONE_F", "R_PARTECIPAZIONE_E"
    ]

    Income = filtering(Income, cols_to_keep=income_cols, new_names=income_new_names)
    Income["COMUNE"] = clean_string_column(Income["COMUNE"])

    dataset = pd.merge(dataset, Income, on = "COMUNE")
    #print(dataset.isna().sum()) #Keep in mind that there are missing data for data analysis



    #################
    ### Vehicles ###
    ################

    Vehicles = pd.read_csv(f"{source}/Veicoli - Pubblico registro automobilistico - comuni (IT1,41_993,1.0).csv", dtype={"Comune (ID)": str})

    Vehicles = filtering(Vehicles, cols_to_keep=Vehicles.columns,  
                        new_names=["com_istat_code", "COMUNE", "AUTOVETTURE_F", "AUTOBUS_F", "MOTOCICLI_F"])


    Vehicles["COMUNE"] = clean_string_column(Vehicles["COMUNE"])
    dataset = pd.merge(dataset, Vehicles, on = "COMUNE")


    #############
    ### Firms ###
    #############

    Firms = pd.read_csv(f"{source}/Unità locali e addetti (IT1,183_285,1.0).csv", decimal=",")

    Firms = filtering(Firms, cols_to_keep=["Comune", '2022 Numero di unità locali delle imprese attive',
        '2022 Numero addetti delle unità locali delle imprese attive (valori medi annui)'], 
                    new_names=["COMUNE", "NUMERO_IMPRESE_F", "NUMERO_LAVORATORI_F"]) 


    Firms["COMUNE"] = clean_string_column(Firms["COMUNE"])
    dataset = pd.merge(dataset, Firms, on = "COMUNE")



    ################
    ### Foreigns ###
    ################

    Foreigns = pd.read_csv(f"{source}/Stranieri residenti al 1° gennaio (IT1,29_7,1.0).csv")


    Foreigns = filtering(Foreigns, cols_to_keep=['Comune',        '2022 Popolazione al 1º gennaio Maschi',
        '2022 Popolazione al 1º gennaio Femmine',], 
                        new_names=["COMUNE", "STRANIERI_MASCHI_F", "STRANIERI_FEMMINA_F"])


    Foreigns["COMUNE"] = clean_string_column(Foreigns["COMUNE"])
    dataset = pd.merge(dataset, Foreigns, on = "COMUNE")

    # To replace the max(1, int(os.cpu_count() / 2)) logic

    ###############
    ### Workers ###
    ###############

    Workers = pd.read_csv(f"{source}/SIR - Occupazione nelle imprese (IT1,DF_DIPS_SIR_IND_TERR_DRT_OCC,1.0).csv", decimal=",")

    workers_new_names = [
        "COMUNE", "ADDETTI_ALTA_TECNOLOGIA_RATIO", "DIPENDENTI_ALTA_TECNOLOGIA_RATIO", "INDIPENDENTI_ALTA_TECNOLOGIA_RATIO", 
        "DIPENDENTI_GRANDI_IMPRESE_RATIO", "ADDETTI_MICRO_IMPRESE_RATIO", "DIPENDENTI_MICRO_IMPRESE_RATIO", 
        "INDIPENDENTI_MICRO_IMPRESE_RATIO", "ADDETTI_IMPRESE_INDIVIDUALI_RATIO", "DIPENDENTI_IMPRESE_INDIVIDUALI_RATIO", 
        "INDIPENDENTI_IMPRESE_INDIVIDUALI_RATIO", "ADDETTI_ISTRUZIONE_TERZIARIA_RATIO", "DIPENDENTI_ISTRUZIONE_TERZIARIA_RATIO", 
        "INDIPENDENTI_ISTRUZIONE_TERZIARIA_RATIO", "ADDETTI_RESIDENTI_RATIO", "DIPENDENTI_RESIDENTI_RATIO", 
        "INDIPENDENTI_RESIDENTI_RATIO", "ADDETTI_STRANIERI_RESIDENTI_STRANIERI_RATIO", "ADDETTI_STRANIERI_RATIO", 
        "DIPENDENTI_STRANIERI_RATIO", "INDIPENDENTI_UNICI_ADDETTI_RATIO", "INDIPENDENTI_GIOVANI_RATIO", 
        "ADDETTI_FEMMINE_RATIO", "DIPENDENTI_FEMMINE_RATIO", "INDIPENDENTI_FEMMINE_RATIO", "ADDETTI_ETA_MEDIA", 
        "DIPENDENTI_ETA_MEDIA", "INDIPENDENTI_ETA_MEDIA"
    ]
    # Grabbing exact indices matching the 28 columns extracted in the R script
    Workers = filtering(Workers, cols_to_keep = ['Comune',       '2021 Addetti delle imprese ad alta tecnologia e intense in conoscenza per 100 addetti',       '2021 Dipendenti delle imprese ad alta tecnologia e intense in conoscenza per 100 dipendenti',       '2021 Indipendenti delle imprese ad alta tecnologia e intense in conoscenza per 100 indipendenti',       '2021 Dipendenti delle grandi imprese per 100 dipendenti',       '2021 Addetti delle microimprese per 100 addetti',       '2021 Dipendenti delle microimprese per 100 dipendenti',       '2021 Indipendenti delle microimprese per 100 indipendenti',       '2021 Addetti delle imprese individuali per 100 addetti',       '2021 Dipendenti delle imprese individuali per 100 dipendenti',       '2021 Indipendenti delle imprese individuali per 100 indipendenti',       '2021 Addetti per 100 residenti (età 25-64 anni, con istruzione terziaria), per sesso',       '2021 Dipendenti per 100 residenti (età 25-64 anni, con istruzione terziaria), per sesso',       '2021 Indipendenti con istruzione terziaria per 100 indipendenti (età 25-64 anni), per sesso',       '2021 Addetti per 100 residenti',       '2021 Dipendenti per 100 residenti',       '2021 Indipendenti per 100 residenti',       '2021 Addetti con cittadinanza straniera per 100 residenti stranieri (età 25-64)',       '2021 Addetti con cittadinanza straniera per 100 addetti (età 25-64 anni)',       '2021 Dipendenti con cittadinanza straniera per 100 dipendenti (età 25-64 anni)',       '2021 Indipendenti delle imprese senza dipendenti per 100 indipendenti',       '2021 Indipendenti di età 25-34 anni per 100 indipendenti, per sesso',       '2021 Femmine per 100 addetti',       '2021 Femmine per 100 dipendenti',       '2021 Femmine per 100 indipendenti',       '2021 Età media degli addetti, per sesso',       '2021 Età media dei dipendenti, per sesso',       '2021 Età media degli indipendenti, per sesso'], new_names=workers_new_names)

    Workers["COMUNE"] = clean_string_column(Workers["COMUNE"]) # We see a lot of 0s, these are %s so, there are a lot of rounding errors (we will treat them as NAs)
    dataset = pd.merge(dataset, Workers, on = "COMUNE")


    #FIX:
    dataset = dataset.groupby("com_istat_code", as_index=False).sum()


    ###############
    ### MUSEUMS ###
    ###############

    museums = pd.read_csv(f"{source}/Caratteristiche strutturali - dati a livello comunale (IT1,DF_MUS,1.0).csv",  dtype={"Comune (ID)": str})
    museums = filtering(museums, cols_to_keep=None, cols_to_delete=["Comune"], new_names=["com_istat_code", "NUMERO_MUSEI"])

    print(dataset.shape)

    dataset = pd.merge(dataset, museums, how = "left", on = "com_istat_code")


    #################
    ### HOSPITALS ###
    #################

    hospitals = pd.read_csv(f"{source}/Istituti di cura  - comuni (IT1,43_994,1.0).csv", dtype={"Comune (ID)": str})
    hospitals = filtering(hospitals, cols_to_keep=["Comune (ID)", "2021 Istituti di cura", "2021 Posti letto in degenza ordinaria", "2021 Posti letto in day hospital"], 
                        new_names=["com_istat_code", "OSPEDALI_F", "POSTI_LETTO_F", "DAY_HOSPITAL_F"])

    dataset = pd.merge(dataset, hospitals, how = "left", on = "com_istat_code")


    #############
    ### BANKS ###
    #############

    banks = pd.read_csv(f"{source}/Servizi bancari - dati comunali (IT1,117_1035,1.0).csv", dtype={"Comune (ID)": str})
    banks = filtering(banks, cols_to_keep=None, cols_to_delete=["Comune"], 
                    new_names=["com_istat_code", "SPORTELLI_F", "DEPOSITI_E", "PRESTITI_E"])


    dataset = pd.merge(dataset, banks,  how = "left", on = "com_istat_code")


    #################
    ### LIBRARIES ###
    #################

    libraries = pd.read_csv(f"{source}/Caratteristiche strutturali - dati a livello comunale (IT1,DF_BIBLIO2,1.0).csv", dtype={"Comune (ID)": str})
    libraries = filtering(libraries, cols_to_keep=None, cols_to_delete=["Comune"], 
                        new_names=["com_istat_code", "POSTI_LETTURA_F", "POSTI_PC_F", "POSTI_DISABILI_F", "CON_WIFI_F", "POSTI_LABORATORI_F", "POSTI_BAMBINI_F"])



    dataset = pd.merge(dataset, libraries, how = "left", on = "com_istat_code")



    ##############
    ### TURISM ###
    ##############

    turism = pd.read_csv(f"{source}/Capacità degli esercizi ricettivi (IT1,DF_TURISMO_CAP,1.0).csv")
    turism = filtering(turism, cols_to_keep=["Comune",'2022 Numero di esercizi Totale esercizi ricettivi Alberghi e strutture simili, alloggi per vacanze e altre strutture per brevi soggiorni, aree di campeggio e aree attrezzate per camper e roulotte',
        '2022 Posti letto Totale esercizi ricettivi Alberghi e strutture simili, alloggi per vacanze e altre strutture per brevi soggiorni, aree di campeggio e aree attrezzate per camper e roulotte',
        '2022 Camere Totale esercizi ricettivi Alberghi e strutture simili, alloggi per vacanze e altre strutture per brevi soggiorni, aree di campeggio e aree attrezzate per camper e roulotte',
        '2022 Bagni Totale esercizi ricettivi Alberghi e strutture simili, alloggi per vacanze e altre strutture per brevi soggiorni, aree di campeggio e aree attrezzate per camper e roulotte'], 
                    new_names=["COMUNE", "NUMERO_ALBERGHI_F", "NUMERO_POSTI_LETTO", "NUMERO_CAMERE_F", "NUMERO_BAGNI_F"])

    turism["COMUNE"] = clean_string_column(turism["COMUNE"])

    dataset = pd.merge(dataset, turism, on = "COMUNE")


    #######################
    ### TOWN WEAKNESSES ###
    #######################

    weakness = pd.read_csv(f"{source}/Indice composito di fragilità comunale (IT1,DF_COMP_FRA_IND_MUNICIPAL,1.0).csv", decimal = ",")
    weakness = filtering(weakness, cols_to_keep=["Comune",  "2022 Annuale Indice composito di fragilità comunale - (decile)",       "2022 Annuale Raccolta indifferenziata dei rifiuti urbani per abitante",       "2022 Annuale Consumo del suolo",       "2022 Annuale Indice di accessibilità ai servizi essenziali",       "2022 Annuale Popolazione di età compresa fra 25 e 64 anni con titolo di studio non oltre la licenza di scuola media inferiore o di avviamento professionale",       "2022 Annuale Tasso di occupazione (20-64 anni)",       "2022 Annuale Densità delle unità locali dell’industria e dei servizi - (ventile)",       "2022 Annuale Addetti in unità locali a bassa produttività di settore per l’industria e i servizi - (ventile)"], 
                        new_names=["COMUNE", "INDICE_COMPOSITO", "RACCOLTA_DIFFERENZIATA_RATIO", "CONSUMO_SUOLO", "ACCESSIBILITA_SERVIZI", "BASSA_EDUCAZIONE_RATIO", "OCCUPAZIONE_RATIO", "DENSITA_INDUSTRIA", "LAVORATORI_POCO_PRODUTTIVI_RATIO"])

    weakness["COMUNE"] = clean_string_column(weakness["COMUNE"])

    dataset = pd.merge(dataset, weakness, on = "COMUNE")



    ######################
    ### SCHOLARIZATION ###
    ######################

    Scholarization = pd.read_csv(f"{source}/DCSS_ISTR_LAV_PEN_2_16042026113957005.csv", encoding="utf_8_sig", dtype={"ITTER107": str, "Territorio": str})

    Schol_filtered = Scholarization[(Scholarization['Sesso'] == 'totale') & 
                                    (Scholarization['Cittadinanza'] == 'totale') &
                                    (Scholarization['ITTER107'].astype(str).str.len() == 6)]

    Schol_II = Schol_filtered[Schol_filtered['Grado di istruzione'] == 'diploma di istruzione secondaria di II grado o di qualifica professionale (corso di 3-4 anni) compresi IFTS'][['Territorio', 'Value']]
    Schol_III = Schol_filtered[Schol_filtered['Grado di istruzione'] == 'diploma di tecnico superiore ITS o titolo di studio terziario di primo livello'][['Territorio', 'Value']]
    Schol_IV = Schol_filtered[Schol_filtered['Grado di istruzione'] == 'titolo di studio terziario di secondo livello e dottorato di ricerca'][['Territorio', 'Value']]

    # Merging them together
    Scholarization_F = Schol_II.rename(columns={'Territorio': 'COMUNE', 'Value': 'SECONDO_GRADO_F'})
    Scholarization_F = pd.merge(Scholarization_F, Schol_III.rename(columns={'Value': 'TERZO_GRADO_F'}), left_on='COMUNE', right_on='Territorio').drop(columns=['Territorio'])
    Scholarization_F = pd.merge(Scholarization_F, Schol_IV.rename(columns={'Value': 'QUARTO_GRADO_F'}), left_on='COMUNE', right_on='Territorio').drop(columns=['Territorio'])

    Scholarization_F['COMUNE'] = clean_string_column(Scholarization_F["COMUNE"])

    dataset = pd.merge(dataset, Scholarization_F, on = "COMUNE")





    ###################
    ### AGRICOLTURE ###
    ###################

    Agriculture =  pd.read_csv(f"{source}/Unità agricole e superfici per zona altimentrica (IT1,DF_DCAT_CENSAGRIC2020_UA_CROPS_1,1.0).csv", thousands=",", na_values="..")

    Agriculture = filtering(Agriculture, cols_to_keep=Agriculture.columns, cols_to_delete=None, 
                    new_names=["COMUNE", "SUPERFICIE_UTILIZZATA_ETTARI", "NUMERO_UNITA_AGRICOLE"])

    Agriculture["COMUNE"] = clean_string_column(Agriculture["COMUNE"])

    dataset = pd.merge(dataset, Agriculture, on = "COMUNE", how = "left")



    #########################
    ### TERRAINS NOT USED ###
    #########################
    Not_used = pd.read_csv(f"{source}/Superfici non utilizzate.csv", thousands=",", na_values="..")


    Not_used = filtering(Not_used, cols_to_keep=Not_used.columns, cols_to_delete=None, 
                    new_names=["COMUNE", "SUPERFICIE_NON_UTILIZZATA"])

    Not_used["COMUNE"] = clean_string_column(Not_used["COMUNE"])

    dataset = pd.merge(dataset, Not_used, on = "COMUNE", how = "left")



    #############
    ### FARMS ###
    #############

    Farm =  pd.read_csv(f"{source}/Bestie varie.csv", thousands=",", decimal=".", na_values="..")

    Farm = filtering(Farm, cols_to_keep=Farm.columns, cols_to_delete=None,
                                    new_names=["COMUNE", "BOVINI", "BUFALI", "OVINI", "CAPRINI", "SUINI", "AVICOLI", "ALVEARI"])

    Farm["COMUNE"] = clean_string_column(Farm["COMUNE"])

    dataset = pd.merge(dataset, Farm, on = "COMUNE" , how = "left")



    #################
    ### GEOGRAPHY ###
    #################

    Altim = pd.read_csv(f"{source}/Zona altimetrica.csv", thousands=",", na_values="..")

    Altim = filtering(Altim, cols_to_keep=Altim.columns, cols_to_delete=None, 
                    new_names=["COMUNE", "MONTAGNA_INTERNA", "MONTAGNA_LITORALE", "COLLINA_INTERNA", "COLLINA_LITORALE", "PIANURA", "SUPERFICIE_TOTALE"])

    Altim["COMUNE"] = clean_string_column(Altim["COMUNE"])

    dataset = pd.merge(dataset, Altim, on = "COMUNE" , how = "left")


    ##########################
    ### INCOME AGRICULTURE ###
    ##########################

    Income_agricolture = pd.read_csv(f"{source}/Redditi_unita_agricole.csv", thousands=",", encoding="latin1", sep = ";")


    Income_agricolture = filtering(Income_agricolture, cols_to_keep=["Territorio  ", "Redditi_unità_agricole (stima)", "Sopra 500K"], cols_to_delete=None, 
                    new_names=["COMUNE", "REDDITI_UNITA_AGRICOLE", "REDDITO_AGRICOLTURA_500K"])

    Income_agricolture["COMUNE"] = clean_string_column(Income_agricolture["COMUNE"])

    dataset = pd.merge(dataset, Income_agricolture, on = "COMUNE" , how = "left")


    ###############
    ### HOUSING ###
    ###############

    Houses = pd.read_csv(f"{source}/Famiglie per titolo di godimento - comuni (IT1,DF_DCSS_HUDW_1_COM,1.0).csv", decimal=",", thousands=",")

    Houses = filtering(Houses, cols_to_keep=Houses.columns, 
                    new_names=["COMUNE", "PROPRIETA", "AFFITTO", "ALTRO_TITOLO", "CASE_ABITATE"])

    Houses["COMUNE"] = clean_string_column(Houses["COMUNE"]) 

    dataset = pd.merge(dataset, Houses, on = "COMUNE")



    #################
    ### CLOCHARDS ###
    #################

    Clochard = pd.read_csv(f"{source}/Persone senza tetto.csv", decimal=",", thousands=",", na_values="..")

    Clochard = filtering(Clochard, cols_to_keep=Clochard.columns, 
                    new_names=["COMUNE", "SENZA_TETTO"])

    Clochard["COMUNE"] = clean_string_column(Clochard["COMUNE"])

    dataset = pd.merge(dataset, Clochard, on = "COMUNE", how = "left")


    #################
    ### COMMUTERS ###
    #################

    Commuters = pd.read_csv(f"{source}/matrix_pendoLAVORO_2021.txt", sep = "\t", dtype={"Procom_lav": str})


    Commuters = Commuters[["Procom_lav", "Pendolari"]]

    Commuters = Commuters.groupby(["Procom_lav"])[["Pendolari"]].sum().reset_index()

    Commuters.columns = ["com_istat_code", "PENDOLARI_F"]




    dataset = pd.merge(dataset, Commuters, on = "com_istat_code" , how = "left")



    #######################
    ### ROADS ACCIDENTS ###
    #######################

    Accidents = pd.read_csv(f"{source}/Incidenti, morti e feriti - comuni (IT1,41_983,1.0).csv")

    Accidents = Accidents[["Comune", "2022 Incidenti stradali con lesioni alle persone"]]
    Accidents.columns = ["COMUNE", "INCIDENTI_F"]

    Accidents["COMUNE"] = clean_string_column(Accidents["COMUNE"])
    dataset = pd.merge(dataset, Accidents, on = "COMUNE")


    ################
    ### GEO-INFO ###
    ################

    italy_towns = gpd.read_file(f"{source}/Com01012022_g_WGS84.shp")


    dataset['com_istat_code'] = dataset['com_istat_code'].astype(str).str.zfill(6)


    dataset_poly = italy_towns.merge(
        dataset, 
        left_on="PRO_COM_T", 
        right_on="com_istat_code", 
        how="inner" # Use "left" if you want to keep all towns, even those missing election data
    )


    dataset_poly = dataset_poly.to_crs(crs)

    dataset_poly.columns = dataset_poly.columns.astype(str)


    return dataset_poly
