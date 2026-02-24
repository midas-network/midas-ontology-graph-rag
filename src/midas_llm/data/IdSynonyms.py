"""Canonical synonym normalization for infectious disease terminology.

All synonyms map to a single canonical form so that both extracted
and expected values resolve to the same string before comparison.
"""

# Normalize to lowercase canonical form
DISEASE_SYNONYMS: dict[str, str] = {
    # Monkeypox / Mpox
    "mpox": "monkeypox",
    "monkeypox": "monkeypox",
    "monkeypox disease": "monkeypox",
    "monkeypox virus": "monkeypox virus",
    "mpox virus": "monkeypox virus",
    "monkeypox virus (mpxv)": "monkeypox virus",
    "mpxv": "monkeypox virus",

    # COVID-19
    "covid-19": "covid-19",
    "covid": "covid-19",
    "covid 19": "covid-19",
    "coronavirus disease 2019": "covid-19",
    "sars-cov-2": "sars-cov-2",
    "sars cov 2": "sars-cov-2",
    "severe acute respiratory syndrome coronavirus 2": "sars-cov-2",
    "2019-ncov": "sars-cov-2",
    "novel coronavirus": "sars-cov-2",
    "hcov-19": "sars-cov-2",

    # Influenza
    "flu": "influenza",
    "influenza": "influenza",
    "seasonal flu": "influenza",
    "seasonal influenza": "influenza",
    "influenza a": "influenza a",
    "influenza b": "influenza b",
    "h1n1": "influenza a/h1n1",
    "h1n1 influenza": "influenza a/h1n1",
    "swine flu": "influenza a/h1n1",
    "h3n2": "influenza a/h3n2",
    "h5n1": "influenza a/h5n1",
    "avian influenza": "influenza a/h5n1",
    "avian flu": "influenza a/h5n1",
    "bird flu": "influenza a/h5n1",
    "h7n9": "influenza a/h7n9",
    "h5n6": "influenza a/h5n6",
    "pandemic influenza": "pandemic influenza",

    # Ebola
    "ebola": "ebola",
    "ebola virus disease": "ebola",
    "evd": "ebola",
    "ebola hemorrhagic fever": "ebola",
    "ebola virus": "ebola virus",
    "ebov": "ebola virus",

    # Dengue
    "dengue": "dengue",
    "dengue fever": "dengue",
    "dengue hemorrhagic fever": "dengue",
    "dhf": "dengue",
    "dengue shock syndrome": "dengue",
    "denv": "dengue virus",
    "dengue virus": "dengue virus",

    # Malaria
    "malaria": "malaria",
    "plasmodium falciparum": "plasmodium falciparum",
    "p. falciparum": "plasmodium falciparum",
    "plasmodium vivax": "plasmodium vivax",
    "p. vivax": "plasmodium vivax",
    "plasmodium malariae": "plasmodium malariae",
    "plasmodium ovale": "plasmodium ovale",
    "plasmodium knowlesi": "plasmodium knowlesi",

    # Tuberculosis
    "tuberculosis": "tuberculosis",
    "tb": "tuberculosis",
    "mycobacterium tuberculosis": "mycobacterium tuberculosis",
    "m. tuberculosis": "mycobacterium tuberculosis",
    "mtb": "mycobacterium tuberculosis",
    "xdr-tb": "xdr tuberculosis",
    "mdr-tb": "mdr tuberculosis",
    "extensively drug-resistant tuberculosis": "xdr tuberculosis",
    "multidrug-resistant tuberculosis": "mdr tuberculosis",

    # HIV/AIDS
    "hiv": "hiv",
    "hiv/aids": "hiv",
    "hiv-1": "hiv-1",
    "hiv-2": "hiv-2",
    "human immunodeficiency virus": "hiv",
    "aids": "aids",
    "acquired immunodeficiency syndrome": "aids",

    # Measles
    "measles": "measles",
    "rubeola": "measles",
    "measles virus": "measles virus",

    # Cholera
    "cholera": "cholera",
    "vibrio cholerae": "vibrio cholerae",
    "v. cholerae": "vibrio cholerae",

    # Zika
    "zika": "zika",
    "zika fever": "zika",
    "zika virus disease": "zika",
    "zika virus": "zika virus",
    "zikv": "zika virus",

    # Chikungunya
    "chikungunya": "chikungunya",
    "chikungunya fever": "chikungunya",
    "chikungunya virus": "chikungunya virus",
    "chikv": "chikungunya virus",

    # Yellow Fever
    "yellow fever": "yellow fever",
    "yellow fever virus": "yellow fever virus",
    "yfv": "yellow fever virus",

    # West Nile
    "west nile": "west nile",
    "west nile virus": "west nile virus",
    "west nile fever": "west nile",
    "wnv": "west nile virus",

    # MERS
    "mers": "mers",
    "mers-cov": "mers-cov",
    "middle east respiratory syndrome": "mers",
    "middle east respiratory syndrome coronavirus": "mers-cov",

    # SARS (original)
    "sars": "sars",
    "sars-cov": "sars-cov",
    "sars-cov-1": "sars-cov",
    "severe acute respiratory syndrome": "sars",

    # Plague
    "plague": "plague",
    "bubonic plague": "plague",
    "pneumonic plague": "plague",
    "yersinia pestis": "yersinia pestis",
    "y. pestis": "yersinia pestis",

    # Typhoid
    "typhoid": "typhoid",
    "typhoid fever": "typhoid",
    "enteric fever": "typhoid",
    "salmonella typhi": "salmonella typhi",
    "s. typhi": "salmonella typhi",

    # Hepatitis
    "hepatitis a": "hepatitis a",
    "hav": "hepatitis a virus",
    "hepatitis b": "hepatitis b",
    "hbv": "hepatitis b virus",
    "hepatitis c": "hepatitis c",
    "hcv": "hepatitis c virus",
    "hepatitis d": "hepatitis d",
    "hdv": "hepatitis d virus",
    "hepatitis e": "hepatitis e",
    "hev": "hepatitis e virus",

    # Polio
    "polio": "polio",
    "poliomyelitis": "polio",
    "poliovirus": "poliovirus",

    # Rabies
    "rabies": "rabies",
    "rabies virus": "rabies virus",
    "rabv": "rabies virus",

    # Pertussis
    "pertussis": "pertussis",
    "whooping cough": "pertussis",
    "bordetella pertussis": "bordetella pertussis",
    "b. pertussis": "bordetella pertussis",

    # Diphtheria
    "diphtheria": "diphtheria",
    "corynebacterium diphtheriae": "corynebacterium diphtheriae",

    # Tetanus
    "tetanus": "tetanus",
    "clostridium tetani": "clostridium tetani",
    "lockjaw": "tetanus",

    # Meningitis
    "meningitis": "meningitis",
    "bacterial meningitis": "bacterial meningitis",
    "meningococcal disease": "meningococcal disease",
    "neisseria meningitidis": "neisseria meningitidis",
    "n. meningitidis": "neisseria meningitidis",

    # RSV
    "rsv": "rsv",
    "respiratory syncytial virus": "rsv",

    # Rotavirus
    "rotavirus": "rotavirus",
    "rotavirus gastroenteritis": "rotavirus",

    # Norovirus
    "norovirus": "norovirus",
    "norwalk virus": "norovirus",
    "stomach flu": "norovirus",

    # Hand Foot and Mouth
    "hand foot and mouth disease": "hand foot and mouth disease",
    "hfmd": "hand foot and mouth disease",

    # Mumps
    "mumps": "mumps",
    "mumps virus": "mumps virus",

    # Rubella
    "rubella": "rubella",
    "german measles": "rubella",
    "rubella virus": "rubella virus",

    # Varicella / Chickenpox
    "chickenpox": "varicella",
    "chicken pox": "varicella",
    "varicella": "varicella",
    "varicella-zoster virus": "varicella-zoster virus",
    "vzv": "varicella-zoster virus",

    # Shingles
    "shingles": "herpes zoster",
    "herpes zoster": "herpes zoster",

    # Smallpox
    "smallpox": "smallpox",
    "variola": "smallpox",
    "variola major": "smallpox",
    "variola virus": "variola virus",

    # Anthrax
    "anthrax": "anthrax",
    "bacillus anthracis": "bacillus anthracis",
    "b. anthracis": "bacillus anthracis",

    # Lyme Disease
    "lyme disease": "lyme disease",
    "lyme": "lyme disease",
    "borrelia burgdorferi": "borrelia burgdorferi",
    "b. burgdorferi": "borrelia burgdorferi",

    # Leishmaniasis
    "leishmaniasis": "leishmaniasis",
    "kala-azar": "leishmaniasis",
    "leishmania": "leishmania",

    # Chagas
    "chagas disease": "chagas disease",
    "chagas": "chagas disease",
    "trypanosoma cruzi": "trypanosoma cruzi",
    "t. cruzi": "trypanosoma cruzi",
    "american trypanosomiasis": "chagas disease",

    # Schistosomiasis
    "schistosomiasis": "schistosomiasis",
    "bilharzia": "schistosomiasis",
    "snail fever": "schistosomiasis",
    "schistosoma": "schistosoma",

    # Nipah
    "nipah": "nipah",
    "nipah virus": "nipah virus",
    "niv": "nipah virus",

    # Hantavirus
    "hantavirus": "hantavirus",
    "hantavirus pulmonary syndrome": "hantavirus",
    "hps": "hantavirus",

    # Rift Valley Fever
    "rift valley fever": "rift valley fever",
    "rvf": "rift valley fever",
    "rift valley fever virus": "rift valley fever virus",
    "rvfv": "rift valley fever virus",

    # Lassa
    "lassa fever": "lassa fever",
    "lassa": "lassa fever",
    "lassa virus": "lassa virus",
    "lasv": "lassa virus",

    # Marburg
    "marburg": "marburg",
    "marburg virus disease": "marburg",
    "marburg hemorrhagic fever": "marburg",
    "marburg virus": "marburg virus",

    # Crimean-Congo Hemorrhagic Fever
    "crimean-congo hemorrhagic fever": "cchf",
    "cchf": "cchf",
    "cchfv": "cchf virus",

    # Gonorrhea
    "gonorrhea": "gonorrhea",
    "gonorrhoea": "gonorrhea",
    "neisseria gonorrhoeae": "neisseria gonorrhoeae",
    "n. gonorrhoeae": "neisseria gonorrhoeae",

    # Syphilis
    "syphilis": "syphilis",
    "treponema pallidum": "treponema pallidum",
    "t. pallidum": "treponema pallidum",

    # Chlamydia
    "chlamydia": "chlamydia",
    "chlamydia trachomatis": "chlamydia trachomatis",
    "c. trachomatis": "chlamydia trachomatis",

    # C. diff
    "c. diff": "clostridioides difficile",
    "c. difficile": "clostridioides difficile",
    "clostridioides difficile": "clostridioides difficile",
    "clostridium difficile": "clostridioides difficile",
    "cdiff": "clostridioides difficile",

    # MRSA
    "mrsa": "mrsa",
    "methicillin-resistant staphylococcus aureus": "mrsa",

    # E. coli
    "e. coli": "escherichia coli",
    "escherichia coli": "escherichia coli",
    "ehec": "escherichia coli",
    "stec": "escherichia coli",

    # Legionnaires
    "legionnaires disease": "legionnaires disease",
    "legionella": "legionella",
    "legionella pneumophila": "legionella pneumophila",

    # Listeria
    "listeriosis": "listeriosis",
    "listeria": "listeria monocytogenes",
    "listeria monocytogenes": "listeria monocytogenes",

    # Candida auris
    "candida auris": "candida auris",
    "c. auris": "candida auris",

    # Mpox clade variants
    "clade i mpox": "monkeypox clade i",
    "clade ii mpox": "monkeypox clade ii",
    "clade iib": "monkeypox clade iib",
    "clade ia": "monkeypox clade ia",
}