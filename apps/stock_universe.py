"""
STOCK UNIVERSE
Centralized list of stocks to track across all features

Contains:
- S&P 500 stocks (503 stocks)
- Organized by sector
- Easy to expand/customize
"""

# S&P 500 Stocks - Comprehensive List
# Updated: 2025

SP500_STOCKS = [
    # Technology (80+ stocks)
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE', 'CSCO', 'ACN',
    'INTC', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW', 'AMAT', 'MU', 'ADI', 'LRCX',
    'KLAC', 'SNPS', 'CDNS', 'PANW', 'PLTR', 'CRWD', 'FTNT', 'ANSS', 'ADSK', 'ROP',
    'FICO', 'MPWR', 'KEYS', 'ZBRA', 'PTC', 'TYL', 'TER', 'STX', 'WDC', 'NTAP',
    'AKAM', 'JNPR', 'FFIV', 'GEN', 'ENPH', 'SEDG', 'FSLR',

    # Communication Services
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
    'EA', 'TTWO', 'NWSA', 'NWS', 'FOXA', 'FOX', 'PARA', 'WBD', 'OMC', 'IPG',
    'MTCH', 'LYV',

    # Healthcare (60+ stocks)
    'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN',
    'BMY', 'ELV', 'SYK', 'GILD', 'VRTX', 'CI', 'REGN', 'CVS', 'BSX', 'MDT',
    'ISRG', 'ZTS', 'BDX', 'HUM', 'MCK', 'COR', 'EW', 'A', 'HCA', 'IQV',
    'RMD', 'IDXX', 'DXCM', 'CNC', 'MTD', 'WAT', 'STE', 'ALGN', 'HOLX', 'BAX',
    'PODD', 'DGX', 'LH', 'MOH', 'CAH', 'UHS', 'VTRS', 'TECH', 'SOLV', 'DVA',
    'BIO', 'XRAY', 'HSIC', 'INCY', 'BIIB',

    # Financials (70+ stocks)
    'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP', 'BLK',
    'C', 'SPGI', 'MMC', 'CB', 'SCHW', 'PGR', 'BX', 'KKR', 'ICE', 'CME',
    'AON', 'MCO', 'USB', 'TFC', 'PNC', 'AIG', 'MET', 'PRU', 'AFL', 'ALL',
    'TRV', 'AMP', 'FIS', 'MSCI', 'DFS', 'BK', 'COF', 'AJG', 'TROW', 'WTW',
    'STT', 'GPN', 'FITB', 'BRO', 'RF', 'CFG', 'KEY', 'HBAN', 'WRB', 'CINF',
    'L', 'NTRS', 'MTB', 'FDS', 'IVZ', 'BEN', 'GL', 'CBOE', 'NDAQ', 'JKHY',
    'EG', 'RJF', 'AIZ', 'PFG', 'WBS', 'ZION', 'ALLY', 'CMA',

    # Consumer Discretionary (50+ stocks)
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'ABNB',
    'CMG', 'ORLY', 'AZO', 'GM', 'F', 'MAR', 'HLT', 'ROST', 'YUM', 'DHI',
    'LEN', 'PHM', 'DG', 'DLTR', 'DPZ', 'ULTA', 'BBY', 'POOL', 'TPR', 'RL',
    'GRMN', 'DECK', 'NVR', 'LVS', 'WYNN', 'MGM', 'EXPE', 'NCLH', 'RCL', 'CCL',
    'MHK', 'WHR', 'LKQ', 'APTV', 'BWA', 'KMX', 'GPC', 'AAP', 'TSCO',

    # Consumer Staples (30+ stocks)
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'GIS',
    'KMB', 'MNST', 'STZ', 'SYY', 'KHC', 'HSY', 'K', 'CHD', 'CLX', 'CPB',
    'CAG', 'HRL', 'SJM', 'MKC', 'TAP', 'TSN', 'LW', 'BG', 'COKE', 'KDP',
    'EL', 'KR', 'SWK', 'TGT',

    # Energy (20+ stocks)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
    'WMB', 'KMI', 'HAL', 'BKR', 'HES', 'FANG', 'DVN', 'EQT', 'CTRA', 'MRO',
    'APA', 'OVV', 'TRGP',

    # Industrials (70+ stocks)
    'CAT', 'GE', 'RTX', 'HON', 'UPS', 'BA', 'LMT', 'DE', 'UNP', 'ADP',
    'MMM', 'GD', 'NOC', 'ETN', 'ITW', 'PH', 'WM', 'CSX', 'EMR', 'NSC',
    'CMI', 'PCAR', 'PAYX', 'TT', 'RSG', 'JCI', 'CARR', 'OTIS', 'FDX', 'IR',
    'FAST', 'VRSK', 'ROK', 'PWR', 'AME', 'AXON', 'DOV', 'HUBB', 'XYL', 'IEX',
    'LDOS', 'CTAS', 'URI', 'SNA', 'GWW', 'ODFL', 'EXPD', 'JBHT', 'CHRW', 'J',
    'ROL', 'ALLE', 'NDSN', 'PNR', 'DAL', 'UAL', 'LUV', 'AAL', 'ALK', 'JBLU',
    'WAB', 'TXT', 'HWM', 'BLDR', 'SWK', 'MAS', 'AOS', 'GNRC', 'FTV',

    # Materials (25+ stocks)
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC',
    'MLM', 'PPG', 'CTVA', 'BALL', 'AVY', 'ALB', 'AMCR', 'IP', 'PKG', 'EMN',
    'CF', 'MOS', 'CE', 'FMC', 'IFF',

    # Real Estate (30+ stocks)
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
    'EQR', 'VICI', 'VTR', 'INVH', 'ARE', 'MAA', 'SBAC', 'ESS', 'EXR', 'DOC',
    'CPT', 'WY', 'HST', 'CBRE', 'BXP', 'FRT', 'KIM', 'REG', 'UDR', 'AIV',

    # Utilities (25+ stocks)
    'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'D', 'EXC', 'XEL', 'ED',
    'WEC', 'ES', 'AWK', 'DTE', 'PPL', 'AEE', 'FE', 'EIX', 'ETR', 'CMS',
    'CNP', 'NI', 'LNT', 'PNW', 'ATO', 'NRG', 'VST', 'AES', 'PCG',
]

# Remove duplicates and sort
SP500_STOCKS = sorted(list(set(SP500_STOCKS)))

# S&P MidCap 400 - Mid-sized companies ($2B - $10B market cap)
SP400_MIDCAP = [
    # Technology
    'ACIW', 'ALRM', 'AIT', 'ALSN', 'AMBA', 'APPF', 'ARLO', 'AVAV', 'AVT', 'AZPN',
    'BOX', 'CACI', 'CALX', 'CDAY', 'CELH', 'CHKP', 'CXT', 'DOMO', 'DT', 'EEFT',
    'EIGI', 'EVBG', 'EXLS', 'FOXF', 'GDDY', 'GIL', 'GLOB', 'GNRC', 'GTN', 'HLIT',
    'ICHR', 'IGT', 'INSP', 'IOSP', 'IT', 'KD', 'LOGI', 'MGNI', 'MIME', 'MKSI',
    'MTSI', 'NEWR', 'NSIT', 'PEGA', 'PI', 'PLXS', 'PTC', 'QLYS', 'RNG', 'SAIC',
    'SAM', 'SLAB', 'SMTC', 'SPSC', 'SSYS', 'TTMI', 'UCTT', 'VIAV', 'VRRM', 'WEX',

    # Industrials
    'AAT', 'ABM', 'ACM', 'AEO', 'AGX', 'ALK', 'APOG', 'ARCB', 'ARI', 'AROC',
    'ASH', 'ATI', 'AYI', 'BCC', 'BECN', 'BLD', 'BMI', 'BV', 'CACI', 'CBT',
    'CECO', 'CHE', 'CIR', 'CRS', 'CSWI', 'CW', 'DORM', 'DY', 'EEFT', 'EPC',
    'ESE', 'ESNT', 'FLS', 'FN', 'FTRE', 'GFF', 'GVA', 'HI', 'HIW', 'HNI',
    'HQY', 'HUBG', 'HXL', 'ITT', 'JBLU', 'JBT', 'KAMN', 'KAR', 'KEX', 'KFY',
    'KNX', 'KTOS', 'LAD', 'LECO', 'LII', 'LSTR', 'LXU', 'M', 'MATW', 'MLKN',
    'MLM', 'MLNX', 'MOG-A', 'MSA', 'MTZ', 'NAVI', 'NEU', 'NFG', 'NNN', 'NSP',

    # Healthcare
    'ACHC', 'ADMA', 'AMED', 'AMN', 'AMSWA', 'ANET', 'ARVN', 'ATRC', 'AVNS', 'BKD',
    'BLFS', 'BRC', 'BRKR', 'CERT', 'CHRS', 'CNC', 'CNK', 'CROX', 'CRVL', 'CYH',
    'CYTK', 'DNLI', 'DRH', 'DSGX', 'EDUC', 'EFSC', 'EHC', 'EHTH', 'ENSG', 'EPAM',
    'EVH', 'EWBC', 'EXEL', 'FHI', 'FMS', 'FORM', 'GLPI', 'GMS', 'GSHD', 'HAYW',
    'HCA', 'HCSG', 'HQY', 'HRMY', 'HSTM', 'IDA', 'IRWD', 'LMAT', 'LRN', 'LSTR',

    # Financials
    'ABCB', 'AUB', 'BANF', 'BANR', 'BBW', 'BCPC', 'BDC', 'BOKF', 'BPOP', 'BRKL',
    'BXS', 'CADE', 'CASH', 'CATY', 'CBRL', 'CBU', 'CCOI', 'CFR', 'CHCO', 'COOP',
    'CPF', 'CRVL', 'CSFL', 'CTRE', 'DCOM', 'DFS', 'EBSB', 'EFSC', 'EGBN', 'ENS',
    'ESSA', 'EWBC', 'FCFS', 'FFBC', 'FFIN', 'FHB', 'FHN', 'FIBK', 'FISI', 'FMER',
    'FNB', 'FRME', 'FULT', 'GBCI', 'HBAN', 'HBT', 'HMN', 'HOMB', 'HOPE', 'HTLF',
    'IBCP', 'IBOC', 'IBTX', 'INDB', 'LKFN', 'MGRC', 'MPB', 'NBTB', 'NWBI', 'OFG',

    # Consumer Discretionary
    'AAP', 'AAWW', 'ABEO', 'ABG', 'AEO', 'AN', 'ANF', 'ARW', 'ASO', 'AXL',
    'AZZ', 'BBWI', 'BFAM', 'BJ', 'BOOT', 'BREW', 'BURL', 'CAKE', 'CAL', 'CCS',
    'CHUY', 'CKH', 'CRI', 'CROX', 'CVCO', 'CWH', 'DAN', 'DAVA', 'DBI', 'DDS',
    'DENN', 'DIN', 'DKS', 'DNOW', 'DRQ', 'DRVN', 'EAT', 'EBAY', 'ETSY', 'FIVE',
    'FL', 'FOUR', 'FOX', 'GCO', 'GES', 'GPI', 'GRBK', 'GVA', 'HBI', 'HGV',

    # Energy
    'APA', 'AR', 'AROC', 'BKR', 'CIVI', 'CLB', 'CRC', 'CRGY', 'CRK', 'DEN',
    'DNR', 'DRVN', 'ESTE', 'FANG', 'FET', 'GPOR', 'HES', 'HP', 'LEU', 'LPI',
    'MGY', 'MUR', 'NBR', 'NFE', 'NOG', 'OII', 'PTEN', 'RES', 'RIG', 'SDRL',

    # Materials
    'AA', 'AKS', 'AMRC', 'AMX', 'ATKR', 'ATR', 'BCPC', 'BCC', 'BERY', 'CCK',
    'CLF', 'CLW', 'CRS', 'CSWC', 'DNOW', 'FUL', 'GEF', 'GPRE', 'HCC', 'IOSP',
    'KWR', 'LECO', 'NEU', 'OLN', 'POOL', 'RYAM', 'SCCO', 'SEE', 'SLGN', 'SMG',

    # Real Estate
    'AAT', 'ACC', 'AHH', 'AIV', 'AKR', 'BDN', 'BFS', 'BNL', 'BRX', 'BXP',
    'CIO', 'CLI', 'COLD', 'CONE', 'CUZ', 'DEI', 'DRH', 'EGP', 'EPR', 'EQC',
    'ESRT', 'FR', 'GNL', 'GPT', 'GTY', 'HIW', 'HMC', 'HR', 'IRT', 'JBGS',

    # Utilities
    'AEE', 'AES', 'ALE', 'AVNS', 'AVA', 'BKH', 'CPK', 'IDA', 'MDU', 'NJR',
    'NWE', 'OGE', 'ORA', 'PNM', 'POR', 'SJW', 'SR', 'SWX', 'UTL', 'WR',
]

# S&P SmallCap 600 - Small companies ($850M - $3.6B market cap)
SP600_SMALLCAP = [
    # Technology
    'AAON', 'AEIS', 'AGLE', 'AIN', 'ATEN', 'ATTO', 'AVDL', 'AVO', 'BCPC', 'BOOM',
    'BRC', 'CALX', 'CCMP', 'CMCO', 'CNS', 'COHU', 'CONN', 'CTS', 'CUI', 'CVLT',
    'CXW', 'CYBE', 'DLB', 'DGII', 'DIOD', 'DLX', 'DORM', 'EACO', 'EVTC', 'EXTR',
    'FOXF', 'FUL', 'GEO', 'GIII', 'GKOS', 'HCKT', 'HHS', 'HIBB', 'HLIT', 'HMSY',
    'HSII', 'HSTM', 'ICFI', 'IDT', 'IEC', 'INFN', 'IRBT', 'ITIC', 'KAMN', 'KELYA',

    # Industrials
    'AAP', 'ABG', 'ACA', 'ACLS', 'AEL', 'AIN', 'ALG', 'AMED', 'ARI', 'ARL',
    'ATEX', 'AVO', 'AZZ', 'B', 'BBW', 'BC', 'BCO', 'BHE', 'BLBD', 'BMI',
    'BV', 'CAL', 'CATO', 'CBZ', 'CENX', 'CKH', 'CLC', 'CLW', 'CMC', 'CMCO',
    'CNMD', 'CNO', 'CNXN', 'CODE', 'COFS', 'CRS', 'CSL', 'CSV', 'CTO', 'CTS',
    'CVI', 'CW', 'DAN', 'DDD', 'DLX', 'DNB', 'DORM', 'DRQ', 'DXC', 'EARN',

    # Healthcare
    'ADUS', 'AEIS', 'AHCO', 'AHH', 'ALKS', 'AMED', 'AMKR', 'AMSF', 'AMSWA', 'ANIP',
    'ANF', 'ANIK', 'APLE', 'ARLP', 'ASND', 'ASPS', 'AVO', 'AVTR', 'BCOR', 'BDC',
    'BGFV', 'BHE', 'BLBD', 'BRC', 'BRBR', 'BRKL', 'CABO', 'CALM', 'CATO', 'CBRL',
    'CBU', 'CCMP', 'CCS', 'CDNS', 'CENT', 'CENTA', 'CERS', 'CEVA', 'CHRS', 'CIR',
    'CLC', 'CLFD', 'CLGX', 'CLH', 'CLNE', 'CLPR', 'CLVS', 'CNMD', 'CNO', 'CNS',

    # Financials
    'AAIC', 'ABCB', 'ABG', 'ABTX', 'ACBI', 'ACNB', 'AEIS', 'AFBI', 'ALHC', 'ALRS',
    'AMAL', 'AMBP', 'AMSF', 'ANAT', 'ANDE', 'ANF', 'ANIK', 'APAM', 'APOG', 'APPY',
    'AROW', 'ASBI', 'ASPS', 'ASTE', 'ATEK', 'ATEX', 'ATRO', 'ATRI', 'AUB', 'AUBN',
    'AVO', 'AVPT', 'AWIN', 'AWR', 'AX', 'AXL', 'BANC', 'BANF', 'BANR', 'BATRA',
    'BATRK', 'BC', 'BCBP', 'BCPC', 'BDGE', 'BDL', 'BFST', 'BGC', 'BGSF', 'BHB',

    # Consumer Discretionary
    'AAP', 'AAWW', 'ABEO', 'ABG', 'ACA', 'ACEL', 'ACLS', 'AEL', 'AEO', 'AIN',
    'ALCO', 'ALE', 'ALG', 'ANF', 'APOG', 'ARI', 'ARL', 'AROW', 'ARQL', 'ARVN',
    'ASB', 'ASBI', 'ASH', 'ASGN', 'ASTE', 'AVO', 'AVTR', 'AXL', 'AYI', 'AZPN',
    'BANF', 'BANR', 'BBW', 'BC', 'BCC', 'BCOR', 'BDC', 'BDL', 'BECN', 'BGFV',
    'BHE', 'BHLB', 'BJ', 'BJRI', 'BKD', 'BKH', 'BKU', 'BLBD', 'BLFS', 'BLMN',

    # Energy
    'AEP', 'AR', 'AROC', 'BCEI', 'BOOM', 'CDEV', 'CLR', 'CRC', 'CRGY', 'CRK',
    'DEN', 'ESTE', 'GPOR', 'GRNT', 'HP', 'LBRT', 'LPI', 'MTDR', 'NBR', 'NR',
    'OAS', 'PTEN', 'REI', 'REPX', 'RES', 'TALO', 'UEC', 'VNOM', 'WTI', 'XEC',

    # Materials
    'AA', 'AAON', 'ACCO', 'AIN', 'AKS', 'ANDE', 'APOG', 'ARCB', 'ASTE', 'ATI',
    'ATKR', 'AVO', 'AXL', 'BCPC', 'BCC', 'BERY', 'BL', 'BMI', 'BOOM', 'BPMC',

    # Real Estate
    'AAT', 'AAXN', 'ALEX', 'AHH', 'AHR', 'AKR', 'ALEX', 'APLE', 'ARI', 'BANR',
    'BFS', 'BNL', 'BRX', 'BXMT', 'CBL', 'CIO', 'CTO', 'CUZ', 'DEI', 'DHC',

    # Utilities
    'AVA', 'AWR', 'BKH', 'CNSL', 'MSEX', 'NJR', 'NWE', 'OGE', 'OTTR', 'PNM',
]

# Combined S&P 1500 (S&P 500 + MidCap 400 + SmallCap 600)
SP1500_ALL_STOCKS = sorted(list(set(SP500_STOCKS + SP400_MIDCAP + SP600_SMALLCAP)))

# Sector classifications - Expanded to include all S&P 500 stocks
SECTORS = {
    'Technology': [
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE', 'CSCO', 'ACN',
        'INTC', 'IBM', 'QCOM', 'TXN', 'INTU', 'NOW', 'AMAT', 'MU', 'ADI', 'LRCX',
        'KLAC', 'SNPS', 'CDNS', 'PANW', 'PLTR', 'CRWD', 'FTNT', 'ANSS', 'ADSK', 'ROP',
        'FICO', 'MPWR', 'KEYS', 'ZBRA', 'PTC', 'TYL', 'TER', 'STX', 'WDC', 'NTAP',
        'AKAM', 'JNPR', 'FFIV', 'GEN', 'ENPH', 'SEDG', 'FSLR',
    ],
    'Communication': [
        'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
        'EA', 'TTWO', 'NWSA', 'NWS', 'FOXA', 'FOX', 'PARA', 'WBD', 'OMC', 'IPG',
        'MTCH', 'LYV',
    ],
    'Healthcare': [
        'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN',
        'BMY', 'ELV', 'SYK', 'GILD', 'VRTX', 'CI', 'REGN', 'CVS', 'BSX', 'MDT',
        'ISRG', 'ZTS', 'BDX', 'HUM', 'MCK', 'COR', 'EW', 'A', 'HCA', 'IQV',
        'RMD', 'IDXX', 'DXCM', 'CNC', 'MTD', 'WAT', 'STE', 'ALGN', 'HOLX', 'BAX',
        'PODD', 'DGX', 'LH', 'MOH', 'CAH', 'UHS', 'VTRS', 'TECH', 'SOLV', 'DVA',
        'BIO', 'XRAY', 'HSIC', 'INCY', 'BIIB',
    ],
    'Financials': [
        'BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'AXP', 'BLK',
        'C', 'SPGI', 'MMC', 'CB', 'SCHW', 'PGR', 'BX', 'KKR', 'ICE', 'CME',
        'AON', 'MCO', 'USB', 'TFC', 'PNC', 'AIG', 'MET', 'PRU', 'AFL', 'ALL',
        'TRV', 'AMP', 'FIS', 'MSCI', 'DFS', 'BK', 'COF', 'AJG', 'TROW', 'WTW',
        'STT', 'GPN', 'FITB', 'BRO', 'RF', 'CFG', 'KEY', 'HBAN', 'WRB', 'CINF',
        'L', 'NTRS', 'MTB', 'FDS', 'IVZ', 'BEN', 'GL', 'CBOE', 'NDAQ', 'JKHY',
        'EG', 'RJF', 'AIZ', 'PFG', 'WBS', 'ZION', 'ALLY', 'CMA',
    ],
    'Consumer Discretionary': [
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'ABNB',
        'CMG', 'ORLY', 'AZO', 'GM', 'F', 'MAR', 'HLT', 'ROST', 'YUM', 'DHI',
        'LEN', 'PHM', 'DG', 'DLTR', 'DPZ', 'ULTA', 'BBY', 'POOL', 'TPR', 'RL',
        'GRMN', 'DECK', 'NVR', 'LVS', 'WYNN', 'MGM', 'EXPE', 'NCLH', 'RCL', 'CCL',
        'MHK', 'WHR', 'LKQ', 'APTV', 'BWA', 'KMX', 'GPC', 'AAP', 'TSCO',
    ],
    'Consumer Staples': [
        'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'GIS',
        'KMB', 'MNST', 'STZ', 'SYY', 'KHC', 'HSY', 'K', 'CHD', 'CLX', 'CPB',
        'CAG', 'HRL', 'SJM', 'MKC', 'TAP', 'TSN', 'LW', 'BG', 'COKE', 'KDP',
        'EL', 'KR', 'SWK', 'TGT',
    ],
    'Energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
        'WMB', 'KMI', 'HAL', 'BKR', 'HES', 'FANG', 'DVN', 'EQT', 'CTRA', 'MRO',
        'APA', 'OVV', 'TRGP',
    ],
    'Industrials': [
        'CAT', 'GE', 'RTX', 'HON', 'UPS', 'BA', 'LMT', 'DE', 'UNP', 'ADP',
        'MMM', 'GD', 'NOC', 'ETN', 'ITW', 'PH', 'WM', 'CSX', 'EMR', 'NSC',
        'CMI', 'PCAR', 'PAYX', 'TT', 'RSG', 'JCI', 'CARR', 'OTIS', 'FDX', 'IR',
        'FAST', 'VRSK', 'ROK', 'PWR', 'AME', 'AXON', 'DOV', 'HUBB', 'XYL', 'IEX',
        'LDOS', 'CTAS', 'URI', 'SNA', 'GWW', 'ODFL', 'EXPD', 'JBHT', 'CHRW', 'J',
        'ROL', 'ALLE', 'NDSN', 'PNR', 'DAL', 'UAL', 'LUV', 'AAL', 'ALK', 'JBLU',
        'WAB', 'TXT', 'HWM', 'BLDR', 'SWK', 'MAS', 'AOS', 'GNRC', 'FTV',
    ],
    'Materials': [
        'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'NUE', 'VMC',
        'MLM', 'PPG', 'CTVA', 'BALL', 'AVY', 'ALB', 'AMCR', 'IP', 'PKG', 'EMN',
        'CF', 'MOS', 'CE', 'FMC', 'IFF',
    ],
    'Real Estate': [
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
        'EQR', 'VICI', 'VTR', 'INVH', 'ARE', 'MAA', 'SBAC', 'ESS', 'EXR', 'DOC',
        'CPT', 'WY', 'HST', 'CBRE', 'BXP', 'FRT', 'KIM', 'REG', 'UDR', 'AIV',
    ],
    'Utilities': [
        'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'D', 'EXC', 'XEL', 'ED',
        'WEC', 'ES', 'AWK', 'DTE', 'PPL', 'AEE', 'FE', 'EIX', 'ETR', 'CMS',
        'CNP', 'NI', 'LNT', 'PNW', 'ATO', 'NRG', 'VST', 'AES', 'PCG',
    ],
}

# Dividend aristocrats (25+ years of dividend increases)
DIVIDEND_ARISTOCRATS = [
    'MMM', 'ABT', 'ABBV', 'AFL', 'APD', 'ALB', 'MO', 'AMCR', 'ADP', 'BDX',
    'BF-B', 'BRO', 'CAH', 'CAT', 'CVX', 'KO', 'CL', 'ED', 'ECL', 'EMR',
    'ESS', 'EXR', 'XOM', 'FRT', 'GD', 'GPC', 'HRL', 'ITW', 'JNJ', 'KMB',
    'LOW', 'MKC', 'MCD', 'MDT', 'NUE', 'PEP', 'PG', 'PPG', 'O', 'SHW',
    'SPGI', 'SWK', 'SYY', 'TROW', 'TGT', 'WMT', 'WST',
]

# High growth tech stocks
HIGH_GROWTH_TECH = [
    'NVDA', 'AMD', 'AVGO', 'NOW', 'PANW', 'CRWD', 'PLTR', 'DXCM', 'ISRG',
    'ALGN', 'FTNT', 'ENPH', 'SEDG', 'FSLR', 'MPWR',
]

# FAANG+ stocks
FAANG_PLUS = ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL', 'MSFT', 'NVDA', 'TSLA']

# Mega caps (>$500B market cap)
MEGA_CAPS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY', 'V', 'UNH', 'XOM', 'JPM']

# Total count
TOTAL_STOCKS = len(SP500_STOCKS)

def get_stock_universe(category='all'):
    """
    Get stock universe by category

    Args:
        category: 'all', 'sp500', 'tech', 'dividend', 'growth', 'mega', 'faang'

    Returns:
        List of stock symbols
    """
    if category == 'all' or category == 'sp500':
        return SP500_STOCKS
    elif category == 'tech':
        return SECTORS['Technology']
    elif category == 'dividend':
        return DIVIDEND_ARISTOCRATS
    elif category == 'growth':
        return HIGH_GROWTH_TECH
    elif category == 'mega':
        return MEGA_CAPS
    elif category == 'faang':
        return FAANG_PLUS
    else:
        return SP500_STOCKS


def get_stocks_by_sector(sector):
    """Get stocks by sector name"""
    return SECTORS.get(sector, [])


def get_all_sectors():
    """Get list of all sectors"""
    return list(SECTORS.keys())


# ============================================================================
# VISUAL UTILITY FUNCTIONS
# ============================================================================

def create_progress_bar(value, max_value=100, width=20, show_percentage=True):
    """
    Create professional progress bar with gray theme

    Args:
        value: Current value (0-max_value)
        max_value: Maximum value (default 100)
        width: Bar width in characters (default 20)
        show_percentage: Show percentage text (default True)

    Returns:
        Formatted string with colored progress bar
    """
    if max_value == 0:
        percentage = 0
    else:
        percentage = min(100, (value / max_value) * 100)

    filled = int((percentage / 100) * width)
    bar = '█' * filled + '░' * (width - filled)

    # Color based on value
    if percentage >= 70:
        color = 'green'
    elif percentage >= 40:
        color = 'yellow'
    else:
        color = 'red'

    if show_percentage:
        return f"[{color}]{bar}[/{color}] [{color}]{percentage:.0f}%[/{color}]"
    else:
        return f"[{color}]{bar}[/{color}]"


def create_strength_bar(value, width=20):
    """
    Create strength indicator bar (for scores 0-100)

    Args:
        value: Strength score (0-100)
        width: Bar width in characters (default 20)

    Returns:
        Formatted string with colored strength bar and rating
    """
    filled = int((value / 100) * width)
    bar = '█' * filled + '░' * (width - filled)

    # Rating based on strength
    if value >= 85:
        rating = "VERY STRONG"
        color = 'bright_green'
    elif value >= 70:
        rating = "STRONG"
        color = 'green'
    elif value >= 50:
        rating = "MODERATE"
        color = 'yellow'
    elif value >= 30:
        rating = "WEAK"
        color = 'red'
    else:
        rating = "VERY WEAK"
        color = 'bright_red'

    return f"[{color}]{bar}[/{color}] [{color}]{rating}[/{color}]"


def get_status_indicator(status_type):
    """
    Get status indicator symbol with color

    Args:
        status_type: Type of status ('live', 'closed', 'bullish', 'bearish',
                     'warning', 'confirmed', 'neutral', 'up', 'down')

    Returns:
        Formatted status indicator string
    """
    indicators = {
        'live': '[green]●[/green] Live',
        'closed': '[bright_black]○[/bright_black] Closed',
        'bullish': '[green]⬆[/green] Bullish',
        'bearish': '[red]⬇[/red] Bearish',
        'warning': '[yellow]⚠[/yellow] Warning',
        'confirmed': '[green]✓[/green] Confirmed',
        'neutral': '[white]●[/white] Neutral',
        'up': '[green]▲[/green]',
        'down': '[red]▼[/red]',
        'flat': '[white]●[/white]'
    }

    return indicators.get(status_type, f'[white]{status_type}[/white]')


def create_comparison_bar(value, min_val, max_val, width=15):
    """
    Create comparison bar showing where a value falls in a range

    Args:
        value: Current value
        min_val: Minimum value in range
        max_val: Maximum value in range
        width: Bar width in characters (default 15)

    Returns:
        Formatted string showing position in range
    """
    if max_val == min_val:
        position = width // 2
    else:
        normalized = (value - min_val) / (max_val - min_val)
        position = int(normalized * width)
        position = max(0, min(width - 1, position))

    bar = '░' * position + '█' + '░' * (width - position - 1)

    # Color based on position (low=red, mid=yellow, high=green)
    if position < width * 0.33:
        color = 'red'
    elif position < width * 0.67:
        color = 'yellow'
    else:
        color = 'green'

    return f"[{color}]{bar}[/{color}]"


if __name__ == "__main__":
    print(f"Total S&P 500 stocks: {TOTAL_STOCKS}")
    print(f"\nSectors: {len(SECTORS)}")
    for sector, stocks in SECTORS.items():
        print(f"  {sector}: {len(stocks)} stocks")
    print(f"\nDividend Aristocrats: {len(DIVIDEND_ARISTOCRATS)}")
    print(f"High Growth Tech: {len(HIGH_GROWTH_TECH)}")
    print(f"FAANG+: {len(FAANG_PLUS)}")
    print(f"Mega Caps: {len(MEGA_CAPS)}")
