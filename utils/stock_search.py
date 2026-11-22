"""
utils/stock_search.py — Indian Stock Market Universe & Smart Symbol Resolver
=============================================================================
Provides comprehensive search and symbol resolution for Indian stocks (NSE & BSE).

Key Capabilities:
1. Search by Company Name (e.g. "Reliance Industries", "Tata Motors", "State Bank")
2. Search by NSE Symbol (e.g. "RELIANCE", "TCS", "INFY", "HDFCBANK")
3. Search by BSE Security Code (e.g. "500325", "532540", "500209", "500180")
4. Exchange Selection: NSE (default), BSE, or AUTO
5. Autocomplete / Search Suggestions matching query across names, symbols & codes
6. Smart Resolution to valid Yahoo Finance tickers (.NS for NSE, .BO for BSE)
7. Fallback to Yahoo Finance Search API for unlisted / newly listed securities
"""

import re
from typing import List, Dict, Optional, Tuple
import yfinance as yf


# =======================================================================
# COMPREHENSIVE INDIAN STOCK UNIVERSE (NSE Symbol, BSE Code, Name, Sector)
# =======================================================================
INDIAN_STOCKS_DATA = [
    # Top Large Cap / Nifty 50
    {"symbol": "RELIANCE", "bse_code": "500325", "name": "Reliance Industries Limited", "sector": "Energy / Conglomerate"},
    {"symbol": "TCS", "bse_code": "532540", "name": "Tata Consultancy Services Limited", "sector": "Information Technology"},
    {"symbol": "HDFCBANK", "bse_code": "500180", "name": "HDFC Bank Limited", "sector": "Banking & Financial Services"},
    {"symbol": "ICICIBANK", "bse_code": "532174", "name": "ICICI Bank Limited", "sector": "Banking & Financial Services"},
    {"symbol": "BHARTIARTL", "bse_code": "532454", "name": "Bharti Airtel Limited", "sector": "Telecommunications"},
    {"symbol": "INFY", "bse_code": "500209", "name": "Infosys Limited", "sector": "Information Technology"},
    {"symbol": "SBIN", "bse_code": "500112", "name": "State Bank of India", "sector": "Public Sector Banking"},
    {"symbol": "ITC", "bse_code": "500875", "name": "ITC Limited", "sector": "FMCG / Tobacco"},
    {"symbol": "HINDUNILVR", "bse_code": "500696", "name": "Hindustan Unilever Limited", "sector": "FMCG"},
    {"symbol": "LT", "bse_code": "500510", "name": "Larsen & Toubro Limited", "sector": "Infrastructure & Engineering"},
    {"symbol": "BAJFINANCE", "bse_code": "500034", "name": "Bajaj Finance Limited", "sector": "Non-Banking Financial"},
    {"symbol": "KOTAKBANK", "bse_code": "500247", "name": "Kotak Mahindra Bank Limited", "sector": "Banking & Financial Services"},
    {"symbol": "HCLTECH", "bse_code": "532281", "name": "HCL Technologies Limited", "sector": "Information Technology"},
    {"symbol": "M&M", "bse_code": "500520", "name": "Mahindra & Mahindra Limited", "sector": "Automobile"},
    {"symbol": "TATAMOTORS", "bse_code": "500570", "name": "Tata Motors Limited", "sector": "Automobile"},
    {"symbol": "AXISBANK", "bse_code": "532215", "name": "Axis Bank Limited", "sector": "Banking & Financial Services"},
    {"symbol": "SUNPHARMA", "bse_code": "524715", "name": "Sun Pharmaceutical Industries Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "TITAN", "bse_code": "500114", "name": "Titan Company Limited", "sector": "Consumer Discretionary"},
    {"symbol": "MARUTI", "bse_code": "532500", "name": "Maruti Suzuki India Limited", "sector": "Automobile"},
    {"symbol": "ULTRACEMCO", "bse_code": "532538", "name": "UltraTech Cement Limited", "sector": "Materials / Cement"},
    {"symbol": "NTPC", "bse_code": "532555", "name": "NTPC Limited", "sector": "Power Generation"},
    {"symbol": "POWERGRID", "bse_code": "532898", "name": "Power Grid Corporation of India Limited", "sector": "Power Transmission"},
    {"symbol": "TATASTEEL", "bse_code": "500470", "name": "Tata Steel Limited", "sector": "Metals & Mining"},
    {"symbol": "ONGC", "bse_code": "500312", "name": "Oil & Natural Gas Corporation Limited", "sector": "Oil & Gas"},
    {"symbol": "ADANIENT", "bse_code": "512599", "name": "Adani Enterprises Limited", "sector": "Conglomerate"},
    {"symbol": "ADANIPORTS", "bse_code": "532921", "name": "Adani Ports and Special Economic Zone Limited", "sector": "Ports & Logistics"},
    {"symbol": "COALINDIA", "bse_code": "533278", "name": "Coal India Limited", "sector": "Metals & Mining"},
    {"symbol": "BAJAJFINSV", "bse_code": "532978", "name": "Bajaj Finserv Limited", "sector": "Financial Services"},
    {"symbol": "WIPRO", "bse_code": "507685", "name": "Wipro Limited", "sector": "Information Technology"},
    {"symbol": "ASIANPAINT", "bse_code": "500820", "name": "Asian Paints Limited", "sector": "Paints & Coatings"},
    {"symbol": "JSWSTEEL", "bse_code": "500228", "name": "JSW Steel Limited", "sector": "Metals & Mining"},
    {"symbol": "GRASIM", "bse_code": "500300", "name": "Grasim Industries Limited", "sector": "Materials / Cement"},
    {"symbol": "NESTLEIND", "bse_code": "500790", "name": "Nestle India Limited", "sector": "FMCG"},
    {"symbol": "TECHM", "bse_code": "532755", "name": "Tech Mahindra Limited", "sector": "Information Technology"},
    {"symbol": "HINDALCO", "bse_code": "500440", "name": "Hindalco Industries Limited", "sector": "Metals & Mining"},
    {"symbol": "BRITANNIA", "bse_code": "500825", "name": "Britannia Industries Limited", "sector": "FMCG"},
    {"symbol": "CIPLA", "bse_code": "500087", "name": "Cipla Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "DRREDDY", "bse_code": "500124", "name": "Dr. Reddy's Laboratories Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "EICHERMOT", "bse_code": "505200", "name": "Eicher Motors Limited", "sector": "Automobile"},
    {"symbol": "TATACONSUM", "bse_code": "500800", "name": "Tata Consumer Products Limited", "sector": "FMCG"},
    {"symbol": "BAJAJ-AUTO", "bse_code": "532977", "name": "Bajaj Auto Limited", "sector": "Automobile"},
    {"symbol": "DIVISLAB", "bse_code": "532488", "name": "Divi's Laboratories Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "APOLLOHOSP", "bse_code": "508869", "name": "Apollo Hospitals Enterprise Limited", "sector": "Healthcare Services"},
    {"symbol": "BPCL", "bse_code": "500547", "name": "Bharat Petroleum Corporation Limited", "sector": "Oil & Gas Refineries"},
    {"symbol": "HEROMOTOCO", "bse_code": "500182", "name": "Hero MotoCorp Limited", "sector": "Automobile"},
    {"symbol": "SHRIRAMFIN", "bse_code": "511218", "name": "Shriram Finance Limited", "sector": "Non-Banking Financial"},
    {"symbol": "INDUSINDBK", "bse_code": "532187", "name": "IndusInd Bank Limited", "sector": "Banking & Financial Services"},
    {"symbol": "SBILIFE", "bse_code": "540719", "name": "SBI Life Insurance Company Limited", "sector": "Insurance"},
    {"symbol": "HDFCLIFE", "bse_code": "540777", "name": "HDFC Life Insurance Company Limited", "sector": "Insurance"},
    {"symbol": "BEL", "bse_code": "500049", "name": "Bharat Electronics Limited", "sector": "Aerospace & Defense"},
    {"symbol": "HAL", "bse_code": "541154", "name": "Hindustan Aeronautics Limited", "sector": "Aerospace & Defense"},
    {"symbol": "TRENT", "bse_code": "500251", "name": "Trent Limited", "sector": "Retail & Fashion"},
    {"symbol": "ZOMATO", "bse_code": "543320", "name": "Zomato Limited", "sector": "Internet / Food Delivery"},
    {"symbol": "JIOFIN", "bse_code": "543940", "name": "Jio Financial Services Limited", "sector": "Financial Services"},

    # High-Interest Next 50 & Midcap Leaders
    {"symbol": "VBL", "bse_code": "540180", "name": "Varun Beverages Limited", "sector": "Beverages"},
    {"symbol": "VEDL", "bse_code": "500295", "name": "Vedanta Limited", "sector": "Metals & Mining"},
    {"symbol": "DLF", "bse_code": "532868", "name": "DLF Limited", "sector": "Real Estate"},
    {"symbol": "SIEMENS", "bse_code": "500550", "name": "Siemens Limited", "sector": "Capital Goods"},
    {"symbol": "ABB", "bse_code": "500002", "name": "ABB India Limited", "sector": "Capital Goods"},
    {"symbol": "PFC", "bse_code": "532810", "name": "Power Finance Corporation Limited", "sector": "Financial Services"},
    {"symbol": "RECLTD", "bse_code": "532955", "name": "REC Limited", "sector": "Financial Services"},
    {"symbol": "IOC", "bse_code": "530965", "name": "Indian Oil Corporation Limited", "sector": "Oil & Gas"},
    {"symbol": "GAIL", "bse_code": "532155", "name": "GAIL (India) Limited", "sector": "Gas Transmission"},
    {"symbol": "TATAPOWER", "bse_code": "500400", "name": "Tata Power Company Limited", "sector": "Power Generation"},
    {"symbol": "ADANIGREEN", "bse_code": "541450", "name": "Adani Green Energy Limited", "sector": "Renewable Energy"},
    {"symbol": "ADANIPOWER", "bse_code": "533096", "name": "Adani Power Limited", "sector": "Power Generation"},
    {"symbol": "CHOLAFIN", "bse_code": "511243", "name": "Cholamandalam Investment and Finance Company", "sector": "Non-Banking Financial"},
    {"symbol": "BANKBARODA", "bse_code": "532134", "name": "Bank of Baroda", "sector": "Public Sector Banking"},
    {"symbol": "PNB", "bse_code": "532461", "name": "Punjab National Bank", "sector": "Public Sector Banking"},
    {"symbol": "CANBK", "bse_code": "532483", "name": "Canara Bank", "sector": "Public Sector Banking"},
    {"symbol": "UNIONBANK", "bse_code": "532477", "name": "Union Bank of India", "sector": "Public Sector Banking"},
    {"symbol": "IDFCFIRSTB", "bse_code": "539437", "name": "IDFC FIRST Bank Limited", "sector": "Banking & Financial Services"},
    {"symbol": "FEDERALBNK", "bse_code": "500469", "name": "The Federal Bank Limited", "sector": "Banking"},
    {"symbol": "AUBANK", "bse_code": "540611", "name": "AU Small Finance Bank Limited", "sector": "Small Finance Bank"},
    {"symbol": "HAVELLS", "bse_code": "517354", "name": "Havells India Limited", "sector": "Consumer Electricals"},
    {"symbol": "POLYCAB", "bse_code": "542652", "name": "Polycab India Limited", "sector": "Cables & Electricals"},
    {"symbol": "DIXON", "bse_code": "540699", "name": "Dixon Technologies (India) Limited", "sector": "Electronics Manufacturing"},
    {"symbol": "SUZLON", "bse_code": "532667", "name": "Suzlon Energy Limited", "sector": "Renewable Energy Equipment"},
    {"symbol": "IRCTC", "bse_code": "542830", "name": "Indian Railway Catering and Tourism Corp", "sector": "Travel & Hospitality"},
    {"symbol": "RVNL", "bse_code": "542649", "name": "Rail Vikas Nigam Limited", "sector": "Railway Infrastructure"},
    {"symbol": "IRFC", "bse_code": "543257", "name": "Indian Railway Finance Corporation Limited", "sector": "Financial Services"},
    {"symbol": "IREDA", "bse_code": "544026", "name": "Indian Renewable Energy Development Agency", "sector": "Financial Services"},
    {"symbol": "MAZDOCK", "bse_code": "543237", "name": "Mazagon Dock Shipbuilders Limited", "sector": "Defense & Shipbuilding"},
    {"symbol": "COCHINSHIP", "bse_code": "540679", "name": "Cochin Shipyard Limited", "sector": "Defense & Shipbuilding"},
    {"symbol": "BHEL", "bse_code": "500103", "name": "Bharat Heavy Electricals Limited", "sector": "Heavy Electricals"},
    {"symbol": "NHPC", "bse_code": "533098", "name": "NHPC Limited", "sector": "Hydroelectric Power"},
    {"symbol": "SJVN", "bse_code": "533206", "name": "SJVN Limited", "sector": "Renewable Power"},
    {"symbol": "TORNTPOWER", "bse_code": "532779", "name": "Torrent Power Limited", "sector": "Power Generation & Distribution"},
    {"symbol": "PIDILITIND", "bse_code": "500331", "name": "Pidilite Industries Limited", "sector": "Chemicals / Adhesives"},
    {"symbol": "BERGEPAINT", "bse_code": "509480", "name": "Berger Paints India Limited", "sector": "Paints"},
    {"symbol": "COLPAL", "bse_code": "500052", "name": "Colgate-Palmolive (India) Limited", "sector": "Personal Care"},
    {"symbol": "DABUR", "bse_code": "500096", "name": "Dabur India Limited", "sector": "FMCG / Ayurvedic"},
    {"symbol": "MARICO", "bse_code": "531642", "name": "Marico Limited", "sector": "FMCG"},
    {"symbol": "GODREJCP", "bse_code": "532424", "name": "Godrej Consumer Products Limited", "sector": "Personal & Home Care"},
    {"symbol": "MCDOWELL-N", "bse_code": "532432", "name": "United Spirits Limited", "sector": "Distilleries"},
    {"symbol": "JUBLFOOD", "bse_code": "533155", "name": "Jubilant FoodWorks Limited", "sector": "QSR / Restaurants"},
    {"symbol": "PAGEIND", "bse_code": "532827", "name": "Page Industries Limited (Jockey)", "sector": "Textiles & Apparel"},
    {"symbol": "ASTRAL", "bse_code": "532830", "name": "Astral Limited", "sector": "Pipes & Building Materials"},
    {"symbol": "DEEPAKNTR", "bse_code": "506401", "name": "Deepak Nitrite Limited", "sector": "Specialty Chemicals"},
    {"symbol": "SRF", "bse_code": "503806", "name": "SRF Limited", "sector": "Specialty Chemicals & Packaging"},
    {"symbol": "PIIND", "bse_code": "523642", "name": "PI Industries Limited", "sector": "Agrochemicals"},
    {"symbol": "UPL", "bse_code": "512070", "name": "UPL Limited", "sector": "Agrochemicals"},
    {"symbol": "LUPIN", "bse_code": "500257", "name": "Lupin Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "AUROPHARMA", "bse_code": "524804", "name": "Aurobindo Pharma Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "TORNTPHARM", "bse_code": "500420", "name": "Torrent Pharmaceuticals Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "MANKIND", "bse_code": "543904", "name": "Mankind Pharma Limited", "sector": "Healthcare & Pharma"},
    {"symbol": "MAXHEALTH", "bse_code": "543220", "name": "Max Healthcare Institute Limited", "sector": "Hospitals"},
    {"symbol": "FORTIS", "bse_code": "532843", "name": "Fortis Healthcare Limited", "sector": "Hospitals"},
    {"symbol": "LTIM", "bse_code": "540005", "name": "LTIMindtree Limited", "sector": "Information Technology"},
    {"symbol": "PERSISTENT", "bse_code": "533179", "name": "Persistent Systems Limited", "sector": "Information Technology"},
    {"symbol": "COFORGE", "bse_code": "532541", "name": "Coforge Limited", "sector": "Information Technology"},
    {"symbol": "MPHASIS", "bse_code": "526299", "name": "Mphasis Limited", "sector": "Information Technology"},
    {"symbol": "KPITTECH", "bse_code": "542651", "name": "KPIT Technologies Limited", "sector": "Automotive Software"},
    {"symbol": "TATAELXSI", "bse_code": "500408", "name": "Tata Elxsi Limited", "sector": "Design & Technology Services"},
    {"symbol": "BSE", "bse_code": "540376", "name": "BSE Limited", "sector": "Financial Exchange"},
    {"symbol": "CDSL", "bse_code": "540716", "name": "Central Depository Services (India) Limited", "sector": "Financial Services"},
    {"symbol": "MCX", "bse_code": "534091", "name": "Multi Commodity Exchange of India Limited", "sector": "Financial Exchange"},
    {"symbol": "ANGELONE", "bse_code": "543235", "name": "Angel One Limited", "sector": "Broking & Fintech"},
    {"symbol": "MOTILALOFS", "bse_code": "532892", "name": "Motilal Oswal Financial Services Limited", "sector": "Financial Services"},
    {"symbol": "HDFCAMC", "bse_code": "541729", "name": "HDFC Asset Management Company Limited", "sector": "Asset Management"},
    {"symbol": "MUTHOOTFIN", "bse_code": "533398", "name": "Muthoot Finance Limited", "sector": "Gold Financing"},
    {"symbol": "MANAPPURAM", "bse_code": "531213", "name": "Manappuram Finance Limited", "sector": "Gold Financing"},
    {"symbol": "ASHOKLEY", "bse_code": "500477", "name": "Ashok Leyland Limited", "sector": "Commercial Vehicles"},
    {"symbol": "TVSMOTOR", "bse_code": "532343", "name": "TVS Motor Company Limited", "sector": "Two Wheelers"},
    {"symbol": "BHARATFORG", "bse_code": "500493", "name": "Bharat Forge Limited", "sector": "Auto Components / Forging"},
    {"symbol": "MOTHERSON", "bse_code": "517334", "name": "Samvardhana Motherson International Limited", "sector": "Auto Components"},
    {"symbol": "BOSCHLTD", "bse_code": "500530", "name": "Bosch Limited", "sector": "Auto Components"},
    {"symbol": "EXIDEIND", "bse_code": "500086", "name": "Exide Industries Limited", "sector": "Batteries"},
    {"symbol": "MRF", "bse_code": "500290", "name": "MRF Limited", "sector": "Tyres"},
    {"symbol": "APOLLOTYRE", "bse_code": "500877", "name": "Apollo Tyres Limited", "sector": "Tyres"},
    {"symbol": "BALKRISIND", "bse_code": "502355", "name": "Balkrishna Industries Limited", "sector": "Off-Highway Tyres"},
    {"symbol": "GODREJPROP", "bse_code": "533150", "name": "Godrej Properties Limited", "sector": "Real Estate"},
    {"symbol": "OBEROIRLTY", "bse_code": "533273", "name": "Oberoi Realty Limited", "sector": "Real Estate"},
    {"symbol": "PRESTIGE", "bse_code": "533274", "name": "Prestige Estates Projects Limited", "sector": "Real Estate"},
    {"symbol": "LODHA", "bse_code": "543287", "name": "Macrotech Developers (Lodha) Limited", "sector": "Real Estate"},
    {"symbol": "SUNTV", "bse_code": "532733", "name": "Sun TV Network Limited", "sector": "Media & Entertainment"},
    {"symbol": "ZEEL", "bse_code": "505537", "name": "Zee Entertainment Enterprises Limited", "sector": "Media & Entertainment"},
    {"symbol": "PVRINOX", "bse_code": "532689", "name": "PVR INOX Limited", "sector": "Film Exhibition"},
    {"symbol": "DELHIVERY", "bse_code": "543529", "name": "Delhivery Limited", "sector": "Logistics & Supply Chain"},
    {"symbol": "CONCOR", "bse_code": "531344", "name": "Container Corporation of India Limited", "sector": "Logistics"},
    {"symbol": "GMRINFRA", "bse_code": "532754", "name": "GMR Airports Infrastructure Limited", "sector": "Airports & Infra"},
    {"symbol": "PAYTM", "bse_code": "543396", "name": "One97 Communications (Paytm) Limited", "sector": "Fintech"},
    {"symbol": "NYKAA", "bse_code": "543384", "name": "FSN E-Commerce Ventures (Nykaa) Limited", "sector": "E-Commerce / Beauty"},
    {"symbol": "POLICYBZR", "bse_code": "543390", "name": "PB Fintech (PolicyBazaar) Limited", "sector": "Fintech / Insurtech"},
    {"symbol": "KAYNES", "bse_code": "543664", "name": "Kaynes Technology India Limited", "sector": "Electronics Manufacturing"},
    {"symbol": "CYIENT", "bse_code": "532522", "name": "Cyient Limited", "sector": "Engineering Design"},
    {"symbol": "SONATSOFTW", "bse_code": "532221", "name": "Sonata Software Limited", "sector": "Information Technology"},
    {"symbol": "BSOFT", "bse_code": "532400", "name": "Birlasoft Limited", "sector": "Information Technology"},
    {"symbol": "HAPPSTMNDS", "bse_code": "543244", "name": "Happiest Minds Technologies Limited", "sector": "Digital IT Services"},
]

# Fast Lookup Maps
NSE_MAP: Dict[str, dict] = {item["symbol"].upper(): item for item in INDIAN_STOCKS_DATA}
BSE_CODE_MAP: Dict[str, dict] = {item["bse_code"]: item for item in INDIAN_STOCKS_DATA}


# =======================================================================
# CORE SEARCH & RESOLUTION FUNCTIONS
# =======================================================================

def search_indian_stocks(query: str, limit: int = 8) -> List[Dict[str, str]]:
    """
    Search Indian stocks by company name, NSE symbol, or BSE numeric security code.
    Returns structured matches with symbol, name, bse_code, and sector.
    """
    query = (query or "").strip().upper()
    if not query:
        return []

    results = []
    seen_symbols = set()

    # 1. Exact BSE code match
    if query in BSE_CODE_MAP:
        item = BSE_CODE_MAP[query]
        results.append({
            "symbol": item["symbol"],
            "name": item["name"],
            "bse_code": item["bse_code"],
            "sector": item["sector"],
            "nse_ticker": f"{item['symbol']}.NS",
            "bse_ticker": f"{item['symbol']}.BO",
            "match_type": "Exact BSE Code"
        })
        seen_symbols.add(item["symbol"])

    # 2. Exact NSE symbol match
    clean_sym = query.replace(".NS", "").replace(".BO", "")
    if clean_sym in NSE_MAP and clean_sym not in seen_symbols:
        item = NSE_MAP[clean_sym]
        results.append({
            "symbol": item["symbol"],
            "name": item["name"],
            "bse_code": item["bse_code"],
            "sector": item["sector"],
            "nse_ticker": f"{item['symbol']}.NS",
            "bse_ticker": f"{item['symbol']}.BO",
            "match_type": "Exact Symbol"
        })
        seen_symbols.add(item["symbol"])

    # 3. Substring / Prefix matching in Universe
    for item in INDIAN_STOCKS_DATA:
        sym = item["symbol"]
        if sym in seen_symbols:
            continue

        name_upper = item["name"].upper()
        bse_code = item["bse_code"]

        # Check symbol prefix/substring, name substring, or BSE code prefix
        if (query in sym) or (query in name_upper) or (query in bse_code):
            results.append({
                "symbol": sym,
                "name": item["name"],
                "bse_code": bse_code,
                "sector": item["sector"],
                "nse_ticker": f"{sym}.NS",
                "bse_ticker": f"{sym}.BO",
                "match_type": "Name/Symbol Match"
            })
            seen_symbols.add(sym)
            if len(results) >= limit:
                break

    # 4. Fallback: If query not found in offline database, query Yahoo Finance Search API
    if len(results) < 2 and len(query) >= 3 and not query.isdigit():
        try:
            yf_search = yf.Search(f"{query} India", max_results=5)
            for q in yf_search.quotes:
                raw_sym = q.get("symbol", "")
                if raw_sym.endswith(".NS") or raw_sym.endswith(".BO"):
                    clean = raw_sym.split(".")[0]
                    if clean not in seen_symbols:
                        exch = "BSE" if raw_sym.endswith(".BO") else "NSE"
                        results.append({
                            "symbol": clean,
                            "name": q.get("shortname") or q.get("longname") or clean,
                            "bse_code": "N/A",
                            "sector": q.get("sector", "Listed Equity"),
                            "nse_ticker": f"{clean}.NS",
                            "bse_ticker": f"{clean}.BO",
                            "match_type": f"Yahoo {exch}"
                        })
                        seen_symbols.add(clean)
                        if len(results) >= limit:
                            break
        except Exception:
            pass

    return results[:limit]


def resolve_stock_symbol(input_query: str, exchange: str = "NSE") -> Tuple[str, Optional[dict]]:
    """
    Intelligently resolves any user input into a valid Yahoo Finance ticker
    and associated company metadata.

    Parameters:
        input_query : Can be:
                      - BSE numeric code (e.g. "500325")
                      - NSE symbol with or without suffix (e.g. "RELIANCE", "RELIANCE.NS")
                      - BSE symbol with or without suffix (e.g. "RELIANCE.BO")
                      - Company name (e.g. "Tata Consultancy Services")
        exchange    : "NSE", "BSE", or "AUTO"

    Returns:
        (resolved_ticker, stock_info_dict)
        Example: ("RELIANCE.NS", {"name": "Reliance Industries Limited", ...})
    """
    raw = (input_query or "").strip().upper()
    if not raw:
        return "RELIANCE.NS", NSE_MAP.get("RELIANCE")

    exchange = exchange.upper().strip()

    # --- Case 1: Pure Numeric BSE Code (e.g. "500325") ---
    if raw.isdigit():
        if raw in BSE_CODE_MAP:
            info = BSE_CODE_MAP[raw]
            sym = info["symbol"]
            ticker = f"{sym}.BO" if exchange == "BSE" else f"{sym}.NS"
            return ticker, info
        else:
            # Code not in top universe, try querying direct .BO in yfinance
            return f"{raw}.BO", {"symbol": raw, "name": f"BSE Code {raw}", "sector": "BSE Security"}

    # --- Case 2: Explicitly passed ticker with suffix (.NS or .BO) ---
    if raw.endswith(".NS"):
        clean = raw[:-3]
        info = NSE_MAP.get(clean)
        if exchange == "BSE":
            return f"{clean}.BO", info
        return raw, info

    if raw.endswith(".BO"):
        clean = raw[:-3]
        info = NSE_MAP.get(clean)
        if exchange == "NSE":
            return f"{clean}.NS", info
        return raw, info

    # --- Case 3: Exact Symbol in Universe (e.g. "TCS", "INFY", "TATASTEEL") ---
    if raw in NSE_MAP:
        info = NSE_MAP[raw]
        suffix = ".BO" if exchange == "BSE" else ".NS"
        return f"{raw}{suffix}", info

    # --- Case 4: Search substring / name match ---
    matches = search_indian_stocks(raw, limit=1)
    if matches:
        top = matches[0]
        sym = top["symbol"]
        suffix = ".BO" if exchange == "BSE" else ".NS"
        return f"{sym}{suffix}", top

    # --- Case 5: Default raw ticker fallback with chosen exchange suffix ---
    clean_sym = re.sub(r'[^A-Z0-9&-]', '', raw)
    suffix = ".BO" if exchange == "BSE" else ".NS"
    return f"{clean_sym}{suffix}", None


def get_popular_stock_chips() -> List[Tuple[str, str, str]]:
    """
    Returns curated list of popular stocks: (Symbol, Short Name, BSE Code)
    """
    return [
        ("RELIANCE", "Reliance", "500325"),
        ("TCS", "TCS", "532540"),
        ("INFY", "Infosys", "500209"),
        ("HDFCBANK", "HDFC Bank", "500180"),
        ("ICICIBANK", "ICICI Bank", "532174"),
        ("ITC", "ITC", "500875"),
        ("TATASTEEL", "Tata Steel", "500470"),
        ("BHARTIARTL", "Airtel", "532454"),
        ("LT", "L&T", "500510"),
        ("TATAMOTORS", "Tata Motors", "500570"),
    ]
