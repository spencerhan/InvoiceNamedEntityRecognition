
# This python file is used to create the very first entity labelled dataset, based on spacy PhraseMatcher and regular expression matcher. 

# The spacy matcher is not working perfectly, as addressed in the final report. 

# The PhraseMatcher used to match every single Merchant name to a handpicked Merchant name list. Therefore it is extremely labour intensive.
# I went through every invoice data to handpick every merchant name found there and then construct the merchant name list. 

# The entity label is done at token level not word level.   
import json
import ndjson
import spacy
import csv
import re
import sys
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.lang.de.punctuation import _quotes
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import plac
from pathlib import Path
import random
import itertools

# default spacy tokenizer will not break works at '/', thus I modified it to break words into tokens at '/'

# however, the tokenizer is still not perfect for this study, as it has been addressed in my report. 
# more time needs to be spent on building a perfect tokenizer 

def custom_tokenizer(nlp):
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[{al}])\.(?=[{au}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
            r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA),
            r'(?<=[{a}])[:<>=](?=[{a}])'.format(a=ALPHA),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])([{q}\]\[])(?=[{a}])".format(a=ALPHA, q=_quotes),
            r"(?<=[{a}])--(?=[{a}])".format(a=ALPHA),
            r"(?<=[0-9])-(?=[0-9])",
            r"(?<=[0-9])/(?=[0-9])",
        ]
    )

    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                     suffix_search=nlp.tokenizer.suffix_search,
                     infix_finditer=infix_re.finditer,
                     token_match=nlp.tokenizer.token_match,
                     rules=nlp.Defaults.tokenizer_exceptions)



# loading spacy tokenizer 
nlp = spacy.load('en_core_web_lg')
nlp.tokenizer = custom_tokenizer(nlp)
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else:
    ner = nlp.get_pipe('ner')

# The purpose of the following three functions is to convert spacy token location to its index location found in a text. 

# Spacy Matcher and PhraseMatcher return the position of a matching token found in a text, but we need the beginning index and end index of a matching token in a text. 

#Therefore, we use matching token location + token length to get the beginning index and end index of a matching token. 
def offsetterP(lbl, doc, matchitem):
    o_one = len(str(doc[0:matchitem[1]]))
    subdoc = doc[matchitem[1]:matchitem[2]]
    o_two = o_one + len(str(subdoc))
    return (o_one, o_two, lbl)

def offsetterT(lbl, doc, matchitem):
    span = str(doc[matchitem[1]:matchitem[2]])
    result = str(doc).find(span,len(str(doc[0:matchitem[1]]))) 
    o_two = result + len(span)
    print(len(span))
    return (result, o_two, lbl)

def on_match(matcher, doc, id, matches):
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        print(span)
        # here we get the information of where the matching token start, and end in an invoice text.  
        print(match_id, string_id, start, end, span.text)



# The following spacy matchers are built based on different regular expressions to locate the DATE, TIME, GST and AMOUNT tokens.

# the result is not perfect and requires a lot of manual corrections later. 
dateMatcher = Matcher(nlp.vocab)
timeMatcher = Matcher(nlp.vocab)
gstMatcher = Matcher(nlp.vocab)
amountMatcher = Matcher(nlp.vocab)
# patterns that we considered that match a date
pattern1 = [{'IS_DIGIT': True}, {'ORTH': '/'}, {'IS_DIGIT': True},
            {'ORTH': '/'}, {'IS_DIGIT': True}]  # number/number/number
pattern2 = [{'IS_DIGIT': True}, {'ORTH': '-'}, {'IS_ALPHA': True},
            {'ORTH': '-'}, {'IS_DIGIT': True}]  # number-string-number
pattern3 = [{'IS_ALPHA': True}, {'ORTH': '-'}, {'IS_DIGIT': True},
            {'ORTH': '-'}, {'IS_DIGIT': True}]  # string-number-number
pattern4 = [{"TEXT": {'IS_DIGIT': True}, "LENGTH": 4}, {'ORTH': '-'}, {"TEXT": {'IS_DIGIT': True},
                                                                       "LENGTH": 2}, {'ORTH': '-'}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 2}]  # yyyy-mm-dd
pattern5 = [{"TEXT": {'IS_DIGIT': True}, "LENGTH": 2}, {'ORTH': '-'}, {"TEXT": {'IS_DIGIT': True},
                                                                       "LENGTH": 2}, {'ORTH': '-'}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 4}]  # dd-mm-yyyy
pattern6 = [{"TEXT": {"REGEX": r"\d{1,2}(?:[stndrh]){2}?"}}, {'IS_ALPHA': True}, {
    'ORTH': ',', 'OP': '+'}, {'IS_DIGIT': True}]  # dates of the form August 10th, 2018
pattern7 = [{'IS_ALPHA': True}, {"TEXT": {"REGEX": r"\d{1,2}(?:[stndrh]){2}?"}}, {
    'ORTH': ',', 'OP': '+'}, {'IS_DIGIT': True}]  # dates of the form 10th August, 2018
pattern8 = [{"TEXT": {'IS_DIGIT': True}, "LENGTH": 2}, {'ORTH': ' '}, {'IS_ALPHA': True}, {
    'ORTH': ' '}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 4}]  # 16 February 2016
pattern9 = [{"TEXT": {'IS_DIGIT': True}, "LENGTH": 4}, {'ORTH': ' '}, {'IS_ALPHA': True}, {
    'ORTH': ' '}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 2}]  # 2016 February 16
pattern10 = [{'IS_ALPHA': True}, {'ORTH': ' ', 'OP': '?'}, {'IS_DIGIT': True}, {
    'ORTH': ',', 'OP': '+'}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 2}]
pattern11 = [{'IS_ALPHA': True}, {'ORTH': ' ', 'OP': '?'}, {'IS_DIGIT': True}, {
    'ORTH': ',', 'OP': '+'}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 4}]  # Dec 05, 15
pattern12 = [{'IS_DIGIT': True}, {'ORTH': ' '}, {'IS_ALPHA': True}, {
    'ORTH': ',', 'OP': '+'}, {'IS_DIGIT': True}]  # 23 Feburary, 2015
pattern13 = [{'IS_DIGIT': True}, {'ORTH': ' ', 'OP': '?'}, {'IS_DIGIT': True}, {
    'ORTH': ',', 'OP': '+'}, {"TEXT": {'IS_DIGIT': True}, "LENGTH": 4}]


# we put all patterns for a data in a list, then add all patterns to a date pattern matcher object. 
patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6,
            pattern7, pattern8, pattern9, pattern10, pattern11, pattern12, pattern13]
for i in patterns:
    dateMatcher.add("DATE", on_match, i)

# patterns that we considered that match a time
pattern1 = [{"TEXT": {"REGEX": r"(1[0-2]|0?[1-9]):([0-5][0-9]):?([0-5][0-9])?"}}, {
    "text": {"REGEX": r"([AaPp][Mm])|(([Aa]|[Pp])\.[Mm]\.)"}}]  # include am pm and a.m. p.m.
pattern2 = [
    {"TEXT": {"REGEX": r"(([1-2][0-5]|0?[1-9]):([0-5][0-9]):?([0-5][0-9])?)"}}]

# we put all patterns for a data in a list, then add all patterns to a time pattern matcher object. 
patterns = [pattern1, pattern2]
for i in patterns:
    timeMatcher.add("TIMESTAMP", on_match, i)


# patterns that we considered that match a amount entity
pattern1 = [{'ORTH': '$', "OP": "+"}, {"Text": {"REGEX": r"(^-?(?:0|[1-9]\d{0,2}(?:,?\d{3})*)(?:\.\d+)?$)"}}]
pattern2 = [{"TEXT":{"REGEX": r"([a-zA-Z]\d{1,3})"}, "OP": "+"}, {'ORTH': '$', "OP": "+"}, {"Text": {"REGEX": r"(^-?(?:0|[1-9]\d{0,2}(?:,?\d{3})*)(?:\.\d+)?$)"}}]
pattern3 = [{"TEXT":{"REGEX": r"([a-zA-Z]\d{1,3}\$?)"}, "OP": "+"}, {"Text": {"REGEX": r"(^-?(?:0|[1-9]\d{0,2}(?:,?\d{3})*)(?:\.\d+)?$)"}}]
patterns = [pattern1,pattern2, pattern3]
for i in patterns:
    amountMatcher.add("AMOUNT", on_match, i)

# patterns that we considered that match a GST entity
pattern1 = [{"TEXT": {"REGEX": r"^[0-9]{2,3}$"}}, {'ORTH': '-', 'OP': '+'}, {"TEXT": {"REGEX": r"^[0-9]{3}$"}}, {"ORTH":"-", "OP": "+"},{"TEXT": {"REGEX": r"^[0-9]{3}$"}}]
patterns = [pattern1]
for i in patterns:
    gstMatcher.add("GST_NO", on_match, i)

# We use spacy's Phrase Matcher to find all Merchant entity tokens in each invoice text.
merchantMatcher = PhraseMatcher(nlp.vocab)
# as mentioned earlier, I handpicked all merchant names from each invoice entires to construct the following merchant name list. 
# phrases = ['GOURMET BURGER KITCHEN','QUICK EASY CONUENIEN', 'C ABCHARGE','NZ DRYCLEANERS','HMSHOST NZ LIMITED','Dominos Pizza', 'Flying Burrito Brothers', 'THATS AMORE', 'Urban Purveyor THE ARGYLE Group BAR', 'VIETNAMESE RESTAURANT', 'THE CASTLE', 'MAGNIFIX NZ L.TD', 'Ractisson Hotel', 'Huntington Pty Lid', 'Brewers Coop','DONUT KING', 'TRATTORIA PREGO', 'BREWDOG SHOREDITCH', 'B3 CAFE', 'California Sauce Labs', 'wagamamd', 'Red Rocks', 'INGCEORTUIN ROASTERS','The Fine Wine Delivery Company','GRANGE HOLBORN','FINE WINE', 'Harbor Court Hotel', 'waterTo DLdUII', 'ANZ', 'SIERRA', 'Slate Restaurant Bar', 'BOULEVARD', 'MIThA', 'AIR NEW ZEALAND', 'Sanctuary Hotel', 'Southern Cross Benefits Ltd', 'Pret A Manger', 'Prince Public Bar', 'The University of Auckland Business School', 'KMART', 'SIDART RESTAURANT', 'NEW ZEALAND POST', 'INGOGO', 'A8 PRIVATE CAB', 'PLURALSIGHT', 'CAFFE VERGNANO', 'CUMULUS INC', 'GRAND HOTE', 'cerprrd', 'Sheraton Tower', 'COMMONS', 'Last Holbor', 'CHILANGO S', 'OLIVER BROWN', 'Auckland Copy Shop Limited', 'WELLINGTON VENUES', 'Subway Wellington Internationa Airport', 'babyclty', 'KMART BOTANY', 'FRAEDOM CORPAT', 'The TITTy Shilling', 'WELLINGTON WELL SHUTTLES INGTON COMB INED', 'BAAN THAI', 'GUZMAN Y GOMEZ', 'NEW WORLD', 'Discount Taxis 005', 'amgos', 'Cafe Organic', 'TAXI TOUNCARS', 'Hotborn Whippet', 'SESAMO', 'UMAMI BURGER', 'Neuielind Pty Limited', 'warehouse, stationery', 'countdown', 'General Distributors Ltd.', 'PLAYACABANA BARRIO COREA', 'meisterlabs', 'MeisterLabs GmbH','takahe, Restaurant Bar','RESTAURANI LRANI','Corporate Bunny Diners', ' NORTH EAST TAX, BROKER','Apple Sales New Zealand','FOURBF BFIVE IVE CBD','HRG New Zealand', 'BURGER KING', 'Optus Networks Pty Ltd','CAFE DESTINO','COBRA LIQUOR','P&Q', 'ROLL N SUSH','Hish Hoibor', 'Joes Garage', 'Starry Kitchen', 'STARRY THAI','TRANZ METRO','Human Resources Institute of New Zealand Incorporated','WELL COURTENAY INGTON PLACE', 'llece', 'CHEERS BAR 8 CRILL', 'Linkedin', 'SUNHAY HOLDINGS PL', 'Zap 4 Esan Thai Cuisine', 'Sixtree', 'SPAGHETTI HOUSE LTD', 'Thistle Bloomsbury Park', 'Mojo Victoria Square', 'FREE STATE COFFEE', 'OPTIMA TOURS, AND LIMOUSINES', 'Adria', 'Spoon In The City', 'ii li sf', 'Uber', 'Internationa Institute of Business Analysis', ' International Institute, of Business Analysis', 'Citadines', 'Eventbrite','WCT TAXIS', 'Satay Malaysia', 'DESOTO CAB', 'Twvickenham Experience Limited', 'PowerPivotPro LLC', 'ebganes', 'Apress Media LLC', 'Hogg Robinson (Travel) Ltd', 'toasted', 'Fraedom', 'ALLSPORTS - DOMINION ROAD', 'GREEN CABS', 'Stirling Sports', 'The Brewers Co-operative', 'BUGER, BURGER', 'UNIVERSAL PHONE REPAIRS', 'PAKhSAV', 'Interflora Pacific Unit Ltd', 'THE CROUN HOTEL', 'Inn of Court', 'THE EXCHANGE HOTEL', 'Plum Burger', 'Fullers Group Limited', 'ORB', 'CLARKE MUIRS PTY LTD', 'FLY PARK', 'Shamiana Restaurant', 'Cafe Hanoi', 'Coles Supernarkets Australla Pty Ltd', 'B A R, Lourinha', 'Telerik Inc', 'SHEBA INDIAN RESTAURANT', 'Clickatel Inc', 'TotoA', 'The Packinuton', 'DIXON ST. DEL', 'CABCHARGE', 'Travelodge', 'RYDGES', 'TAX HDLHARGE', 'BRCC TAXI BROKERS', 'S CAFE', 'AMAL TAXIS', 'SkySport Grill', 'Pandoro Panetteria', 'Steer & Beer', 'LILYS FLORIST', 'Roadmunk', 'Mahuhu Espresso', 'aycar Electronics New Zealand', 'ROTI CHENAI', 'T-LINK LIMITED', 'RK Convenience store CBD', 'Caltex Grey Lynn', 'Independent Computer Search Pty Ltd', 'Budger', 'MOJO PONEKE', 'BUNNINGS WAREHOUSE' , 'HUTT AND CITY TAXIS', 'trademe', 'STELLAR KITCHEN', 'optus Retailco Pty Ltd', 'thewarehousell', 'Sydneys Airport Train', 'Exccllence Billiards', 'supershuttle', 'GODIVA', 'TRYING IT OUT LIMITE', 'STOKER MODELS', 'Z ELLERSLIE', 'LONDON UNDERGROUND LIMITED', 'Warehouse Statiocery Ltd', 'gloria', 'tarro, FRESH' , 'RAILWAY HOTEL WINDSO', 'WESTPAC', 'Rosie', 'NOVOTEL', 'STEP ENTERTAINMENT', ' Esquires Tangihua Street', 'yjourney', 'Halswell Lodge', 'Symantec Corporation', 'Steak Do', 'Trax Cafe & Bar', 'AROMA', 'Z Cluay St', 'OCONNELLS CENTENARY HOTEL', 'EDDY 4 TAXIS', 'PORTELLO ROSSO', 'AirportLink', 'HAWKER', 'Goldline Taxis', 'Autobahn Cafe Papakura', 'Jetstar', 'PB Technologies', 'New Inn Pizza', 'Cabbys', 'BAKERS DELIGHT ST HELIERS', 'EBISU', 'OREILLY', 'Village 88',  'SM TAXI', 'Alert, TAXXI', 'THE COFFEE CLUB', 'Pizza Orgasmica', 'IMAS TAXI', 'MUSEUM ART HOTEL', 'Fortuna C', 'AUCKLAND GOLDLINE TA', 'NONBOON POON', 'NEW TERLAND POST', 'MDonalds, Intarnstional', 'Postdot Technologies Private Limited', 'Pacific Leisure Group Limited', 'Datachoice Solutions Ltd', 'IL POMODORO ITALIAN', 'LOTUS SUPERMARKET', 'BeanBox', 'Escape Games Limited', 'PizzaExpress', 'ALEX THE COBBLER', 'Warehouse Stationery Ltd', 'ABN TAXI', 'Dell US', 'Waitrose', 'PIZZA HUT NZ', 'State of Grace', 'BAIA THE ITALIAN', 'CURICE CAFE & BAR', 'Codemania', 'SPENDVI SION LINITED', 'BIKANERVALA', 'COUNTDOWN AUCKLAND', 'CAFE VANILLA', 'GROUPON', 'ARVEL BAR AND GRILL', 'Noel Leeming', 'GOLD BAND', 'IMA-Deli', 'The Growth Faculty', 'TALKING STICK RESORT', 'GOURMET DELIVERIES QUEENSLAND', 'DRAGONFLY','Scrum Alliance, Inc.', 'DAIKOKU', 'Johnny Barrs Fresh Food Bistro', 'rewarehousens','Spot less Facility Services', 'LILLIPUT MINIGOLF', 'WESTMERE BUTCHERY LTD', 'Mercure','accorhotels', 'accor hotels', 'NER TALANO POST', 'COPTHORNE, HOTEL & RESORT', 'SYLVIA PARK FOOD BARN LTD', 'PAKnSAVE', 'staywellgroup','stay well group', 'Park Regis Citv Centre', 'SAI Investments NZ Ltd', 'dick smith', 'GOLDLINE TAXI', 'LS Travel Retail NZ Limited', 'Automation Guild', 'Easy Tiger', 'SMART CABS LIMITED', 'te Issue Brreze', 'ANCESTRAL', 'Coles Supermarkets Australia Pty Ltd', 'Better Burger','Victoria Food Services' ,'BP 2GO ELLERSLIE', 'McDonalds', 'MONTECITO', 'LA BELLA ITALIA', 'JONES GROCER STORES', 'BLUE LINE TAXI', 'The Abbey', ' KARIANNE CHALMERS', 'Sals', 'Taka Enterprise Ltd.', 'CLASSICO PASTA AND PIZZA', 'New World Metro', 'AUCKLAND CO-OP TAXIS', 'BLASTACARS WEST AUCKLAND LTD', 'MoMos SAN FRANCISCO GRIL, BAR', 'DIN TAI FUNG', 'Venture by Design Limited', 'Henry Coffee Bar', 'FIRST DIRECT TAXIS', 'Wildfire Churrascaria', 'Paper Plus Parnell', 'paperplus', 'BROTHERS, BREWERY', 'BROTHERS BREWERY', 'Brothers Beer', 'Farmers', 'FARMERS-NEWMARKET', 'CABBIEXPRESS', 'Hotels.com', 'GECAL ENTERPRISES PT', 'Harbour View Motel', 'N L Krisht Ltd', 'Mo jo Coffee', 'Safari Ontine Books', 'Safari Books Online', 'Quay Street Cafe', 'RESTAURANT ASHA', 'BLUE STAR','cabit', 'VINYL BAR', 'MAX BRENNER AUST', 'LOUEBITE', 'CORPORATE CABS', 'Movida', 'Mclonalds Manners Street', 'MIDNIGHT ESPRESSO', 'GETRIDE', 'IC Agile LLC', 'icagile', 'MEARS, TRANSPORTATION, GROUP', 'MearsTaxi','Eves Pantry', 'EUROPCAR NEW ZEALAND', 'The Naked Duck', 'WDS HOTELS PTY LTD', 'Heathrow Taxi Services', 'Arisun', 'Segafredo Zanetti New Zealand Limited', 'DELMONICO GOURMET FOOD MARKET', 'ELLERSLIE WINE CELLARS LTD', 'GoDaddy', 'WISHBONE AIRPORT', 'Billie Chu', 'Parcelforce', 'CUMULUS EATING HOUSE', 'BAKERS DELIGHT REMUE', 'BAKERS DELIGHT REMUERA', 'Amano', 'GLENGARRYS', 'BREW ON QUAY', 'Enigma Cafe', 'Visa4UK', 'GUY PATERSON, ENTERPRISES LTO', 'The Famous Curry Bazaar', 'ARTIS', 'Virgin', 'FIDELS' , '55 SEJFI FRPTATET', 'Corner Bakery Cafe', 'LA STATION DES SFORTS', 'THE LONDON EDITION', 'Honey Cafe', 'MO jO' ,'Waterloo Station' , 'St John', 'STJOHN', 'TGI Fridays UK', 'Australian Government', 'Department of Immigration', 'Border Protection', 'Steakhouse', 'GPO Wood Fired Pizza', 'QANTAS AIRWAYS', 'BRIX CAFE', 'The Espresso Bar', 'THE DOG & BEAR', 'Gastronomy Cassels Ltd.', 'GitHub', 'MAMAK MALAYSIAN', 'chancellorhotel', 'madisonlondon', 'NOEL LEEMING','FRESCO', 'Fresco', 'GREEN, CABS', 'BP Cornect Frankton', 'inland Revenue Departrnet', 'OPORTO', 'oporto', 'Tank Juice Bar', 'Rebel Sport', 'LOUS FISH SHACK', 'SKYCITY', 'Auckland Int. Airport', 'Rebel Sport','The Barber Shop', 'NORTHCOTE TAUERN', 'Caffe Ladro', 'Australia & New Zealand Testing Board (ANZTB)', 'ANZTB', 'Co Hote we imstor', 'KINKI', 'OfficeMax', 'SUBWAY', 'Stevens', 'AREAS USA LAX', 'MOF Bar & Kitchen', 'Sear S Fine Foods', 'Mad Mex', 'NLINE SECURITYSERVICES', 'The Menties Sydney', 'BEDEUNU', 'thewarehousew', 'Woolworths', 'CAFE EXCELLO', 'The General Assembly', 'PASCOE', 'Expedia', 'GrabOne', 'White Lion', 'COSTA COFFEE', 'Microsoft', 'MW EAT LTD', 'OSTRO', 'AIR CANADA', 'Smith The Grocer', 'BLACK CABS', 'Embarcadero', 'Wellinston Hellinston Star', 'MARKS8, SPENCER', 'SMITH & CAUGHEY', 'TeamViewer', 'Como St Cafe Ltd', 'COAST, CAFE', 'SUNCORP BANK', 'AMORA HOTEL', 'Matterhorn', 'NEW ZEALAND RED CROSS', ' IRISH TIMES', 'lokio Hotel', 'Zoo Bar', 'The Ship Tavern', 'E&0 Kitchen & Bar', 'BLUE INE TAXI', 'Scrum Alliance', 'LITTLE PECKISH', 'Cafe tontrope', 'FRESCO', 'Red Robin Gourmet Burgers', 'NAPA FARMS BAR & CAFE', 'ZAFERON', 'Oitolana & Milse', 'HANGOUT CAFE', 'ELLERSLIE INTERNATIONAL', 'Lucy Liu Kitchen & Bar', 'Bookmanque', 'BOOK MARQUE', 'SERVERCASHER', 'Harbour Lights', 'Bier Make', 'Shed 5', 'Shed5', 'HAURAKI', 'Grand Mercure', 'NEM WORLD', 'ITACHO SUSHI', 'Scratch Bakers', 'LOVE A DUCK', 'Rubber Monkey', 'Wine Loft', 'WINE LOFT', 'rubbermonkey', 'Caffe De Mondo', 'tastesonthefly', 'RYDGES', 'Rydges', 'The Store', 'AlphaCity Ltd', 'AlphaCity', 'GLG GROUP PTY LTD', 'WILSON PARKING', 'DOOASAD', 'Z Glen Innes', 'Secure Parking NZ Ltd', 'TAVICHAFIGE NZ LTD', 'COURTYARD BY MARRIOTT', 'S TALHINI ROD', ' Beirut', 'BAKE & BEANS', 'thewarehouse', 'Comfort Cab Co', 'NEW ZEALAND MOVERS', 'HRG AUCKLAND', 'Squirrels Store', 'airsquirrels', 'RENDEZUOUS GRAND HOT', 'ROKUJUNI', 'amsii', 'Oaks Goldsbrough', 'oakshotelsresorts', 'flexipurchase', 'AEROPORT TAXI & LIMOUSINE', 'Scarfes Bar', 'BURGERFUEL', 'Sweet Mothers Kitchen', 'Park Regis Concierge Apts', 'Andaz Otawa', 'andaz', 'hudsons, COffee', 'The Esplande', 'Devopsdays', 'devopsdays', 'Flowers After Hours', 'BI UE INE TAXI', 'The Tonaville Hetal Trat, GRAND HOTEL', 'maxbrenner', 'No 1 BBQ Restaurant', 'TAXI4UR', 'chipotle', 'FOURX, POINTS', 'fourpoints', 'JOES BRYCE GARAGE', 'estausa', 'nzte', 'NEW ZEALAND, TRADE & ENTERPRISE', 'Travelex', 'The Lambton', '1 COUPON', 'tradingplace', 'Apple Distribution Internationa', 'Officeworks', 'HOTEL WINDSOR', 'thehotelwindsor','GOLDEN AUTUMN LEAVE', 'THE ENGINE ROOM', 'Radisson BlU Eduardiar', 'Meriton Property Services Pty Ltd', 'MERITON', 'Meriton Serviced Apartments', 'BAVARIA', 'Hyatt Regency', 'Shaky Isles', 'THE PORTERHOUSE', 'Techdirt', 'bechdirt', 'fraedom', 'sierracoffee', '7 ELEVEN', 'Delissimo Deli', 'Wilso Parking', 'Glengarry Wines Ltd', 'GLENGARRY', 'GLENGARRY', 'WATERSIDE HOTEL', 'Wtersich Flindoraity ltd', 'THAI PRINCESS', 'Thai Princess', 'Apple', 'La Cloche Central', 'Z Quay St', 'Z Johnsonville', 'SORRENTO in the PARK', 'CARETAKER','Papa Goose', 'LE FLMMOR RUBS', '2degrees', 'inmotionstores', 'sen Frencisco IntArport', 'Museum Art Hotel', 'MUSEUM, ART HOTEL','WEST PLAZA HOTEL', 'the West Plaza Hotel','westplaza', 'THE POMMELERS REST', 'REBO CAFE & BAR', 'PRETTA', 'FEDERAL EXPRESS', 'Emirates', 'BLUE BIRD COFFEE', 'S Mobile.', 'T-MOBILES', 'T-MOBILE', 'LETS DU COFFEE PTY', 'FIRE SERVICES LTD', 'Kata Online Learning', 'kata.teachable', 'ROXY BEAUJOLIS LTD', 'China Restaurant & Bar', 'SUPERNORMAL', 'Magshop', 'magshop', 'Amazon Web Services', 'Amazon.com', 'amazon.com','Amazon Export Sales LLC', 'Frasers', 'BettYs', ' bettys', 'Baggins Shoes', 'BARGAINS GALORE', 'GMCABS', 'G, CABS', 'old Bank of England', 'HERCULES MORSE', 'Restaurant 1', 'RISE', 'PRESTIGIA', 'TRAVEL EMOTIONS LTD', 'Solarwinds Software Europe Limited', 'TOTO', 'Grand Hyutl Seotty', 'HYATT', 'Grand Hyatt Seattle', 'ADINA APARTMENT HOTEL', 'PizzaExpress', 'ITICKET.C CONZ Limited', 'Department of Homeland Security', 'BAGGINS SHOES', 'FAWKNER EXPRESS', 'Hotal Management (Featherston St) Limited', 'EIGHTY EIGHT CUPS', 'HASHIGO ZAKE', 'UNCLE BILLS WHOLESAL', 'CAFE DU NOUVEAU HONDE INC', 'Luxor Cab', 'universalorlando', 'Shoreline Bar', 'AIS', 'PRIME HOTELS', 'Grand Mercure Apartments', 'McDonaldrc', 'Bunnings', 'DISCOUNT TAXIS LTD', 'THE CHEESECAKE SHOP', 'Caffe Nero', 'Blade Master', 'blademaster', 'CENTRAL, COMPUTERS', 'Centra Computers San Francisco', 'PILKINGTONS', 'PAUL MARTINS AMER GRILL', 'Hadoop Summit', 'The Vintry', 'vintryec4', 'PANCAKES ON THE ROCKS', 'Movida Aqui', 'Auckland, Airport', 'airnewzealand', 'Air New Zealand Limited', 'INTERCONTINENTAL.', 'MELBOURNE THE RIALTO', 'intercontinental', 'Nordstron', 'SEGAFREDO ZANETTI NEW ZEALAND LTD', 'segafredo', 'THAI CLASSIC CUISINE', 'The Warehouse', 'The Kingaley', 'BLH Hotais Management', 'Belgtan Beer Cafe Heritage', 'Cook Strait Bar', 'Trev Park', 'Hand Rock Cafe', 'Marche International', 'marche', 'MERITON', 'Meriton Property Services Pty Ltd', 'Meriton Property Services Pty ltd', 'LE PAVILLON NANPIC INC', 'Fork & Brewer', 'CUCKOO COCKTAIL EMPORIUM', 'habuii', 'Babu Ji', 'SSP UK', 'posbosshq', 'Octopus Deploy', 'Delaware', 'Little India', 'LITTLE INDIA', 'Brioche Bakery and Cafe', 'AIRLINE LIMOUSINE', 'Pndepfyme', 'Pandoro Panetteria Limited', 'myalrportshopphg', 'Auckland Airport Ltd', 'travelpharm', 'ravelpharm', 'CHOW RESTAURANT 24', 'chow', 'impc', 'The London Print Company', 'LANEWAY CAFE', 'ThirstyBear Brewing Company', 'Mish Mosh', 'CARLTON EVENTS LIMITED', 'Noahs New York Bagels', 'pexce rentais', 'UNITED', 'PhotoBox', 'SWEAT SHOP', 'Sweat Shop', 'GENERAL DISTRIBUTORS LIMITED', 'Tho Esprosso Room', 'Sainsburys Supermarkets Ltd', 'OAKS CASINO TOWERS', 'BlueSnap', 'American Airlines', 'JB HIFI', 'GREEN CABS SAFER LTD', 'DOYLE COLLECTION', 'FINANCIAL DISTRICT HARDWARE', 'Cook Strait Bar', 'PIZZERIA LIBRETTO', 'Shiraaz Fine Indian Cuisine', 'KITCHENWARE SUPERSTORE', 'GALLERY EFY CAFE', 'Spendvision', 'BECASSE SYDNEY', 'THE HOTEL WINDSOR', 'THE, HOTEL WINDSOR', 'Merrywell Burger & Bar', 'Crown Melbourne Ltd.', 'New Zea land Post Limited', 'New Zealand Post', 'Kres Restaurant', 'TAXI LImo SERVICES', 'KIWI CABS', 'TAXI AND LMO SERVICE', 'Armacillo Willy Mateo', 'Digital River Ireland Ltd', 'Optus Trading Retaic', 'The Cheesecake Factory', 'TRANACII CORDAR', 'CROWIN IMETROPOIL', 'NORTH SHORE TAXIS', 'Sherlock Holmes', 'The Glass Goose', 'pragprog', 'BAYK, BASK', 'LES DELA GAMiNS DE MTL', 'Urban Turban', 'THE, COFFEE, CLUB', 'WOOLWORTHS', 'Shutterstock Netherlands', 'Ots Deperpr', 'BAY COFFEE', 'Sempre', 'MARRIOTT', 'buysub.con', 'lookat.co.nz', 'Shaky, Isles', 'Newmarket Hote', 'LAMAROS CAFE BAR', 'McDonald', 'ZOHO', 'Urban Soul', 'RADACAD Systems Limited', 'WAREHOUSE STATIONERY LTD', 'PJ OBriens', 'FAIRLZ', 'MAC BREWBAR', 'BETLR17ESANII','Nosh Mt Maunganui', 'Urban Gourmet Ltd', 'URBAN, OURMET', ' H O T E L &, C A S I N O', 'PJ OBriens', 'SAPPHIRE', 'Z TRIANGLE', 'NZC TAXI', 'Sublime Text', 'Sublime HQ Pty Ltd', 'Netregistry Pty Ltd', 'THE ARGYLE', 'Tank Fort Street', 'LA THURLI SNACK ON TEET BAR', 'ST-HUBERT DU PARC', 'McDanalds Myer II', 'WHITCOULLS', 'Whitcoulls', 'The EE stare', 'Hotel Parking', 'THE COUNTY GENERAL', 'SECOND CUP', 'JERVOIS STEAK HOUSE', 'AUGUSTINE ON GEORGE', 'British Airways', 'Lynnmal', 'footprint, associates ltd', 'POKENO COUNTRY CAFE', 'Safitel', 'Great India Restaurant', 'Eau De Uie', 'T&A GROUP PTY LTD', 'Delaware, North, GRINDHOUSE', 'Er Car Rontal, N2 Leisure Lid', 'webjet', 'Webjet Marketing Pty Ltd', 'THE SOURCE', 'PRAVDA CAFE', 'newzealandnatural', 'New Zealand Natura Quay Street', 'Nicks Seafood Restaurant', 'paperplus', 'Paper Plus', 'whitcoull 2011 Limited', 'Rogue & Vagabond', 'Little Rumour', 'CULPRIT', 'culprit', 'The White Hart', 'Surestyle Limited', 'The East Chinese Rest', 'Ottawa Market Keg', 'Mitre 10', 'NSW, Transport', 'FRANCS', 'TAN KNVOXOY', 'Agile on the Beach', 'hyatt', 'RHUB CONFERENCE','Uppercrust', 'MUTUAL COMMUNICATION', 'Mutual Communication Ltd', 'HELL Pizza Bishopdale', 'HIGHLANDER BAR', 'AMBASSADOR TAX', 'VERIFONE TAXI', 'BP Connect Papakura', 'Taylor Cafe', 'Adina Apartment Hotel', 'Zina', 'KICHEF', 'Starbucks Coffee Canada', 'CABVISION NETWORK LTD', 'Quay Park Pharmacy', 'thewarehousell', 'DHL Express', 'Humble Bundle', 'SACO The Serviced Apartment Company', 'Waleifrore, Grill', 'THEINFORMATION MANAGEMENT GROUP', 'The Information Management Group New Zealand Limited', 'CALTEX PENROSE', 'Sterling 101', 'Bethel Woods', 'Dell Canada, ''Grand Hyatt San Francisco', 'Farvey Norman Stores (NZ) Pty Ltd', 'Harvey Norman', 'CAFE FRED', 'Apigee', 'APIGEE', 'inland Revenue Department', 'MERCURY ESPRESSO', 'PACK & SEND Auckland City', 'BYRON', 'LORETTA', 'GPO Oyster Bar','STEAK AND CO', 'HRG Consortia', 'manhattanoteltimessquare', 'OrderMate', 'SPEEDY SIGNS DOWNTOWN', 'Smythe Holdings Ltd', 'Bitrise Ltd.', 'areosso', 'DIAL A CAB', 'R&R BBQ Shop', 'Rod & Roger', 'CAPITAI TAXI', 'Whitcoull 2011 Limited', 'ACTIVE AUTO TAXI', 'M 0 N S 0 0 N P 0 0 N', 'The Big Game Company', 'P A R K HOTEL', 'Park Hotel Lambton Quay Ltd', 'Roses Are Red', 'Briscoes', 'BRISCOES', 'T2 NZ Ltd', 'MERCURE SYDNEY POTTS POINT', 'ANIXIS PTY LTD', 'Harvey, NOrMan, (NZ), Pry, Ltd', 'NELSON CITY TAXIS', 'Fairmont Hotels & Resorts', 'Stockaio', 'stockaid', 'Stockaid', 'The Jefferson', 'Paddle.com Market Ltd', 'waitrose', 'OLUNNEYS TIMES SQUARE PUB', 'FIREPOT CAFE', 'Simunovich Olive Estate Limited', 'EER BISTRO', 'THE DOCK ON QUEEN', 'RESTAURAKT-BAR LA BAIE', ' Odettes Eatery', 'WGTN TAXI', 'HB COMBINED TAXIS', 'Sofra Mayfair', 'Auckland Airport', 'CARLTON PARTE HENDERSON', 'carltonpartyhire', 'DOUBLETREE', 'HILTON', 'HockeyApp', 'Westfield', 'Cantina', 'CANTINA', 'FOG HARBOR', 'Destination Quick Service', 'Rocket Kitchen', 'Chat Thai', 'MiniTool Support', 'MISTER MINIT', 'KAFFEE EIS', 'Anason', 'Typo', 'TYPO', 'Cafe Carl Bane', 'OPERA KITCHEN', 'Spend Vision', 'thestationsf', 'apregnantmanpub', 'Eves Pantry Limited', 'Metliks Me inks', 'Hungry Jacks', 'OnLine Centre Pty Ltd', 'CHAM O ER E OF COMMERCE', 'ORLANDO RISORT', 'Anto RIto', 'MURPHY & CO', 'GALBRAITHS ALE HOUSE', 'The Smith', 'Destination Talent Pty Ltd', 'RESTAURANT AREPERA DU', 'Kombl Coffee', 'rapalloav', 'Rapallo', 'HYATT REGENCY SAN FRANCISCO', 'ThuNE RBIRE', 'CAFE ROUGE', 'THE SHOT', 'Onward Travel Solutions Ltd', 'Hartnells Cafe', 'Food Truck Garage', 'TE PAPA ESPRESSO',  'TESCO, express', 'New Zealand Office Supplies', 'GRAND HYATT NEW YORK', 'CoPrTor Tax', 'Pret A Manger', 'Moore WIlson & Co Ltd', 'Creative Edge Food Co Ltd', 'JetBrains', 'Delhi Streets', 'delhistreets', 'BUZZ A BASKET', 'Wotif travel', 'wotif.travel', 'Wotif', 'NW SERVICES', 'M&S, EST.', 'marksandspencer', 'The Knights Templar', 'Femtuol LAnLLOu', 'Nandos', 'Phillips Seafood' , 'missionbay oufe', 'I. Store Computers', 'KINGS, PLANT BARN', 'ROGERS.', 'LITTLE STAR LYNYARD', 'United Parcel Service - FLIWAY (NZ) LIMITED', 'Exrin', 'CAFE BRETON', 'Copiesplus', 'copiesplus', 'DC Tech', 'CO-0P CA8S', 'Black Cottage Cafe', 'CMT UK LTD', 'ATLANTIC TAXI', 'ATLANTIC TAXI', 'The Setup on Manners', 'setupmanners', 'hmshost', 'HMS, C, H S T', 'Boulevard Brasserie', 'BOULEVARD, Brasserie', 'SSP Emirates LLC', 'Pro Go Palnts Ltd.', 'ovolohotels', 'homed', 'Sheraton Salt Lake City Hotel', 'Oresero Nalac', 'Sydney Madang Restaurant', 'Ticketmaster', 'ticketmaster', 'Bell Queen and Lee', 'STAPLES Canada', 'Amazon.ca', 'SSP AUSTRALIA CATER', 'Pizza Hut', 'LOO TAXI', 'QUICK FIX SHOE REPAI', 'KFC NZ', 'cVS/pharmacy', 'Ideal Auckland Downtow', 'IDEAL DOWNTOWN', 'Coffee Culture ', 'MD Digital Mobility', 'MAROO FOOD SERUICES', 'That Expross', 'Edwardian Bloomsbury Street Limited', 'JB HI-FI', 'Dorval Q0 H4Y', 'Dunkin Donuts', 'AIRLINE TAXI AND VAN', 'THE, CITTIE OF YORKE', 'per-tutti', 'Bar Bellaccino', 'j2 Global ANZ Limited', 'MAPLE LEAF TAXI', 'ELEMENTS CAFE', 'WOW Wholesale', 'airbnb', 'Airbnb', 'Waitamata Clay Target Club Inc', 'Benjamin TISSOT', 'Rendezvous Hotel Melbourne', 'Under Armour', 'In Vogue Blooms', 'Leon, The Strand', 'GAUCHO CHANCERY LANE', 'gauchorestaurants', 'Oaken', 'aswines', 'Snapper Rock Wines L.P.', 'invogueblooms', 'PARROT HOUSE RESTAUR', 'MOTOKA RENTALS LTD', 'DLL, IMACHOTEI', 'Taylor St Baristas', "Fratelli Fresh", 'ProductPlan', 'Lewis Oxford St', 'mario Dorkey', 'Airport. Express.' , 'The Big Easy Covent Garde', 'rowefarms', 'RoweFarms', 'jumpwithus', 'stanfordcourt', 'The Stanford Court San Francisco', 'Woodside Cafe', 'JUMP Avondale', 'Jaycar', 'Jaycor', 'mario Dorkey', 'Marlow Donkey', 'OMNI HOTELs S RESORTS', 'Sushi Hon', 'BARRIO CERVECERIA', 'UNIVERSAL PHONE REPAIR', 'ARDNET', 'Food Street', 'Coursera', 'CAFE BROWN SUGAR', 'MyGet Enterprse', 'JB HI-F|', 'TOWN CRIER PUB-HALFWAY', 'ASB Bank Limited', 'GWR', 'LOST IN THE TIME MINI GOLE', 'SUSHI & GRILL', 'SHERLOCK HOLMES', '1947 Eatery', 'AIRFLIGHT SERVICES', 'LOUIS SERGEANT SWEET COUTURE', 'ATS 266', 'Safari Hospitality Ltd', 'Botswana Butchery', 'Fenice Restaurant', 'Cogglelt Ltd', 'Coggle', 'coggle', 'ARMS F&B', 'arms-fnb', 'FRESCO', 'First Data Merchant Solutions', 'First Data, Merchant Solutions', 'mvauron', 'L Atelier du Fromage', 'SIGNCRAFT', 'Signcraft', 'Welcome Eatery', 'AIR NEW ZEALAND', 'Hello IT', 'SSP Australia Catering Pty Ltd', 'Elive Limited', 'Pandoro Panetteria', 'AL ALIMO', 'F E F INT I T', 'GIRAFFE.', 'SPIZE', 'Onni Tech', 'omni itech', 'The Halrus Cafe', 'Salt Lick BBQ', 'wagamama', 'Bier Markt', 'Arbory', 'Caffe *:k Cino', 'PATRIA', 'Front Siret LCNO', 'THE BRESLIN BAR & GR', 'TBREOLIN, BAR de GRILL', 'Microsoft Azure', 'CMMI Institute', 'CMMIInstitute', 'Yellow Card Srvs', 'Yellow Cab Coop', 'COLLETTES', 'THE PICKWICK HOTEL', 'IN A RUSH KITCHEN', 'BSAT', 'The Palm Beach Pad Palm Beach Holiday Apartment', 'Bachcare', 'The Yakitor) House', 'Stripe. Inc (US)', 'Cloud4J, Inc.', 'Flower Child', 'ICE SYDNEY PTY LIMITED', 'United Airlines', 'Inflight Wi-Fi', 'Pilot Coffee Downtown (FCP)', 'MT EDEN CONVENIENCE', 'Mt Eden Convenience', 'MEDALLIO', 'Micros Demo System', 'MALABAR JUNCTION', 'SHUJI SUSHI', 'FERRY CAFE', 'AUSTRALIA, POST', 'auspost', 'AUSTRALIA POST', 'COLUMBUS COFFEE', 'KMART', 'THE BLOOMSBURY', 'CAFE TRENDS', 'Chotto Matte', 'Subway', 'Linkedin Singapore Pte. Ltd', 'LIMO/TAX SERVICE', 'Home Brew Shop NZ', 'RENDEZVOUS, HOTE', 'Cloudthief PTY LTD.', 'MOjo Coffee', 'hellolindochinekitchen', 'indochinekitchen', 'Indochine', 'Spicy Korea', 'Officeworks', 'Officeworks Ltd', 'officeworks', 'OFficeworks', 'SHAKE SHACK', 'WissWlis', 'williamswarn', 'OAKS ON COLLINS', 'Starbucks Farnborough', 'telerik', 'Telerik Inc.','GRAND HOTEL MELBOURNE', 'FRAEDON COMPANY', 'FLYAWAY', '2 Cheap Botany', 'Humnmingbird Eatery & Bar', 'hummingbird', 'bettys', 'Baggins Shoes','Machina', 'intercontinental', 'segafredo','InVisionApp, Inc.', 'countdown', 'browserstack', 'Digital River Ireland Ltd.', 'Pragmatic', 'doppioespresso', 'doppio, ESPRESSO','London Underground', 'TOPGOLF', 'lookat', 'urbansoul', 'radacad','CASTAWAYS LODGE LTD','urbangourmet','netregistry', 'STARBUCKS','footprint-cars', 'LUGGAGECITY','BURGER LIQUOR', 'xamarin', 'Xamarin', 'culprit','highlanderbar','Rocket Print Ltd'] # 1318

phrases = ['MyOaks', 'MYOAKS', 'Westin Austin', 'GORDON HARRIS', 'Gordon Harris', 'Nandos', 'NANDOS NORTHBANK', 'km-taxi', 'mwave', 'Mwave', 'THE L.OOP, DUTY FREE', 'Adelong Electronics', 'Subway', 'subway', ' Amazon Web Services, Inc.', 'DINING MCKENNA', 'Fraedom', 'COREL', 'fraedom', 'DHL Express (New Zealand) Limited', 'vodafone', 'Vodafone Queen St', 'Onward Travel Solutions Ltd', 'New Zealand Sugar Company', 'SUNCORP', 'Paddle', 'fraedom', 'JAZZY & JIM PTY. LTD', 'ANZ', 'KAWAU BOATING CLUB', 'LADY MARMALADE', 'AUSTRALIAN VISA APPLICATION CENTRE', 'TT SERVICES NEW ZEALAND', 'The Unbakery', 'ROYAL OAK SHOE REPAI', 'Giles','LuckY Buddha', 'MAMAN', 'Fukuko', 'magshop', 'FRAEDOM', ' Bauer Media Pty Ltd.', 'Beach Central Cafe', 'CHINA CITY RESTAURAN', 'CX Computer Superstore', 'Spendvision', 'Tiggee, LLC', 'Auckland Transport', 'THE SHELF', 'POPUP CAFE', 'One Up Restaurant', 'Auckland Transport', 'THE KIWI ESCAPE GAME', 'OYSTER PLACE CHOP', 'warehouse, stationery', 'Warehouse Stationery Ltd', 'WORKhealth Limited', 'Fraedom Company Ltd', 'AIR NEW ZEALAND', 'Air New Zealand', 'airnewzealand', 'Grand Hyatt', 'GRAND, HYATT', 'Meatball & Wine Bar', 'ARVEEN KUMAR TAXI', 'ARRY PTY LTD', 'ANZ', 'Metlink Well', 'Aria Resort', 'Kiwi Souvenirs', 'Century Oaks', 'NOEL EXPRESS INC', 'The Lula Inn', 'CononwealthBank', 'NICKS SEAFOOD BAR &, GRILL', 'Dogpatch', 'Dantes Pizzeria', 'Wild Poppies Limited', 'Gourmet Curry Hut', 'Eclipse Cafe', 'Mrsd', 'HMS', 'LAS BANDERAS', 'Tret 1 A Mamser', 'MeDonald', 'Z Beach Road', 'yourwebsite', 'Ortolana & Milse', 'CafeAR', 'Mantra On Kent', 'Max on Hardware', 'VEGAN FOR L & H', 'New Jersey EWR Taxi', 'Trello', 'trello', 'Google Play', 'Giapo', 'giapo', 'ANZ', 'ISTRY SHOE REPAIRS', 'Beck Taxi', 'BECK TAX', 'MARRIOTT MARQUIS', 'MARRIOTT', 'Marriott', 'OSIB 50th Street Operator LLC', 'fraedom', 'Australian Government', 'Boulcott Street Bistro', 'boulcottstreetbistro', 'CARLSON WAGONLIT', 'Carlson Wagonlit Travel', 'SUBWAY', 'Carlson, Wagonlit', 'DOCKSIDE ESTAURANT', 'Cable Bay Vineyards', 'Flywheel Taxi', 'Retail NZ Limited', 'White WOINGS', 'Amalfi Pizza', 'ibis', 'lbis', 'accor', 'SAPORO','Liberty Apartment Hotel', 'Yogij Food Mart', 'tollroad', 'Zambrero','Featherstone Zambrero', 'Toto Pizza', 'IMMIPRO LTD', 'CHAYA TB', 'MACS', 'Too Fat Buns', 'The Paddock', 'ROLLING STONE T7', 'Starbuck', 'East Day Spa', 'fraedom', 'Modulis.ca Inc', 'modulis', 'Elliott Stable', 'Apple Pty Ltd', 'apple', 'AKREMTAX', 'Norwestin SOBER Drugs Inc', 'Pops for Champagne', 'mobile-360', 'Mobile-360', 'Amazon Export Sales LLC', 'Hilton', 'HILTON ATLANTA', 'NZ POST LTD', 'New Zealand Post', 'nzpost', 'AIR NEW ZEALAND', 'Air New Zealand', 'Marua Rd Cafe', 'PAZZI PER LA PIZZA', 'CAFFE CORTO', 'PB Tech', 'PB Technologies Ltd', 'pbtech', 'Fraedom Company Ltd', 'HEADOUARTERS', 'Visa4UK', 'Microsoft', 'spendvision', 'microsoft', 'xbox', 'COMMONWEALTH BANK', 'Warehouse Stationery', 'JUMP RESTAURANT', 'urger Burger & Fish Fish', 'burgerburger', 'ICABS', 'Fiori Cafe', 'Amazon.co.uk', 'Coggle', 'coggle', 'Meridian Pty Ltd', 'fraedom', 'NEW WORLD', 'Kreem Cafe Mopule', 'TE ARO COFFEE', 'Future of the Future', 'CGGLEIT LIMITED', 'patara', 'URGE COFFEE 8 TEA BOUTIQUE', 'STARBUCKS', 'RIFTYFIVE', 'facesandnames', 'OLA', 'PAPPARICH', 'SOUL, BAR & BISTRO', 'soulbar', 'All-Door Solutions', 'Kitchen by Mike', 'Sals', 'elive', 'Elive Limited', 'BIKANERVALA', 'Expert Infotech', 'Expert Electronics Ltd', 'The Coffee Club', 'digicert', 'Digicert, Inc.', 'digicert', 'THE TRUST', 'Bloomsbury Publishing', 'bloomsbury', 'VIRGIN AUSTRALIA', 'Esquires', 'Doist Ltd.', 'Smokes Poutinerie', 'The Lunchroom', 'Eggshel Cafe', 'SMOKESPOUTINERIE', 'EASTRIDGE, FLOWERS', 'eastridgeflowers', 'Austin Cafe', 'Leuven Belgian Beer Cafe', 'Tech4U', 'thelobbylounge', 'Lobby lounge', 'COFFEE, THEADEMICS', 'WHITE FWONGS', 'Maritime Hospitality Limited']

# we add the merchant name list to a spacy phrase matcher object to construct the merchant phrase dictionary.
[merchantMatcher.add("merchant", on_match, nlp(i)) for i in phrases]

res = []
to_train_ents = []

# now we loop through the text found at each invoice data and use the above-mentioned matchers to identify matching entity tokens with its start and end index in each invoice text and corresponding (matching) entity label (GST, AMOUNT, MERCHANT etc.) 
with open('formated_textract.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    header = next(csv_reader)
    fieldnames = ['text', 'entities']
    with open('annotatedData_test.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, quotechar='"',quoting=csv.QUOTE_NONNUMERIC,)
        writer.writeheader()
    # Iterate over each row in the csv using reader object
        for row in itertools.islice(csv_reader, 1400, 1602):
            # row variable is a list that represents a row in csv
            mnlp_line = nlp(row[0])
            dateMatches = dateMatcher(mnlp_line)
            timeMatches = timeMatcher(mnlp_line)
            gstMatches = gstMatcher(mnlp_line)
            amountMatches = amountMatcher(mnlp_line)
            phraseMatches = merchantMatcher(mnlp_line)
            dateRES = [offsetterT("DATE", mnlp_line, x) for x in dateMatches]
            timeRES = [offsetterT("TIME", mnlp_line, x) for x in timeMatches]
            gstRES = [offsetterT("GST", mnlp_line, x) for x in gstMatches]
            amountRES = [offsetterT("AMOUNT", mnlp_line, x) for x in amountMatches]
            phraseRES = [offsetterT("MERCHANT", mnlp_line, x) for x in phraseMatches]
            res = dateRES + timeRES + gstRES + amountRES + phraseRES
            entity = {"entities": res}
            writer.writerow({'text': str(row)[2:-2], 'entities': str(entity)})
            #to_train_ents.append((str(row), dict(entities=res)))


# optimizer = nlp.begin_training()
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
# with nlp.disable_pipes(*other_pipes):
#     for itn in range(20):
#         losses = {}
#         random.shuffle(to_train_ents)
#         for item in to_train_ents:
#             nlp.update([item[0]], [item[1]], sgd= optimizer, drop=0.35, losses = losses)
sys.stdout.close()
#ner = EntityRecognizer(nlp.vocab)
