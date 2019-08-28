import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    replace_by_blank_symbols = re.compile('\u00bb|\u00a0|\u00d7|\u00a3|\u00eb|\u00fb|\u00fb|\u00f4|\u00c7|\u00ab|\u00a0\ude4c|\udf99|\udfc1|\ude1b|\ude22|\u200b|\u2b07|\uddd0|\ude02|\ud83d|\u2026|\u201c|\udfe2|\u2018|\ude2a|\ud83c|\u2018|\u201d|\u201c|\udc69|\udc97|\ud83e|\udd18|\udffb|\ude2d|\udc80|\ud83e|\udd2a|\ud83e|\udd26|\u200d|\u2642|\ufe0f|\u25b7|\u25c1|\ud83e|\udd26|\udffd|\u200d|\u2642|\ufe0f|\udd21|\ude12|\ud83e|\udd14|\ude03|\ude03|\ude03|\ude1c|\udd81|\ude03|\ude10|\u2728|\udf7f|\ude48|\udc4d|\udffb|\udc47|\ude11|\udd26|\udffe|\u200d|\u2642|\ufe0f|\udd37|\ude44|\udffb|\u200d|\u2640|\udd23|\u2764|\ufe0f|\udc93|\udffc|\u2800|\u275b|\u275c|\udd37|\udffd|\u200d|\u2640|\ufe0f|\u2764|\ude48|\u2728|\ude05|\udc40|\udf8a|\u203c|\u266a|\u203c|\u2744|\u2665|\u23f0|\udea2|\u26a1|\u2022|\u25e1|\uff3f|\u2665|\u270b|\u270a|\udca6|\u203c|\u270c|\u270b|\u270a|\ude14|\u263a|\udf08|\u2753|\udd28|\u20ac|\u266b|\ude35|\ude1a|\u2622|\u263a|\ude09|\udd20|\udd15|\ude08|\udd2c|\ude21|\ude2b|\ude18|\udd25|\udc83|\ude24|\udc3e|\udd95|\udc96|\ude0f|\udc46|\udc4a|\udc7b|\udca8|\udec5|\udca8|\udd94|\ude08|\udca3|\ude2b|\ude24|\ude23|\ude16|\udd8d|\ude06|\ude09|\udd2b|\ude00|\udd95|\ude0d|\udc9e|\udca9|\udf33|\udc0b|\ude21|\udde3|\ude37|\udd2c|\ude21|\ude09|\ude39|\ude42|\ude41|\udc96|\udd24|\udf4f|\ude2b|\ude4a|\udf69|\udd2e|\ude09|\ude01|\udcf7|\ude2f|\ude21|\ude28|\ude43|\udc4a|\uddfa|\uddf2|\udc4a|\ude95|\ude0d|\udf39|\udded|\uddf7|\udded|\udd2c|\udd4a|\udc48|\udc42|\udc41|\udc43|\udc4c|\udd11|\ude0f|\ude29|\ude15|\ude18|\ude01|\udd2d|\ude43|\udd1d|\ude2e|\ude29|\ude00|\ude1f|\udd71|\uddf8|\ude20|\udc4a|\udeab|\udd19|\ude29|\udd42|\udc4a|\udc96|\ude08|\ude0d|\udc43|\udff3|\udc13|\ude0f|\udc4f|\udff9|\udd1d|\udc4a|\udc95|\udcaf|\udd12|\udd95|\udd38|\ude01|\ude2c|\udc49|\ude01|\udf89|\udc36|\ude0f|\udfff|\udd29|\udc4f|\ude0a|\ude1e|\udd2d|\uff46|\uff41|\uff54|\uff45|\uffe3|\u300a|\u300b|\u2708|\u2044|\u25d5|\u273f|\udc8b|\udc8d|\udc51|\udd8b|\udd54|\udc81|\udd80|\uded1|\udd27|\udc4b|\udc8b|\udc51|\udd90|\ude0e')
    replace_by_apostrophe_symbol = re.compile('\u2019')
    replace_by_dash_symbol = re.compile('\u2014')
    replace_by_u_symbols = re.compile('\u00fb|\u00f9')
    replace_by_a_symbols = re.compile('\u00e2|\u00e0') 
    replace_by_c_symbols = re.compile('\u00e7') 
    replace_by_i_symbols = re.compile('\u00ee|\u00ef') 
    replace_by_o_symbols = re.compile('\u00f4') 
    replace_by_oe_symbols = re.compile('\u0153')
    replace_by_e_symbols = re.compile('\u00e9|\u00ea|\u0117|\u00e8')
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|,;]')
    text = replace_by_e_symbols.sub('e', text)
    text = replace_by_a_symbols.sub('a', text)
    text = replace_by_o_symbols.sub('o', text)
    text = replace_by_oe_symbols.sub('oe', text)
    text = replace_by_u_symbols.sub('e', text)
    text = replace_by_i_symbols.sub('e', text)
    text = replace_by_u_symbols.sub('e', text)
    text = replace_by_apostrophe_symbol.sub("'", text)
    text = replace_by_dash_symbol.sub("_", text)
    text = replace_by_blank_symbols.sub('', text)

    #For English
    #text = ''.join([c for c in text if ord(c) < 128])
    text = text.replace("\\", "")
    #text = text.encode("ascii", errors="ignore").decode()
    text = text.lower() # lowercase text
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

