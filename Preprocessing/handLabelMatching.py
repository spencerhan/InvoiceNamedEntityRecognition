# this is a testing python file, used to manually check the accuracy of spacy string matcher and regular expression matcher. 
import itertools
import itertools
import csv
import spacy

import re # 1139 - 1302
word = r'BADGE KING, 3A/45 THE CONCOUR, AUCKLAND, EFTPOS 78120701, TERMINAL, TIME, 14JUN 12:52, TRAN 000047, CREDI, UISA, M, .8599, AUTH 158104, PURCHASE, N2$101.77, TOTAL, NZ$101.77, ACCEPT WITH SIG, CUSTOMER COP5'

text = r'NZ\$101.77'

print(len(text))

matches = re.finditer(text, word)
s = ['('+ str(match.start()) + ', '+ str(match.start() + len(text)-1) + ", 'DATE')"  for match in matches]
print(str(s).replace('"',''))

