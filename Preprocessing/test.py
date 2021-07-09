import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
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
patterns =  [pattern1, pattern2, pattern3, pattern4, pattern5]

matcher = Matcher(nlp.vocab)
def on_match(matcher, doc, id, matches):
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        print(match_id, string_id, start, end, span.text)

# def offsetterP(lbl, doc, matchitem):
#     print(doc[0:matchitem[1]])
#     o_one = len(str(doc[0:matchitem[1]]))
#     subdoc = doc[matchitem[1]:matchitem[2]]
#     print(subdoc)
#     o_two = o_one + len(str(subdoc))
#     print(len(str(subdoc)))
#     return (o_one, o_two, lbl)

def offsetterT(lbl, doc, matchitem):
    span = str(doc[matchitem[1]:matchitem[2]])
    result = str(doc).find(span,len(str(doc[0:matchitem[1]]))) 
    o_two = result + len(span)
    print(len(span))
    return (result, o_two, lbl)


for i in patterns:
    matcher.add("DATE", on_match, i)
doc = nlp("21:8157168, AUCKI AND, EFTPOS, TERMINAL, 64732402, TIME, 05NOV15 12 :- 24, TRAN 016047, CREDIT, VISA, CARD, 0742, CONTACTLESS, Visa, RID: A000000003, PIX: 1010, AAC:4C25A1523F996391, ATC: 0060, AUTH 536635, PURCHASE, NZ$46.00, TOTAL, NZ$46.00, ACCEPTED, CUSTOMER COPY")

matches = matcher(doc)
res = [offsetterT("amount", doc, x) for x in matches]
print('test')
