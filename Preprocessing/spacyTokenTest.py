# this is a test file, to test how spacy tokeniser performs 
import json,ndjson, spacy
with open(r'C:\Users\spenc\OneDrive - The University of Auckland\Study\UoA\Project\OCR - Sample of the Sample\OCR - Sample of the Sample\ocr-metadata.ndjson') as f:
        data = ndjson.load(f)
jsonFormat = json.loads(json.dumps(data))
textRact = json.loads(jsonFormat[0]['textractResponse'])
text = []
for block in textRact['blocks']:
    if block['blocktype'] == 'WORD':
        #print(block['text'])
        text.append(block['text'])
text = str(text).replace("'", '')[1:-1]
print(text)

nlp = spacy.load("en_core_web_lg")
doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)


    