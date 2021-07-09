import pandas as pd
import ast
path_toData = r'‪C:\Users\spenc\OneDrive - The University of Auckland\Study\UoA\Project\matching_new.csv'
data = pd.read_csv(path_toData.strip("‪u202a"),header=0, nrows=1200)
data.head(n=5)
#LowerCase
seriesLower = data['text'].apply(lambda x: x.lower())
entities = data['entities'].apply(lambda x: ast.literal_eval(x))
dataLower = seriesLower.to_frame().join(entities)
dataLower.head(n=5)
#entity setting
import spacy 
sp = spacy.load('en_core_web_lg')
dataToken = dataLower['text'].apply(lambda x: sp(x))
dataEntities = dataToken.to_frame().join(dataLower['entities'])
dataEntities.head(n=5)
def entitySetter(_token, _inputDataSeries):
    for entity in _inputDataSeries['entities']:
        if _token.idx >= int(entity[0]) and _token.idx < int(entity[1]):
            if _token.idx == int(entity[0]):
                _token.ent_type_ = "B-" + entity[2]
            else:
                _token.ent_type_ = "I-" + entity[2]
            break
        else:
            _token.ent_type_ = 'O'
    return _token
dataEntities.iloc[0]
len(dataEntities.index)
i = 0
while i < len(dataEntities.index):
    for token in dataEntities.iloc[i]['text']:
        token = entitySetter(token, dataEntities.iloc[i]['entities'])
    i = i + 1
print(len(dataEntities.iloc[1]['text']))
preproccessDf = pd.DataFrame(columns = ['invoice_id', 'token', 'POS', 'tag'])
i = 1109
while i < len(dataEntities.index):
    j = 0
    while j < len(dataEntities.iloc[1]['text']):
        print(dataEntities.iloc[1]['text'][j].text)
        j = j+1
    i = i +1