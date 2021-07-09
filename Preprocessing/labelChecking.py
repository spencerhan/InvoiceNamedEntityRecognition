# this file is used to check whether there are typos in entity labels.
# it converts the json-like string in the entity column to python dictionary (therefore, typos will throw exception).
import pandas as pd
import ast
data = pd.read_csv('matching_new.csv', header=0)
print(data.head(n=5))
newCol = []
# ast.literal_eval evaluates the underlying string based on its form,  if it is in python dictionary form, 
# it will try to convert it to a python dictionary.

# this snippet is an experimental one
x = data["entities"].apply(lambda x: ast.literal_eval(x))
print(x)
print(type(x))
# apply the ast.literal_eval to the entire dataset.
for index, row in data.iterrows():
    rowDict = ast.literal_eval(row.entities)
    entities = [a for a in rowDict["entities"] if len(a) > 0]
    entityText = []
    if len(entities) > 0:
        [entityText.append('(' + row.text[a[0]:a[1]] + ' , ' + a[2] + ')')
         for a in entities]
    else:
        entityText.append([])
    newCol.append(entityText)
data['matching'] = newCol
data.to_csv('matching_new_2.csv', index=False)
print('')
