# this file loop through the textract metadata, and extract only the text data of it. 
# in the end you would have a csv file with two columns: invoice id and text content of each invoice. 
import json
import ndjson
import csv
def SavingText(response):
    text = []
    for block in response['blocks']:
        print('Id: {}'.format(block['id']))
        if block['blocktype'] == 'LINE':
            if len(text) > 0 :
                print('Print text \n')
                print('    Detected: ' + str(text))
                text = []
            print('    Detected: ' + block['text'])
        if block['blocktype'] == 'WORD':
            text.append(block['text'])
        print('    Type: ' + block['blocktype'])
    print('----------------------------------------------')

def main():
    with open(r'C:\Users\spenc\OneDrive - The University of Auckland\Study\UoA\Project\OCR\ocr-metadata.ndjson') as f:
        data = ndjson.load(f)
    type(data)
    jsonFormat = json.loads(json.dumps(data))
    type(jsonFormat)
    fieldnames = ['text']
    with open('formated_textract.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, quotechar='"',quoting=csv.QUOTE_NONNUMERIC,)
            writer.writeheader()
            for item in jsonFormat:
                line = {}
                #print(item['correlationid'])
                textractResponse = json.loads(item['textractResponse'])
                textValue = []
                for block in textractResponse['blocks']:
                    if block['blocktype'] == 'LINE':
                        if 'text' in block:
                            #print(block['text'])
                            textValue.append(block['text'])
                line['text'] = str(textValue)[1:-1].replace("'","").replace('"',"")
                writer.writerow(line)
main()