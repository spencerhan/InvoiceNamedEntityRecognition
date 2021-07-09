# this file extract the confidence score from the original textract metadata. 
# it works similar to the textractProcessing file, however the output now has three columns, with an addtional column indicates
# the overall textract confidence score for each invoice. 
import json
import ndjson
import csv
import statistics
def SavingText(response):
    # this section loop through the textract metadata, and spits out the confidence score at each line node, 
    # it combines all text found at the WORD node to the final text output (each text output represents all text found on one invoice)
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
    fieldnames = ['text','overall_confidence']
    # this section outputs all text found on each invoice entry, and overall confidence score averaged (based on number of WORD node).
    with open('formated_textract_confidence.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, quotechar='"',quoting=csv.QUOTE_NONNUMERIC,)
            writer.writeheader()
            for item in jsonFormat:
                line = {}
                #print(item['correlationid'])
                textractResponse = json.loads(item['textractResponse'])
                textValue = []
                confidenceValue = []
                for block in textractResponse['blocks']:
                    if block['blocktype'] == 'LINE':
                        if 'text' in block:
                            #print(block['text'])
                            textValue.append(block['text'])
                            # confidence scores of all WORD node for each inovoice are averaged.
                            confidenceValue.append(float(block['confidence']))           
                overallConfidence = statistics.mean(confidenceValue)
                # this removes the quotation marks at the beginning and end of each text.  
                line['text'] = str(textValue)[1:-1].replace("'","").replace('"',"")
                line['overall_confidence'] = overallConfidence
                writer.writerow(line)
main()