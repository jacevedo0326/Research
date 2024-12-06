import tensorflow as tf
import keras_nlp
import json
from generators import tokenizer
from Prepared_Data.dataset_params import BERT_MODEL, SEQ_LEN
from tqdm import tqdm 
import os
import argparse
# accuracy_base = tf.keras.metrics.CategoricalAccuracy()

# limit = 3  # How many files to process
# file_path = '/mnt/c/Users/Joshua/Dev/ICRA2024/gptResponsesP04_R01.json' # File path to json file being processed
listOfMetaActionRecovered = []
listOfActionRecovered = []
metaActionTokensTensor= []
correctMetaActionTokensTensor=[]
# ListOfFolders = ['gptResponsesP01_R02.json', 'gptResponsesP02_R01.json', 'gptResponsesP02_R02.json', 'gptResponsesP04_R01.json', 
#                  'gptResponsesP05_R01.json', 'gptResponsesP05_R02.json', 'gptResponsesP08_R01.json', 'gptResponsesP08_R03.json', 
#                  'gptResponsesP09_R02.json', 'gptResponsesP11_R01.json', 'gptResponsesP14_R02.json', 'gptResponsesP16_R01.json' ]
parser = argparse.ArgumentParser()

parser.add_argument('-gpt_path', type= str, help= 'Path to the folder which contains gpt responses')

args = parser.parse_args()
ListOfFolders = [os.path.join(args.gpt_path, x) for x in os.listdir(args.gpt_path)]
dictOfLosses = {}
def true_recovery(y_true, y_pred):
    print(f'ytrue is {y_true}')
    print(f'ypred is {y_pred}')
    correctMetaActionsPure = []
    metaActionsPure = [] 
    trueOrFalse = []
    trueCount = 0
    falseCount = 0
    # indicies101 = tf.where(y_true == 101)
    # indicies102 = tf.where(y_true == 102)
    # indicies0 = tf.where(y_true == 0)
    # selected_indicies = tf.logical_not(tf.logical_or(indicies0, tf.logical_or(indicies101, indicies102)))
    # correctMetaActionsPure = y_true[selected_indicies]
    # metaActionsPure = y_pred[selected_indicies]
    for index, _ in enumerate(y_true):
        if y_true[index] == 101 or y_true[index] == 102:
            continue
        elif y_true[index] == 0:
            continue
        else:
            correctMetaActionsPure.append(y_true[index].numpy())
            metaActionsPure.append(y_pred[index].numpy())
    
    # for item in y_pred:
    #     if item == 101:
    #         continue
    #     else:
    #         metaActionsPure.append(item.numpy())
    
    print(f'CorrectMetaActionPureList: {correctMetaActionsPure}')
    print(f'MetaActionPureList: {metaActionsPure}')
    for k in range(len(correctMetaActionsPure)):
        if correctMetaActionsPure[k] == metaActionsPure[k]:
            trueOrFalse.append('True')
        else:
            trueOrFalse.append('False')
    print(f'recoveredList: {trueOrFalse}')
    for item in trueOrFalse:
        if item == 'True':
            trueCount +=1
        if item == 'False':
            falseCount +=1
    recovered = trueCount / (trueCount + falseCount)
    return recovered
# Why doesnt this set of functions work
# def true_recovery(y_true, y_pred):
#     print(f'ypred is: {y_pred}')
#     print(f'ytrue is: {y_true}')
#     correctMetaActionsPure = []
#     MetaActionsPure = [] 
#     for item in y_true:
#         if item == 101 or item == 102:
#             continue
#         elif item == 0: #Is there any token which is 0?
#             continue
#         else:
#             correctMetaActionsPure.append(item)
#     for item in y_pred:
#         if item == 101 or item == 102:
#             continue
#         elif item == 0:
#             continue
#         else:
#             MetaActionsPure.append(item)
#     print(f'CorrectMetaActionPureList: {correctMetaActionsPure}')
#     print(f'MetaActionPureList: {MetaActionsPure}')

def string_to_bert(input_string):
    preprocessor = keras_nlp.models.BertPreprocessor(tokenizer=tokenizer, sequence_length=SEQ_LEN)
    
    bert_tokens = preprocessor(input_string)

    return bert_tokens

def process_json_file():
    # Load the JSON file
    
    

    for folder in ListOfFolders:
        file_path = f'/mnt/c/Users/Joshua/Dev/ICRA2024/{folder}'
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create a tqdm progress bar
        for key, value in tqdm(list(data.items()), desc="Processing files"):
            # Extract Meta-Action
            metaAction = value.get('Meta-Action', 'Not found')
            correctMetaAction = value.get('Correct Meta-Action', 'Not found')
            
            # Extract Action 
            action = value.get('Action', 'Not found')
            correctAction = value.get('Correct Action', 'Not found')
            
            # # expand the action and meta-actions
            # metaActionExpanded = tf.expand_dims(tf.constant(metaAction), 0)
            # correctMetaActionExpanded = tf.expand_dims(tf.constant(correctMetaAction), 0)
            # actionExpanded = tf.expand_dims(tf.constant(action), 0)
            # correctActionExpanded = tf.expand_dims(tf.constant(correctAction), 0)
            
            # Convert the input string to BERT embeddings
            metaActionTokens = string_to_bert(metaAction)
            correctMetaActionTokens = string_to_bert(correctMetaAction)
            actionTokens = string_to_bert(action)
            correctActionTokens = string_to_bert(correctAction)
            
                    
            # # Sees what data structure tokens are
            # print(f'metaActionTokens: {metaActionTokens}')
            # print(f'correctMetaActionTokens: {correctMetaActionTokens}')
            # print(f'actionTokens: {actionTokens}')
            # print(f'correctActionTokens: {correctActionTokens}')
            
            # print('-'*50 + 'look here' + '-'*50 )
            # print(list(metaActionTokens.values())[:4])
            # print('-'*50 + 'look here' + '-'*50 )
            # Here we convert list to tensor
            metaActionTokensTensor = metaActionTokens['token_ids']
            correctMetaActionTokensTensor = correctMetaActionTokens['token_ids']
            actionTokensTensor = actionTokens['token_ids']
            correctActionTokensTensor = correctActionTokens['token_ids']

            # # Sees what data structure tokensTensors are
            # print(f'metaActionTokensTensor: {metaActionTokensTensor}')
            # print(f'correctMetaActionTokensTensor: {correctMetaActionTokensTensor}')
            # print(f'actionTokensTensor: {actionTokensTensor}')
            # print(f'correctActionTokensTensor: {correctActionTokensTensor}')

            metaActionRecovered = true_recovery(correctMetaActionTokensTensor, metaActionTokensTensor)
            actionRecovered = true_recovery(correctActionTokensTensor, actionTokensTensor)
            
            listOfMetaActionRecovered.append(metaActionRecovered)
            listOfActionRecovered.append(actionRecovered)
            print('---')
        print(metaActionRecovered)
        print(actionRecovered)
        # Calculate and print mean losses
        finalMetaActionRecovered = sum(listOfMetaActionRecovered)/len(listOfMetaActionRecovered)
        finalActionRecovered = sum(listOfActionRecovered)/len(listOfActionRecovered)
        dictOfLosses.update({folder: {'Meta Action Recoved': finalMetaActionRecovered, 'Action Recovered': finalActionRecovered}})
        print(dictOfLosses)
        print(f'Folder {file_path} done being processed')
    return dictOfLosses
dictOfLosses = process_json_file()
print(f'dictOfLosses is {dictOfLosses}')
