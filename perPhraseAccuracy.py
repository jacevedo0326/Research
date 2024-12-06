import json
from tqdm import tqdm
import argparse
import os
ListOfFolders = ['gptResponsesP01_R02.json', 'gptResponsesP02_R01.json', 'gptResponsesP02_R02.json', 'gptResponsesP04_R01.json', 
                 'gptResponsesP05_R01.json', 'gptResponsesP05_R02.json', 'gptResponsesP08_R01.json', 'gptResponsesP08_R03.json', 
                 'gptResponsesP09_R02.json', 'gptResponsesP11_R01.json', 'gptResponsesP14_R02.json', 'gptResponsesP16_R01.json']

def compare_actions(data):
    total_correct_actions = 0
    total_correct_meta_actions = 0
    total_actions = 0

    for file_name, action_data in data.items():
        predicted_action = action_data["Action"]
        predicted_meta_action = action_data["Meta-Action"]
        correct_action = action_data["Correct Action"]
        correct_meta_action = action_data["Correct Meta-Action"]

        total_actions += 1

        if predicted_action == correct_action:
            total_correct_actions += 1
        if predicted_meta_action == correct_meta_action:
            total_correct_meta_actions += 1

    action_accuracy = (total_correct_actions / total_actions) * 100
    meta_action_accuracy = (total_correct_meta_actions / total_actions) * 100

    return round(action_accuracy, 3), round(meta_action_accuracy, 3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpt_path', type= str, help= 'Path to the folder which contains gpt responses')

    args = parser.parse_args()
    ListOfFolders = [os.path.join(args.gpt_path, x) for x in os.listdir(args.gpt_path)]
    for folderName in tqdm(ListOfFolders):
        with open(folderName, "r") as f:
            data = json.load(f)
        actionAccuracy, metaActionAccuracy = compare_actions(data)
        print(f'File: {folderName}')
        print(f'Action Accuracy: {actionAccuracy}%')
        print(f'Meta-Action Accuracy: {metaActionAccuracy}%')