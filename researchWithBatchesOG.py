import os
import json
import tqdm
import random
import re
from openai import OpenAI
from openAIKey import API_KEY

client = OpenAI(api_key=API_KEY)

# Global variables
listOfBodyParts3D = ['hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
                     'Spine', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']
listofBodyParts2D = ['nose', 'leftEye', 'rightEye', 'LeftEar', 'rightEar', 'leftShoulder', 'leftElbow', 'leftWrist',
                     'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']
count = 1
currentAction = ""
currentMetaAction = ""
nextAction = ""
nextMetaAction = ""
numberOfRuns = 10  # Specify how many runs we are doing.
totalInputToken = 0
totalOutputToken = 0

def getDataFromJsonFiles(folderPath, fileName):
    global currentAction, currentMetaAction, nextAction, nextMetaAction

    with open(fileName, 'r', encoding='utf-8') as json_file:
        try:
            json_data = json.load(json_file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {fileName}")
            return None

        finalResult = ""
        finalResult3D = ""
        finalResult2D = ""

        # Access the values directly using the keys
        currentAction = json_data.get("current_task_str", "Key not found")
        currentMetaAction = json_data.get("current_task_meta_str", "Key not found")
        nextAction = json_data.get("next_task_str", "Key not found")
        nextMetaAction = json_data.get("next_task_meta_str", "Key not found")

        for key, extractedData in json_data.items():
            if "token" in key or "embed" in key:  # Skip if key contains 'token' or 'embed'
                continue
            if isinstance(extractedData, list):
                raw_data = f"{key}: {', '.join(map(str, extractedData))}"
            elif isinstance(extractedData, str):
                raw_data = f"{key}: {extractedData}"
            elif isinstance(extractedData, float):
                raw_data = f"{key}: {str(extractedData)}"

            if '3D_poses' in key:
                temp = '3D_poses: '
                for bodyPartNum, bodyPart in enumerate(listOfBodyParts3D):
                    partData = bodyPart + ' ['
                    tx = extractedData[bodyPartNum*6 + 0][1] if isinstance(extractedData[bodyPartNum*6 + 0], list) else extractedData[bodyPartNum*6 + 0]
                    partData += 'Tx=' + str(tx) + ', '
                    ty = extractedData[bodyPartNum*6 + 1][1] if isinstance(extractedData[bodyPartNum*6 + 1], list) else extractedData[bodyPartNum*6 + 1]
                    partData += 'Ty=' + str(ty) + ', '
                    tz = extractedData[bodyPartNum*6 + 2][1] if isinstance(extractedData[bodyPartNum*6 + 2], list) else extractedData[bodyPartNum*6 + 2]
                    partData += 'Tz=' + str(tz) + ', '
                    rx = extractedData[bodyPartNum*6 + 3][1] if isinstance(extractedData[bodyPartNum*6 + 3], list) else extractedData[bodyPartNum*6 + 3]
                    partData += 'Rx=' + str(rx) + ', '
                    ry = extractedData[bodyPartNum*6 + 4][1] if isinstance(extractedData[bodyPartNum*6 + 4], list) else extractedData[bodyPartNum*6 + 4]
                    partData += 'Ry=' + str(ry) + ', '
                    rz = extractedData[bodyPartNum*6 + 5][1] if isinstance(extractedData[bodyPartNum*6 + 5], list) else extractedData[bodyPartNum*6 + 5]
                    partData += 'Rz=' + str(rz) + ', '
                    partData += '], '
                    temp += partData
                finalResult3D += temp

            elif '2D' in key:
                finalResult2D += '2D_poses: '  # Print '2D_poses' only once before camera view point 0
                for cameraViewPoint in range(len(extractedData)):  # Takes into account how there are different camera angles
                    temp = f'camera view point: {cameraViewPoint} '
                    for bodyPartNum, bodyPart in enumerate(listofBodyParts2D): 
                        partData = bodyPart + ' ['
                        tx = extractedData[cameraViewPoint][bodyPartNum][1]
                        partData += 'Tx=' + str(tx) + ', '
                        ty = extractedData[cameraViewPoint][bodyPartNum][2]
                        partData += 'Ty=' + str(ty) + ', '
                        tconfidence = extractedData[cameraViewPoint][bodyPartNum][3]
                        partData += 'confidence = ' + str(tconfidence) + ', '
                        partData += '], '
                        temp += partData
                    finalResult2D += temp + ' '
                break  # Exit after processing the first camera viewpoint

        finalResult = finalResult3D + finalResult2D
        return finalResult

def processGptResponse(fileName, gptResponse, outputJsonPath):
    try:
        with open(outputJsonPath, 'r') as f:
            gptResponseDict = json.load(f)
    except FileNotFoundError:
        gptResponseDict = {}

    # Regex patterns
    metaActionPattern = r'Meta-Action:?\s*"?([^"\n]+)"?'
    actionPattern = r'Action:?\s*"?([^"\n]+)"?'

    # Find all matches
    metaActions = re.findall(metaActionPattern, gptResponse, re.IGNORECASE | re.MULTILINE)
    actions = re.findall(actionPattern, gptResponse, re.IGNORECASE | re.MULTILINE)

    fileDict = {}

    if metaActions:
        fileDict['Meta-Action'] = metaActions[-1].strip()  # Take the last match
    else:
        fileDict['Meta-Action'] = "No Meta-Action found"
    
    if actions:
        fileDict['Action'] = actions[-1].strip()  # Take the last match
    else:
        fileDict['Action'] = "No Action found"

    fileDict['Correct Meta-Action'] = nextMetaAction
    fileDict['Correct Action'] = nextAction

    gptResponseDict[fileName] = fileDict

    with open(outputJsonPath, 'w') as f:
        json.dump(gptResponseDict, f, indent=4)

    print(f"Processed and saved response for file: {fileName}")
    print(f"Meta-Action: {fileDict['Meta-Action']}")
    print(f"Action: {fileDict['Action']}")

if __name__ == "__main__":
    outputFilePath = r'E:\projects\Research\researchOutputWithBatches.txt'  # Specify the full path to the output file
    gptResponsesJsonPath = r'E:\projects\Research\gptResponsesWithBatches.json'  # New JSON file for GPT responses
    jsonlFilePath = r'E:\projects\Research\batch_requests.jsonl'  # New JSONL file for batch requests
    actionMetaActionFile = r'E:\projects\Research\Action-Meta-action-list.pdf' #The file we want to use to give openAI options

    # Clear both files at the start of the run
    with open(outputFilePath, 'w', encoding='utf-8') as f:
        pass
    with open(gptResponsesJsonPath, 'w', encoding='utf-8') as f:
        json.dump({}, f)
    with open(jsonlFilePath, 'w', encoding='utf-8') as jsonl_file:
        pass

    folderPath = r'E:\\Prepared_Data\\Prepared_Data\\training\\P01_R01'
    filesInFolder = [fileName for fileName in os.listdir(folderPath) if fileName.endswith(".json")]
    random.shuffle(filesInFolder)
    selectedFiles = filesInFolder[:numberOfRuns]

    print("Selected files for processing:")
    for idx, file in enumerate(selectedFiles):
        print(f"{idx + 1}. {file}")

    with open(outputFilePath, 'a', encoding='utf-8') as f:  # Open file in append mode for the loop
        for fileName in tqdm.tqdm(selectedFiles):
            wholeFileName = os.path.join(folderPath, fileName)
            print(f"\nStarting to process file: {wholeFileName}")

            if not os.path.exists(wholeFileName):
                print(f"File does not exist: {wholeFileName}")
                continue

            fileResult = getDataFromJsonFiles(folderPath, wholeFileName)
            if fileResult is None:
                continue

            # Write the file's result to the output text file
            f.write(f"\nProcessing file: {wholeFileName}\n")
            f.write(f"{fileResult}\n")
            f.write('-----------------Block ends here ------------------------\n')

            # Create the JSON object for the JSONL file
            jsonl_data = {
                "custom_id": fileName, 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content":"""
                        The user will give you data about a person's body parts and positioning, this data is given from a 3d persepctive 
                        where you recieve each body parts position in a x,y,z, format and their rotations in an x,y,z format.
                        You also get each body parts position in a 2d perspective from multiple camera angles in an x,y format and you get a confidence level. 
                        I then need you to tell the user what person observed is doing and predict what they are about to do. 
                        You are to match each action to a specific meta-action and action found in the file you were given. 
                        Take your time, there is absolutley no rush I want the answer to be correct not quick.
                        """},
                        {"role": "user", "content":"""
                        Here are the positions of the main joints and body parts of a person from a 3D and 2D perspective. 
                        They are labeled accordingly.
                        """},
                        {"role": "user", "content":"""
                        The format should be exactly as follows. Don't incllude (New line) but do have a line in between meta-action and Action
                        Meta-Action: (give the predicted next meta-action). (New line) 
                        Action: (give the predicted next action here). 
                        """},
                        {"role": "system", "content": fileResult},
                        {"role": "user", "content":"""
                        Do not output any inforation about the following threads. The reader doesn't need to know about the microsoft excell file.
                        """},
                        {"role": "user", "content":"""
                        The possible actions and meta actions which you are to pick from are given in the pdf file titled 'Action-Meta-action-list'. 
                        Do not tell ther reader about this. You also do not have to include the number in the most lefthand column. 
                        For example, if you pick taking components, it should be taking components, not taking components9.
                        The action also shouldn't include any numbers. Also don't reference your sources.
                        """},
                        {"role": "user", "content":"""

                        """},
                        {"role": "user", "content":"""
                            Please do not explain yourself. The only output you should give is the meta-action and action as stated above
                        """}
                    ]
                }
            }

            # Write the JSON object to the JSONL file
            with open(jsonlFilePath, 'a', encoding='utf-8') as jsonl_file:
                jsonl_file.write(json.dumps(jsonl_data) + '\n')

            print(f"Processed and saved JSONL entry for file: {fileName}")
    print("Uploading the file")
    #Here we upload the batch file to openai
    batchRequests = client.files.create(
        file = open("E:\\projects\\Research\\batchRequests.jsonl", "rb"),
        purpose= "batch"
    )
    
    batchRequestsInputFileId = batchRequests.id

    batch_requests = client.batches.create(
        input_file_id = batchRequestsInputFileId,
        endpoint = "/v1/chat/completions",
        completion_window = "24h",
        metadata = {
            "description": "Tool to predict intent of a person"
        }
    )

    print(batch_requests)
    print(client.batches.retrieve(batch_requests.id))
    #Here we are making a file with the output
    content = client.files.content(client.batches.retrieve(batch_requests.id).output_file_id)
    content.write_to_file("E:\projects\Research\batchRequestsOutput.jsonl")
