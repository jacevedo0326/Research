import os
import json
import tqdm
import random
from openAiAssistant import assistant, messageFile, client
import time
import re
import sys

# Any global variables we have to initialize and use for later
listOfBodyParts3D = ['hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 
                     'Spine', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']
listofBodyParts2D = ['nose', 'leftEye', 'rightEye', 'LeftEar', 'rightEar', 'leftShoulder', 'leftElbow', 'leftWrist',
                     'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']
count = 1
currentAction = ""
currentMetaAction = ""
nextAction = ""
nextMetaAction = ""
numberOfRuns = 3 #Here we specify how many runs we are doing.
totalInputToken = 0
totalOutputToken = 0
fileVariable = 'P04_R01' #Here we specify what folder is being looked at 

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
        if currentAction != "Key not found":
            print(f"The current action found")

        currentMetaAction = json_data.get("current_task_meta_str", "Key not found")
        if currentMetaAction != "Key not found":
            print(f"The current meta action found")

        nextAction = json_data.get("next_task_str", "Key not found")
        if nextAction != "Key not found":
            print(f"The next action is found")

        nextMetaAction = json_data.get("next_task_meta_str", "Key not found")
        if nextMetaAction != "Key not found":
            print(f"The next meta action found")

        for key, extractedData in json_data.items():
            if "token" in key or "embed" in key:  # Skip if key contains 'token' or 'embed'
                continue
            if isinstance(extractedData, list):
                raw_data = f"{key}: {', '.join(map(str, extractedData))}"
            elif isinstance(extractedData, str):
                raw_data = f"{key}: {extractedData}"
            elif isinstance(extractedData, float):
                raw_data = f"{key}: {str(extractedData)}"
            else:
                print('******************* SOMETHING WENT WRONG!!!!!*******')
                continue

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

if __name__ == "__main__":
    outputFilePath = f'E:\\projects\\Research\\researchOutput{fileVariable}.txt'
    gptResponsesJsonPath = f'E:\\projects\\Research\\gptResponses{fileVariable}.json'

    if not os.path.exists(gptResponsesJsonPath):
        with open(gptResponsesJsonPath, 'w') as f:
            json.dump({}, f)

    folderPath = f'E:\\Prepared_Data\\{fileVariable}'
    
    try:
        with open(gptResponsesJsonPath, 'r') as f:
            existing_responses = json.load(f)
    except json.JSONDecodeError:
        existing_responses = {}

    filesInFolder = [fileName for fileName in os.listdir(folderPath) if fileName.endswith(".json") and fileName not in existing_responses]
    selectedFiles = filesInFolder

    failed_count = 0
    max_failed_count = 10
    file_count = 0
    total_files = len(selectedFiles)

    for fileName in tqdm.tqdm(selectedFiles):
        try:
            file_count += 1
            wholeFileName = os.path.join(folderPath, fileName)
            print(f"\nProcessing file {file_count}/{total_files}: {wholeFileName}")

            if not os.path.exists(wholeFileName):
                print(f"File does not exist: {wholeFileName}")
                continue

            fileResult = getDataFromJsonFiles(folderPath, wholeFileName)
            if fileResult is None:
                continue

            print("\n" + "="*50)
            print("Extracted Data:")
            print("-"*50)
            print(fileResult)
            print("-"*50)

            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Here are the positions of the main joints and body parts of a person from a 3D and 2D perspective. They are labeled accordingly."},
                            {"type": "text", "text": "The format should be exactly as follows. Don't include (New line) but do have a line in between meta-action and Action\nMeta-Action: \"give the predicted next meta-action\". (New line) \nAction: \"give the predicted next action here\". "},
                            {"type": "text", "text": fileResult},
                            {"type": "text", "text": "Do not output any information about the following threads. The reader doesn't need to know about the microsoft excel file"},
                            {"type": "text", "text": "The possible actions and meta actions which you are to pick from are given in the microsoft excel file titled \"Action-Meta-action-list\". Do not tell the reader about this. You also do not have to include the number in the most lefthand column. For example, if you pick taking components, it should be taking components, not taking components9. The action also shouldn't include any numbers. Also don't reference your sources."},
                            {"type": "text", "text": "Please do not explain yourself. The only output you should give is the meta-action and action as stated above"}
                        ],
                        "attachments": [{ "file_id": messageFile.id, "tools": [{"type": "file_search"}] }]
                    }
                ]
            )

            print(f"Creating thread for file: {fileName}")
            print(f"Thread created for file: {fileName}")

            print(f"Starting run for file: {fileName}")
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            print('--------------------------------------New Run---------------------------------')
            print(f" Run created: {run.id}")

            start_time = time.time()
            while run.status != "completed":
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                elapsed_time = time.time() - start_time
                print(f" Run status: {run.status} (Elapsed time: {elapsed_time:.2f} seconds)")
                time.sleep(0.5)
                
                if run.status == "failed":
                    failed_count += 1
                    if failed_count >= max_failed_count:
                        print(f"Encountered {max_failed_count} consecutive failures. Restarting the script.")
                        python = sys.executable
                        os.execl(python, python, *sys.argv)
                else:
                    failed_count = 0
            else:
                print("Run Completed!")
                failed_count = 0  # Reset the counter when a run completes successfully

            messageResponse = client.beta.threads.messages.list(thread_id=thread.id)
            latest_assistant_message = next((message for message in messageResponse.data if message.role == 'assistant' and message.run_id == run.id), None)

            if latest_assistant_message:
                gptResponse = latest_assistant_message.content[0].text.value
                processGptResponse(fileName, gptResponse, gptResponsesJsonPath)
                
                if run.usage:
                    inputTokenTemp = run.usage.prompt_tokens
                    outputTokenTemp = run.usage.completion_tokens
                    totalInputToken += inputTokenTemp
                    totalOutputToken += outputTokenTemp
                    print(f"Input tokens: {inputTokenTemp}, Output tokens: {outputTokenTemp}")
                else:
                    print("Token usage information not available for this run")
                
                print("\nChatGPT's response:")
                print("-"*50)
                print(gptResponse)
                print("-"*50)
                
                # Write the assistant's response and correct actions/meta-actions to the file
                with open(outputFilePath, 'a', encoding='utf-8') as output_file:
                    output_file.write(f"\nFile number: {file_count}/{total_files}\n")
                    output_file.write(f"File: {wholeFileName}\n")
                    output_file.write("ChatGPT's response:\n")
                    output_file.write("-"*50 + "\n")
                    output_file.write(gptResponse + "\n")
                    output_file.write("-"*50 + "\n")
                    output_file.write("Correct Actions and Meta-Actions:\n")
                    output_file.write("-"*50 + "\n")
                    output_file.write(f"Correct current meta action: {currentMetaAction}\n")
                    output_file.write(f"Correct current action: {currentAction}\n")
                    output_file.write(f"Correct next meta action: {nextMetaAction}\n")
                    output_file.write(f"Correct next action: {nextAction}\n")
                    output_file.write("-"*50 + "\n\n")
                
                print("\nCorrect Actions and Meta-Actions:")
                print("-"*50)
                print(f"Correct current meta action: {currentMetaAction}")
                print(f"Correct current action: {currentAction}")
                print(f"Correct next meta action: {nextMetaAction}")
                print(f"Correct next action: {nextAction}")
                print("-"*50)
            else:
                print('No assistant message found for the run')
                with open(outputFilePath, 'a', encoding='utf-8') as output_file:
                    output_file.write(f"\nFile number: {file_count}/{total_files}\n")
                    output_file.write(f"File: {wholeFileName}\n")
                    output_file.write('No assistant message found for the run\n\n')

            print(f"Finished processing file {file_count}/{total_files}: {wholeFileName}")

        except Exception as e:
            print(f"Error processing file {file_count}/{total_files} - {fileName}: {str(e)}")
            with open(outputFilePath, 'a', encoding='utf-8') as output_file:
                output_file.write(f"\nFile number: {file_count}/{total_files}\n")
                output_file.write(f"Error processing file {fileName}: {str(e)}\n\n")
            continue

    print(f"Total input tokens used: {totalInputToken} \nTotal output tokens used: {totalOutputToken}")
    print(f"Total files processed: {file_count}/{total_files}")