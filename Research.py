import os
import json
import tqdm
import random
from openAiAssistant import assistant, message_file, client
import time
import re

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
numberOfRuns = 5 #Here we specify how many runs we are doing.
totalInputToken = 0
totalOutputToken = 0

def get_data_from_json_files(folder_path, file_name):
    global currentAction, currentMetaAction, nextAction, nextMetaAction

    with open(file_name, 'r', encoding='utf-8') as json_file:
        try:
            json_data = json.load(json_file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_name}")
            return None

        final_result = ""
        final_result3D = ""
        final_result2D = ""

        # Access the values directly using the keys
        currentAction = json_data.get("current_task_str", "Key not found")
        if currentAction != "Key not found":
            print(f"The current action is: {currentAction}")

        currentMetaAction = json_data.get("current_task_meta_str", "Key not found")
        if currentMetaAction != "Key not found":
            print(f"The current meta action is: {currentMetaAction}")

        nextAction = json_data.get("next_task_str", "Key not found")
        if nextAction != "Key not found":
            print(f"The next action is: {nextAction}")

        nextMetaAction = json_data.get("next_task_meta_str", "Key not found")
        if nextMetaAction != "Key not found":
            print(f"The next meta action is: {nextMetaAction}")

        for key, extracted_data in json_data.items():
            if "token" in key or "embed" in key:  # Skip if key contains 'token' or 'embed'
                continue
            if isinstance(extracted_data, list):
                raw_data = f"{key}: {', '.join(map(str, extracted_data))}"
            elif isinstance(extracted_data, str):
                raw_data = f"{key}: {extracted_data}"
            elif isinstance(extracted_data, float):
                raw_data = f"{key}: {str(extracted_data)}"
            else:
                print('******************* SOMETHING WENT WRONG!!!!!*******')
                continue

            if '3D_poses' in key:
                temp = '3D_poses: '
                for body_part_num, body_part in enumerate(listOfBodyParts3D):
                    part_data = body_part + ' ['
                    tx = extracted_data[body_part_num*6 + 0][1] if isinstance(extracted_data[body_part_num*6 + 0], list) else extracted_data[body_part_num*6 + 0]
                    part_data += 'Tx=' + str(tx) + ', '
                    ty = extracted_data[body_part_num*6 + 1][1] if isinstance(extracted_data[body_part_num*6 + 1], list) else extracted_data[body_part_num*6 + 1]
                    part_data += 'Ty=' + str(ty) + ', '
                    tz = extracted_data[body_part_num*6 + 2][1] if isinstance(extracted_data[body_part_num*6 + 2], list) else extracted_data[body_part_num*6 + 2]
                    part_data += 'Tz=' + str(tz) + ', '
                    rx = extracted_data[body_part_num*6 + 3][1] if isinstance(extracted_data[body_part_num*6 + 3], list) else extracted_data[body_part_num*6 + 3]
                    part_data += 'Rx=' + str(rx) + ', '
                    ry = extracted_data[body_part_num*6 + 4][1] if isinstance(extracted_data[body_part_num*6 + 4], list) else extracted_data[body_part_num*6 + 4]
                    part_data += 'Ry=' + str(ry) + ', '
                    rz = extracted_data[body_part_num*6 + 5][1] if isinstance(extracted_data[body_part_num*6 + 5], list) else extracted_data[body_part_num*6 + 5]
                    part_data += 'Rz=' + str(rz) + ', '
                    part_data += '], '
                    temp += part_data
                final_result3D += temp

            elif '2D' in key:
                final_result2D += '2D_poses: '  # Print '2D_poses' only once before camera view point 0
                for cameraViewPoint in range(len(extracted_data)):  # Takes into account how there are different camera angles
                    temp = f'camera view point: {cameraViewPoint} '
                    for body_part_num, body_part in enumerate(listofBodyParts2D): 
                        part_data = body_part + ' ['
                        tx = extracted_data[cameraViewPoint][body_part_num][1]
                        part_data += 'Tx=' + str(tx) + ', '
                        ty = extracted_data[cameraViewPoint][body_part_num][2]
                        part_data += 'Ty=' + str(ty) + ', '
                        tconfidence = extracted_data[cameraViewPoint][body_part_num][3]
                        part_data += 'confidence = ' + str(tconfidence) + ', '
                        part_data += '], '
                        temp += part_data
                    final_result2D += temp + ' '
                break  # Exit after processing the first camera viewpoint

        final_result = final_result3D + final_result2D
        return final_result

def process_gpt_response(file_name, gpt_response, output_json_path):
    try:
        with open(output_json_path, 'r') as f:
            gpt_response_dict = json.load(f)
    except FileNotFoundError:
        gpt_response_dict = {}

    # Regex patterns
    meta_action_pattern = r'Meta-Action:?\s*"?([^"\n]+)"?'
    action_pattern = r'Action:?\s*"?([^"\n]+)"?'

    # Find all matches
    meta_actions = re.findall(meta_action_pattern, gpt_response, re.IGNORECASE | re.MULTILINE)
    actions = re.findall(action_pattern, gpt_response, re.IGNORECASE | re.MULTILINE)

    file_dict = {}

    if meta_actions:
        file_dict['Meta-Action'] = meta_actions[-1].strip()  # Take the last match
    else:
        file_dict['Meta-Action'] = "No Meta-Action found"
    
    if actions:
        file_dict['Action'] = actions[-1].strip()  # Take the last match
    else:
        file_dict['Action'] = "No Action found"

    gpt_response_dict[file_name] = file_dict

    with open(output_json_path, 'w') as f:
        json.dump(gpt_response_dict, f, indent=4)

    print(f"Processed and saved response for file: {file_name}")
    print(f"Meta-Action: {file_dict['Meta-Action']}")
    print(f"Action: {file_dict['Action']}")

if __name__ == "__main__":
    output_file_path = r'E:\projects\Research\researchOutput.txt'  # Specify the full path to the output file
    gpt_responses_json_path = r'E:\projects\Research\gpt_responses.json'  # New JSON file for GPT responses

    # Clear both files at the start of the run
    with open(output_file_path, 'w', encoding='utf-8') as f:
        pass
    with open(gpt_responses_json_path, 'w', encoding='utf-8') as f:
        json.dump({}, f)

    folder_path = r'E:\\Prepared_Data\\Prepared_Data\\training\\P01_R01'
    files_in_folder = [file_name for file_name in os.listdir(folder_path) if file_name.endswith(".json")]
    random.shuffle(files_in_folder)
    selected_files = files_in_folder[:numberOfRuns]

    print("Selected files for processing:")
    for idx, file in enumerate(selected_files):
        print(f"{idx + 1}. {file}")

    with open(output_file_path, 'a', encoding='utf-8') as f:  # Open file in append mode for the loop
        for file_name in tqdm.tqdm(selected_files):
            try:
                wholeFileName = os.path.join(folder_path, file_name)
                print(f"\nStarting to process file: {wholeFileName}")

                if not os.path.exists(wholeFileName):
                    print(f"File does not exist: {wholeFileName}")
                    continue

                file_result = get_data_from_json_files(folder_path, wholeFileName)
                if file_result is None:
                    continue

                print(f"Processing file: {wholeFileName}")
                print(file_result)
                print('-----------------Block ends here ------------------------')

                print(f"Creating thread for file: {file_name}")
                thread = client.beta.threads.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Here are the positions of the main joints and body parts of a person from a 3D and 2D perspective. They are labeled accordingly."
                                },
                                {
                                    "type": "text",
                                    "text": '''
                                    The format should be exactly as follows. Don't incllude (New line) but do have a line in between meta-action and Action
                                    Meta-Action: "give the predicted next meta-action". (New line) 
                                    Action: "give the predicted next action here". 
                                    '''
                                },
                                {
                                    "type": "text",
                                    "text": "Do not output any inforation about the following threads. The reader doesn't need to know about the microsoft excell file"
                                },
                                {
                                    "type": "text",
                                    # Here is the data we are giving it
                                    "text": file_result
                                },
                                {
                                    "type": "text",
                                    "text": '''
                                    The possible actions and meta actions which you are to pick from are given in the microsoft excell file titled "Action-Meta-action-list". 
                                    Do not tell ther reader about this. You also do not have to include the number in the most lefthand column. 
                                    For example, if you pick taking components, it should be taking components, not taking components9.
                                    The action also shouldn't include any numbers
                                    '''
                                },
                                {
                                    "type": "text",
                                    "text": "Please do not explain yourself. The only output you should give is the meta-action and action as stated above"
                                },
                            ],
                            "attachments": [
                                { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
                            ]
                        }
                    ]
                )
                print(f"Thread created for file: {file_name}")

                print(f"Starting run for file: {file_name}")
                run = client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=assistant.id
                )
                print('--------------------------------------New Run---------------------------------')
                print(f" Run created: {run.id}")

                # This lets us know whether the run is being completed or not
                while run.status != "completed":
                    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                    print(f" Run status: {run.status}")
                    time.sleep(0.5)
                else:
                    print("Run Completed!")

                # Here we print a response back
                messageResponse = client.beta.threads.messages.list(thread_id=thread.id)  # This accesses the thread
                latest_assistant_message = None
                for message in messageResponse.data:
                    if message.role == 'assistant' and message.run_id == run.id:
                        latest_assistant_message = message
                        break
                if run.usage:
                    inputTokenTemp = run.usage.prompt_tokens
                    outputTokenTemp = run.usage.completion_tokens
                    totalInputToken += inputTokenTemp
                    totalOutputToken += outputTokenTemp
                    print(f"Input tokens: {inputTokenTemp}, Output tokens: {outputTokenTemp}")
                else:
                    print("Token usage information not available for this run")
                # Print the latest response
                if latest_assistant_message:
                    gpt_response = latest_assistant_message.content[0].text.value
                    process_gpt_response(file_name, gpt_response, gpt_responses_json_path)
                    
                    print(gpt_response)
                    print(f"File number: {count}\nProcessing file: {wholeFileName} \n{gpt_response} \nThe correct current action should be: {currentAction} \nThe correct current meta action should be: {currentMetaAction} \nThe correct next action should be: {nextAction} \nThe correct next meta action should be: {nextMetaAction}", file=f)
                    print(f"")
                else:
                    print('No assistant message found for the run')
                count += 1
                print(f"Finished processing file: {wholeFileName}")
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
                continue
    print(f"Total input tokens used: {totalInputToken} \nTotal output tokens used: {totalOutputToken}")