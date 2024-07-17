import os
import json
import tqdm

# Any global variables we have to initialize and use for later
listOfBodyParts3D = ['place holder', 'hips','RightUpLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'Left Foot', 
                   'Spine', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']
listofBodyParts2D = ['place holder','nose', 'leftEye', 'rightEye', 'LeftEar', 'rightEar', 'leftShoulder','leftElbow','leftWrist',
                     'rightWrist', 'leftHip', 'rightHip','leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle']
dataDict = {}


def process_files_in_folder(folder_path):
    count = 1
    for file_name in tqdm.tqdm(os.listdir(folder_path)):
        if file_name.endswith(".json"):  # Check if the file is JSON
            wholeFileName = os.path.join(folder_path, file_name)
            with open(wholeFileName, 'r') as json_file:
                json_data = json.load(json_file)
                print(f"Processing file: {wholeFileName}")
                final_result = ""

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
                        cleaned_parts_list = listOfBodyParts3D[1:]
                        for body_part_num, body_part in enumerate(cleaned_parts_list): # for x in y -> x updates automatically as we move thorugh y
                            temp = '3D_poses: ' + body_part + ' ['
                            tx = extracted_data[body_part_num*6 + 0][1] if isinstance(extracted_data[body_part_num*6 + 0], list) else extracted_data[body_part_num*6 + 0]
                            temp += 'Tx=' + str(tx) + ', '
                            ty = extracted_data[body_part_num*6 + 1][1] if isinstance(extracted_data[body_part_num*6 + 1], list) else extracted_data[body_part_num*6 + 1]
                            temp += 'Ty=' + str(ty) + ', '
                            tz = extracted_data[body_part_num*6 + 2][1] if isinstance(extracted_data[body_part_num*6 + 2], list) else extracted_data[body_part_num*6 + 2]
                            temp += 'Tz=' + str(tz) + ', '
                            rx = extracted_data[body_part_num*6 + 3][1] if isinstance(extracted_data[body_part_num*6 + 3], list) else extracted_data[body_part_num*6 + 3]
                            temp += 'Rx=' + str(rx) + ', '
                            ry = extracted_data[body_part_num*6 + 4][1] if isinstance(extracted_data[body_part_num*6 + 4], list) else extracted_data[body_part_num*6 + 4]
                            temp += 'Ry=' + str(ry) + ', '
                            rz = extracted_data[body_part_num*6 + 5][1] if isinstance(extracted_data[body_part_num*6 + 5], list) else extracted_data[body_part_num*6 + 5]
                            temp += 'Rz=' + str(rz) + ', '
                            
                            temp += '], '
                            # print(temp)
                            final_result += temp

                        # numberOfIterations = len(extracted_data)
                        # for i in range(0, numberOfIterations, 6):
                        #     item = extracted_data[i]
                        #     if isinstance(item, list) and len(item) > 1 and i + 6 < len(listOfBodyParts):
                        #         dataDict[f'3data{i+1}'] = item[1]
                        #         print(f"{listOfBodyParts[i+1]} Tx: {dataDict[f'3data{i+1}']}, "
                        #             f"Ty={dataDict.get(f'3data{i+2}', 'N/A')}, "
                        #             f"Tz={dataDict.get(f'3data{i+3}', 'N/A')}, "
                        #             f"Rx={dataDict.get(f'3data{i+4}', 'N/A')}, "
                        #             f"Ry={dataDict.get(f'3data{i+5}', 'N/A')}, "
                        #             f"Rz={dataDict.get(f'3data{i+6}', 'N/A')}")



                    elif '2D' in key:
                        cleaned_parts_list2D = listofBodyParts2D[1:]
                        for cameraViewPoint in range(len(extracted_data)): #Takes into account how there are different camera angles
                            for body_part_num, body_part in enumerate(cleaned_parts_list2D): # for x in y -> x updates automatically as we move thorugh y
                                temp = '2D_poses: ' + 'camera view point: ' + str(cameraViewPoint) + ' { '+ body_part + ' ['
                                tx = extracted_data[cameraViewPoint][body_part_num][1]
                                temp += 'Tx=' + str(tx) + ', '
                                ty = extracted_data[cameraViewPoint][body_part_num][2]
                                temp += 'Ty=' + str(ty) + ', '
                                tconfidence = extracted_data[cameraViewPoint][body_part_num][3]
                                temp += 'confidence = ' + str(tconfidence) + ', '
                                
                                temp += ']'
                                temp += '}'
                                # print(temp)
                                final_result += temp
                print(final_result)
                print('-----------------Block ends here ------------------------')


folder_path = r'E:\\Prepared_Data\\Prepared_Data\\training\\P01_R01'
process_files_in_folder(folder_path)
