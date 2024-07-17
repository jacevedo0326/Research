import openai
from openai import OpenAI #What is the difference between this and the first one
import time
from openAIKey import API_KEY

# Initialize the OpenAI client with your API key
client = openai.OpenAI(api_key=API_KEY)

# Put the snippet of data we want the ai to analyze here
dataBeingLookedAt = """
Processing file: E:\\Prepared_Data\\Prepared_Data\\training\\P01_R01\10169.json
3D_poses: hips [Tx=-50.063631, Ty=94.40869899999998, Tz=-11.737820750000001, Rx=1.19313775, Ry=-73.20344925, Rz=-3.0312725, ], 3D_poses: RightUpLeg [Tx=0.343955, Ty=-41.284794, Tz=-1.350768, Rx=2.4619134999999996, Ry=11.3015995, Rz=-0.1846615, ], 3D_poses: RightFoot [Tx=25.983967, Ty=-0.062121, Tz=-0.0025235000000000006, Rx=-16.615688, Ry=-8.277639749999999, Rz=-3.57851725, ], 3D_poses: LeftUpLeg [Tx=0.0017765, Ty=-41.87448775, Tz=0.17761375, Rx=0.040609, Ry=-40.0618325, Rz=0.152264, ], 3D_poses: LeftLeg [Tx=9.25702325, Ty=-1.586394, Tz=0.15655975, Rx=0.15054700000000001, Ry=-0.73979175, Rz=-2.63351875, ], 3D_poses: Left Foot [Tx=26.4853225, Ty=-0.0142975, Tz=-0.058492999999999996, Rx=-28.69624175, Ry=-62.68074125, Rz=-41.8878765, ], 3D_poses: Spine [Tx=12.780302500000001, Ty=-0.022596500000000002, Tz=-0.141394, Rx=-7.548778749999999, Ry=9.3686675, Rz=-64.99687575, ], 3D_poses: Head [Tx=3.1999410000000004, Ty=6.91390275, Tz=-0.09096125, Rx=2.83658125, Ry=1.35828775, Rz=-0.98459725, ], 3D_poses: RightShoulder [Tx=-0.6563490000000001, Ty=-42.27439225, Tz=0.54479525, Rx=5.17636075, Ry=-40.1258285, Rz=7.29304175, ], 3D_poses: RightArm [Tx=-26.0034635, Ty=-0.0500135, Tz=-0.01216375, Rx=-20.805690499999997, Ry=-3.33296775, Rz=8.817348500000001, ], 3D_poses: RightForeArm [Tx=-0.057072, Ty=-41.8241455, Tz=0.00019024999999999988, Rx=28.95738325, Ry=-18.5851495, Rz=-3.8043815, ], 3D_poses: RightHand [Tx=-9.2900165, Ty=-1.5806009999999997, Tz=0.09928375, Rx=-7.93918925, Ry=-1.4309319999999999, Rz=-2.479265, ], 3D_poses: LeftShoulder [Tx=-26.4961005, Ty=-0.0007575000000000012, Tz=0.12896249999999998, Rx=14.828904249999999, Ry=82.04071225, Rz=21.823961750000002, ], 3D_poses: LeftArm [Tx=-12.762632250000001, Ty=0.14280525, Tz=0.16372375, Rx=-45.92429625, Ry=13.3904095, Rz=66.13076025000001, ], 3D_poses: LeftForeArm [Tx=-3.15092175, Ty=6.92934025, Tz=0.049332249999999994, Rx=-1.5032969999999999, Ry=33.156435, Rz=0.7963760000000001, ], 3D_poses: LeftHand [Tx=0.02351625, Ty=14.3062125, Tz=0.04491874999999999, Rx=0.6281774999999999, Ry=-2.74504925, Rz=-0.1326935, ], 2D_poses: camera view point: 0 { nose [Tx=0.6589207649230957, Ty=0.7798092365264893, 
confidence = 0.06103918328881264, ]}2D_poses: camera view point: 0 { leftEye [Tx=0.7069743275642395, Ty=0.7835136651992798, confidence = 0.04735730588436127, ]}2D_poses: camera view point: 0 { rightEye [Tx=0.030225610360503197, Ty=0.7864304780960083, confidence = 0.04984626919031143, ]}2D_poses: camera view point: 0 { LeftEar [Tx=0.9926832318305969, Ty=0.7981844544410706, confidence = 0.06180670112371445, ]}2D_poses: camera view point: 0 { rightEar [Tx=0.0815562754869461, Ty=0.8338464498519897, confidence = 0.05793289467692375, ]}2D_poses: camera view point: 0 { leftShoulder [Tx=0.9977133274078369, Ty=0.777574896812439, confidence = 0.17513705790042877, ]}2D_poses: camera view point: 0 { leftElbow [Tx=0.9930102229118347, Ty=0.864767849445343, confidence = 0.15407641232013702, ]}2D_poses: camera view point: 0 { leftWrist [Tx=0.9827144742012024, Ty=0.7230112552642822, confidence = 0.2948172688484192, ]}2D_poses: camera view point: 0 { rightWrist [Tx=0.8658266067504883, Ty=0.8859853744506836, confidence = 0.257354199886322, ]}2D_poses: camera view point: 0 { leftHip [Tx=0.9302648305892944, Ty=0.7029656171798706, confidence = 0.19900749623775482, ]}2D_poses: camera view point: 0 { rightHip [Tx=0.694858968257904, Ty=0.8477045297622681, confidence = 0.21374592185020447, ]}2D_poses: camera view point: 0 { leftKnee [Tx=0.9293989539146423, Ty=0.7871518135070801, confidence = 0.43237829208374023, ]}2D_poses: camera view point: 0 { rightKnee [Tx=0.8890830278396606, Ty=0.8418458700180054, confidence = 0.4225304126739502, ]}2D_poses: camera view point: 0 { leftAnkle [Tx=0.1146719828248024, Ty=0.7786461114883423, confidence = 0.5054314732551575, ]}2D_poses: camera view point: 0 { rightAnkle [Tx=0.0660385712981224, Ty=0.8379474878311157, confidence = 0.4965881407260895, ]}2D_poses: camera view point: 1 { nose [Tx=0.4042831361293793, Ty=0.7128241062164307, confidence = 0.6325451731681824, ]}2D_poses: camera view point: 1 { leftEye [Tx=0.04371228069067001, Ty=0.7108629941940308, confidence = 0.6231051087379456, ]}2D_poses: camera view point: 1 { rightEye [Tx=0.3713807165622711, Ty=0.7123673558235168, confidence = 0.6223556995391846, ]}2D_poses: camera view point: 1 { LeftEar [Tx=0.15910537540912628, Ty=0.6840544939041138, confidence = 0.6211256980895996, ]}2D_poses: camera view point: 1 { rightEar [Tx=0.9220626354217529, Ty=0.7046047449111938, confidence = 0.6221194863319397, ]}2D_poses: camera view point: 1 { leftShoulder [Tx=0.9942236542701721, Ty=0.6747345328330994, confidence = 0.6713128089904785, ]}2D_poses: camera view point: 1 { leftElbow [Tx=0.996238112449646, Ty=0.6983642578125, confidence = 0.6920127868652344, ]}2D_poses: camera view point: 1 { leftWrist [Tx=0.9616210460662842, Ty=0.6905491948127747, confidence = 0.7092089056968689, ]}2D_poses: camera view point: 1 { rightWrist [Tx=0.9854016304016113, Ty=0.7081971168518066, confidence = 0.77833491563797, ]}2D_poses: camera view point: 1 { leftHip [Tx=0.9332790374755859, Ty=0.7064845561981201, confidence = 0.6838369369506836, ]}2D_poses: camera view point: 1 { rightHip [Tx=0.9700098037719727, Ty=0.7402954697608948, confidence = 0.7326599955558777, ]}2D_poses: camera view point: 1 { leftKnee [Tx=0.996795117855072, Ty=0.685562014579773, confidence = 0.8356198072433472, ]}2D_poses: camera view point: 1 { rightKnee [Tx=0.9973044395446777, Ty=0.7034738063812256, confidence = 0.8453158736228943, ]}2D_poses: camera view point: 1 { leftAnkle [Tx=0.9804967045783997, Ty=0.6945183873176575, confidence = 0.9426034688949585, ]}2D_poses: camera view point: 1 { rightAnkle 
[Tx=0.9850603938102722, Ty=0.7139014601707458, confidence = 0.9513530135154724, ]}2D_poses: camera view point: 2 { nose [Tx=0.09510008990764618, Ty=0.27224376797676086, confidence = 0.4231742024421692, ]}2D_poses: camera view point: 2 { leftEye [Tx=0.054096769541502, Ty=0.2686317265033722, confidence = 0.42695870995521545, ]}2D_poses: camera view point: 2 { rightEye [Tx=0.025388944894075394, Ty=0.2836248278617859, confidence = 0.4308708608150482, ]}2D_poses: camera view point: 2 { LeftEar [Tx=0.7473339438438416, Ty=0.25664225220680237, confidence = 0.46254992485046387, ]}2D_poses: camera view point: 2 { rightEar 
[Tx=0.4173848032951355, Ty=0.3232335150241852, confidence = 0.4624462425708771, ]}2D_poses: camera view point: 2 { leftShoulder [Tx=0.6862881779670715, Ty=0.2267288714647293, confidence = 0.4790724515914917, ]}2D_poses: camera view point: 2 { leftElbow [Tx=0.7448199987411499, Ty=0.33940589427948, confidence = 0.4842931926250458, ]}2D_poses: camera view point: 2 { leftWrist [Tx=0.6971189975738525, Ty=0.180637389421463, confidence = 0.42831528186798096, ]}2D_poses: camera view point: 2 { rightWrist [Tx=0.7434142231941223, Ty=0.3741396367549896, confidence = 0.4633362591266632, ]}2D_poses: camera view point: 2 { leftHip [Tx=0.8981610536575317, Ty=0.19102808833122253, confidence = 0.32101383805274963, ]}2D_poses: camera view point: 2 { rightHip [Tx=0.9005992412567139, 
Ty=0.38768357038497925, confidence = 0.42164137959480286, ]}2D_poses: camera view point: 2 { leftKnee [Tx=0.1836039125919342, Ty=0.23306655883789062, confidence = 0.43054628372192383, ]}2D_poses: camera view point: 2 { rightKnee [Tx=0.20401780307292938, Ty=0.30631619691848755, confidence = 0.4327780306339264, ]}2D_poses: camera view point: 2 { leftAnkle [Tx=0.2024153470993042, Ty=0.2304302155971527, confidence = 0.38967373967170715, ]}2D_poses: camera view point: 2 { rightAnkle [Tx=0.22324037551879883, Ty=0.31605905294418335, confidence = 0.3793770372867584, ]}
"""

# Define maximum token limits (if needed)
max_prompt_tokens = 500
max_completion_tokens = 1000

# Create the assistant (include file reader tool)
assistant = client.beta.assistants.create(
    name="Tool to predict a person's action",
    instructions="The user will give you data about a person's body parts and positioning and I need you to tell me what they are doing and predict what they are about to do.",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_reader"}
    ],
    model="gpt-3.5-turbo",
)

# Upload the text file (Is this what we have to do?)
file = client.files.create(
    file=open("positions.txt", "rb"),  # We need to replace this with the file we are about to create
    purpose="data"
)

# Create a thread with the file content
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
                    # Here is what I believe to be the temporary way to input out data
                    "text": dataBeingLookedAt
                },
                {
                    "type": "file",
                    "file_id": file.id
                }
            ]
        }
    ]
)

# Print the thread response to check the result
# I don't know exactly why or if I should do this
print(thread)

# Here we being to create our run
run = client.beta.threads.runs.create_and_poll(
  thread_id=thread.id,
  assistant_id=assistant.id
)
print(f" Run created: {run.id}")

# This lets us know whether the run is being completed or not
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(thread_id = thread.id, run_id=run.id)
    print(f" Run status: {run.status}")
    time.sleep(0.5)
else:
    print("Run Completed!")

# Here we print a response back
messageResponse = client.beta.threads.messages.list(thread_id = thread.id) #This accesses the thread
messages = messageResponse.data # Just transfering the data
#Do I have to print back the latest messages
latestMessage = messages[0]
print(f"Response: {latestMessage.content[0].text.value}")

