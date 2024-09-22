from openai import OpenAI
import os
import time
from openAIKey import API_KEY



# Initialize the OpenAI client with your API key
client = OpenAI(api_key=API_KEY)

file_id_path = "file_id.txt"
if os.path.exists(file_id_path):
    with open(file_id_path, "r") as f:
        file_id = f.read().strip()
else:
    # Upload the user-provided file to OpenAI once
    folderPath = r'E:\projects\Research'
    fileName = 'Action-Meta-action-list.xlsx'
    FileName = os.path.join(folderPath, fileName)

    messageFile = client.files.create(
        file=open(FileName, "rb"), purpose="assistants"
    )
    
    # Save the file ID for future use
    file_id = messageFile.id
    with open(file_id_path, "w") as f:
        f.write(file_id)

# Create the assistant (include file reader tool)
assistant = client.beta.assistants.create(
    name = "Tool to predict a person's action",
    description = "This is an assistant that will be used to predict what a person's actions will soon be.",
    instructions = """
    The user will give you data about a person's body parts and positioning, this data is given from a 3d persepctive 
    where you recieve each body parts position in a x,y,z, format and their rotations in an x,y,z format.
    You also get each body parts position in a 2d perspective from multiple camera angles in an x,y format and you get a confidence level. 
    I then need you to tell the user what person observed is doing and predict what they are about to do. 
    You are to match each action to a specific meta-action and action found in the file you were given. 
    Take your time, there is absolutley no rush I want the answer to be correct not quick
    """,
    # temperature = 1,
    tools = [{"type": "file_search"}],
    model = "gpt-4o-mini",
    response_format = "auto",
    temperature = 0.4,
    # metadata = {
    #     "notes": "Focus on analyzing spatial relationships and movements to predict actions accurately. Make sure to use all the data given to come up with a single conlcusion",
    #     "format": "Firstly give the matching action and explain why you think that, then right below it the matching action and explain why you think that",
    #     "thought process": "First look at all the coordinates, from a 3d point of view, then look at all the coordinates from a 2d point of view. Then compare both before coming up with a conclusion",    },
)

# #Here we are uploading the files to openAI

#Here we are making the path
folderPath = r'E:\projects\Research'
fileName = 'Action-Meta-action-list.pdf'
FileName = os.path.join(folderPath, fileName)



# Upload the user provided file to OpenAI
messageFile = client.files.create(
  file=open(FileName, "rb"), purpose="assistants"
)
if __name__ ==  "__main__":
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
                        "text": '''
                        The format should be as follows. Meta-Action: "give meta action and follow up with an explanation of why you came up that meta-action here ". 
                        (New line) Action: "give action here and follow up with an explanation of how you came up with that action here". 
                        (New line) What their following move will be: "say what your prediction is here". To clarify, there should be a space in meta action, 
                        action, and the prediction
                        '''
                    },
                    {
                        "type": "text",
                        "text": 'The possible actions and meta actions which you are to pick from are given in the microsoft excell file titled "Action-Meta-action-list". '
                    }
                ],
                "attachments": [
                    { "file_id": messageFile.id, "tools": [{"type": "file_search"}] }
                ]
            }
        ]
    )

    # Print the thread response to check the result

    # Here we being to create our run
    run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
    )
    print('--------------------------------------New Run---------------------------------')
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
    latest_assistant_message = None
    for message in messageResponse.data:
        if message.role == 'assistant' and message.run_id == run.id:
            latest_assistant_message = message
            break
    #Print the latest response
    if latest_assistant_message:
        print(latest_assistant_message.content[0].text.value)
    else:
        print('No assistant message found for the run')

