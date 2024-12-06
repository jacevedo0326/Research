# GPT data generation
We used python 3.12

package requirements were handled through pip. See requirementsForGeneration.txt

To replicate results, run research.py with the following arguments

`-gpt_path`= this repos proccessed responses

`-output_path`= this repos path for output files

`-file_name`= the name of the sample being proccessed

`-prepared_data_path`= the path to where the samples exist

you will need your own openAI api key declared as API_KEY in file openAIKey.py

# GPT Evalution 
We used python 3.8

package requirements were handled through pip. See requirementsForValidation.txt

To replicate results, run the following files: 

perPhraseAccuracy.py with the argument `-gpt_path`= this repos proccessed responses

perTokenAccuracy.py with the argument `-gpt_path`= this repos proccessed responses

embeddedLoss.py
