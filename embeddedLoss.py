import tensorflow as tf
import keras_nlp
import json
from generators import tokenizer
from losses import embed_loss
from Prepared_Data.dataset_params import BERT_MODEL, SEQ_LEN
from tqdm import tqdm

tf.config.run_functions_eagerly(True)

# tf.debugging.set_log_device_placement(True)

# limit = 3  # How many files to process
file_path = '.'  # File path to json file being processed

preprocessor = keras_nlp.models.BertPreprocessor(tokenizer=tokenizer, sequence_length=SEQ_LEN)
embedder = keras_nlp.models.BertBackbone.from_preset(BERT_MODEL)


def string_to_bert_embedding(input_string):
    bert_inputs = preprocessor(input_string)
    bert_embeddings = embedder(bert_inputs)['sequence_output']

    return bert_embeddings

import os
def process_json_file(file_path, limit=None):
    # Load the JSON file
    meta_averages = []
    action_averages = []
    for gpt_json in os.listdir(file_path):
        if 'gptResponses' not in gpt_json:
            continue

        with open(gpt_json, 'r') as f:
            data = json.load(f)

        # # Determine the number of items to process
        # items_to_process = min(limit, len(data)) if limit else len(data)

        # Create a tqdm progress bar
        listOfMetaActionLosses = []
        listOfActionLosses = []
        for key, value in tqdm(list(data.items()), desc="Processing files"):
            # Extract Meta-Action
            metaAction = value.get('Meta-Action', 'Not found')
            correctMetaAction = value.get('Correct Meta-Action', 'Not found')
            metaAction = [metaAction]
            correctMetaAction = [correctMetaAction]
            meta_action_len = metaAction.count(' ') + 3
            correct_meta_action_len = correctMetaAction.count(' ') + 3
            meta_action_len = max(meta_action_len, correct_meta_action_len)
            metaAction = tf.convert_to_tensor(metaAction, dtype=tf.string)
            correctMetaAction = tf.convert_to_tensor(correctMetaAction, dtype=tf.string)
            # print(metaAction)
            # print(correctMetaAction)

            # Extract Action
            action = value.get('Action', 'Not found')
            correctAction = value.get('Correct Action', 'Not found')
            action = [action]
            correctAction = [correctAction]
            action_len = action.count(' ') + 3
            correct_action_len = correctAction.count(' ') + 3
            action_len = max(action_len, correct_action_len)

            metaActionEmbedded = string_to_bert_embedding(metaAction)
            # print(f'metaActionEmbedded: {metaActionEmbedded}')
            conditional = tf.ones_like(metaActionEmbedded)
            conditional = tf.cumsum(conditional, axis=-2)
            metaActionEmbedded = tf.where(conditional < meta_action_len, metaActionEmbedded[0],
                                          tf.zeros_like(metaActionEmbedded)[0])

            correctMetaActionEmbedded = string_to_bert_embedding(correctMetaAction)
            conditional = tf.ones_like(correctMetaActionEmbedded)
            conditional = tf.cumsum(conditional, axis=-2)
            correctMetaActionEmbedded = tf.where(conditional < meta_action_len, correctMetaActionEmbedded[0],
                                                 tf.zeros_like(correctMetaActionEmbedded)[0])

            actionEmbedded = string_to_bert_embedding(action)
            # print(f'actionEmbedded: {actionEmbedded}')
            conditional = tf.ones_like(actionEmbedded)
            conditional = tf.cumsum(conditional, axis=-2)
            actionEmbedded = tf.where(conditional < action_len, actionEmbedded[0], tf.zeros_like(actionEmbedded)[0])

            correctActionEmbedded = string_to_bert_embedding(correctAction)
            conditional = tf.ones_like(correctActionEmbedded)
            conditional = tf.cumsum(conditional, axis=-2)
            correctActionEmbedded = tf.where(conditional < action_len, correctActionEmbedded[0],
                                             tf.zeros_like(correctActionEmbedded)[0])

            # expand the action and meta-actions
            metaActionExpanded = tf.expand_dims(metaActionEmbedded, 0)
            correctMetaActionExpanded = tf.expand_dims(correctMetaActionEmbedded, 0)
            actionExpanded = tf.expand_dims(actionEmbedded, 0)
            correctActionExpanded = tf.expand_dims(correctActionEmbedded, 0)
            # print(f'metaActionExpanded: {metaActionExpanded}')
            # print(f'correctMetaActionExpanded: {correctMetaActionExpanded}')
            # print(f'ActionExpanded: {ActionExpanded}')
            # print(f'correctActionExpanded: {correctActionExpanded}')

            metaActionLoss = embed_loss(metaActionExpanded, correctMetaActionExpanded)
            # print(f'embedLoss: {metaActionLoss}')
            actionLoss = embed_loss(actionExpanded, correctActionExpanded)
            # print(f'embedLoss: {actionLoss}')
            # print(metaActionLoss.shape)

            metaAverage = tf.reduce_mean(metaActionLoss, axis=-1)
            print(f'average: {metaAverage}')
            metaNumvalid = tf.math.count_nonzero(metaAverage)
            metaNumvalid = tf.cast(metaNumvalid, dtype=tf.float32)
            print(f'numvalid: {metaNumvalid}')
            metaAverage = tf.reduce_sum(metaAverage) / metaNumvalid
            metaAverage = tf.where(tf.math.is_nan(metaAverage), tf.constant(0.0, dtype=tf.float32),
                                   metaAverage)  # New lines!!!!!!!

            actionAverage = tf.reduce_mean(actionLoss, axis=-1)
            print(f'average: {actionAverage}')
            numvalid = tf.math.count_nonzero(actionAverage)
            numvalid = tf.cast(numvalid, dtype=tf.float32)
            print(f'numvalid: {numvalid}')
            actionAverage = tf.reduce_sum(actionAverage) / numvalid
            actionAverage = tf.where(tf.math.is_nan(actionAverage), tf.constant(0.0, dtype=tf.float32),
                                     actionAverage)  # New lines!!!!!!!

            # Prints out Averages
            #print(f'Meta Average: {metaAverage}')
            #print(f'Action Average: {actionAverage}')

            listOfMetaActionLosses.append(metaAverage)
            listOfActionLosses.append(actionAverage)

        meanMetaActionLoss = sum(listOfMetaActionLosses) / len(listOfMetaActionLosses)
        meanActionLoss = sum(listOfActionLosses) / len(listOfActionLosses)
        print(f'meanMetaActionLoss: {meanMetaActionLoss}', f'meanActionLoss: {meanActionLoss}')
        meta_averages.append(meanMetaActionLoss)
        action_averages.append(meanActionLoss)

    action_average_loss = sum(action_averages)/len(action_averages)
    meta_average_loss = sum(meta_averages) / len(meta_averages)
    return meta_average_loss, action_average_loss


if __name__ == '__main__':
    finalMetaActionLoss, finalActionLoss = process_json_file(file_path)
    print(f'Meta Action loss is {finalMetaActionLoss}')
    print(f'Action loss is {finalActionLoss}')