'''
this file takes in all 3 answer files for one topic and creates 3 files properly formatted for evaluation with MM_Safety_Bench
'''

import json
import os
import argparse

def read_json_lines(filepath):
    with open(filepath, 'r') as file:
        return [json.loads(line) for line in file]


def generate_reformatted_data(args, og_queries, answers):
    model_name = args.model_name
    reformatted_data = {}
    for qid, qex in og_queries.items():
        qid = int(qid)
        reformatted_data[qid] = qex.copy()
        reformatted_data[qid]['ans'] = {
            model_name: {
                "text": answers[qid]['text']
            }
        }
    return reformatted_data

def reformat_data(args):

    # create all the directories to organize the files if they do not yet exist 
    sd_directory = f'{args.response_directory}/{args.model_name}_responses/reformatted_responses/SD_responses'
    sdt_directory = f'{args.response_directory}/{args.model_name}_responses/reformatted_responses/SDTYPO_responses'
    typo_directory = f'{args.response_directory}/{args.model_name}_responses/reformatted_responses/TYPO_responses'

    if not os.path.exists(sd_directory):
        os.makedirs(sd_directory)
    if not os.path.exists(sdt_directory):
        os.makedirs(sdt_directory)
    if not os.path.exists(typo_directory):
        os.makedirs(typo_directory)

    sd_orig = f'{args.response_directory}/{args.model_name}/original_responses/SD_answer_files/01-Illegal_Activity_SD_queries_answers.jsonl'
    sdt_orig = f'{args.response_directory}/{args.model_name}/original_responses/SDTYPO_answer_files/01-Illegal_Activity_SDTYPO_queries_answers.jsonl'
    typo_orig = f'{args.response_directory}/{args.model_name}/original_responses/TYPO_answer_files/01-Illegal_Activity_TYPO_queries_answers.jsonl'

    if not os.path.exists(sd_orig):
        raise ValueError("cannot find SD answers file")
    if not os.path.exists(sdt_orig):
        raise ValueError("cannot find SDTYPO answers file")
    if not os.path.exists(typo_orig):
        raise ValueError("cannot find TYPO answers file")
    
    # read json data

    with open(sd_orig, 'r') as sda, \
        open(sdt_orig, 'r') as sdta, \
        open(typo_orig, 'r') as ta, \
        open(args.og_query_path, 'r') as og:
        
        sd_answers = [json.loads(line) for line in sda]
        sdt_answers = [json.loads(line) for line in sdta]
        typo_answers = [json.loads(line) for line in ta]
        og_queries = json.load(og)
    
    # reformat the data 
    
    reformatted_sd = generate_reformatted_data(args, og_queries, sd_answers)
    reformatted_sdt = generate_reformatted_data(args, og_queries, sdt_answers)
    reformatted_typo = generate_reformatted_data(args, og_queries, typo_answers)

    # write to new fles in correct directories

    with open(f'{sd_directory}/{args.scenario_number}-SDreformatted.json', 'w') as sd_ref: 
        json.dump(reformatted_sd, sd_ref)
    
    print(f'Succesfully reformatted the data present in {sd_orig} to {sd_directory}/{args.scenario_number}-SDreformatted.json')

    with open(f'{sdt_directory}/{args.scenario_number}-SDTYPOreformatted.json', 'w') as sdt_ref:
        json.dump(reformatted_sdt, sdt_ref)

    print(f'Succesfully reformatted the data present in {sdt_orig} to {sdt_directory}/{args.scenario_number}-SDTYPOreformatted.json')
        
    with open(f'{typo_directory}/{args.scenario_number}-TYPOreformatted.json', 'w') as t_ref:
        json.dump(reformatted_typo,t_ref)

    print(f'Succesfully reformatted the data present in {typo_orig} to {typo_directory}/{args.scenario_number}-TYPOreformatted.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("scenario_name", type=str, help="for example: illegal activity")
    parser.add_argument("scenario_number", type=str, help="for example: 01 for illegal activity")
    parser.add_argument("model_path", type=str, default="llava-v1.5-13b")
    
    args = parser.parse_args()

    args.og_query_path = f'processed_questions/{args.scenario_number + "-" + args.scenario_name}.json'
    args.response_directory = "answers/llava_responses"

    args.model_name = os.path.basename(args.model_path)

    reformat_data(args)