#!/bin/bash

set -e

# uasage
usage() {
  echo "Usage: $0 --query-code <code> or $0 --all"
  exit 1
}

# argcheck
if [[ $# -lt 1 ]]; then
  usage
fi

declare -A activity_map=(
  ["01"]="Illegal_Activity"
  ["02"]="HateSpeech"
  ["03"]="Malware_Generation"
  ["04"]="Physical_Harm"
  ["05"]="EconomicHarm"
  ["06"]="Fraud"
  ["07"]="Sex"
  ["08"]="Political_Lobbying"
  ["09"]="Privacy_Violence"
  ["10"]="Legal_Opinion"
  ["11"]="Financial_Advice"
  ["12"]="Health_Consultation"
  ["13"]="Gov_Decision"
)

# if we pass in the code flag (if you want all codes to run pass -all falg)
run_commands_for_code() {
  code=$1
  activity_name=${activity_map[$code]}

  if [ -z "$activity_name" ]; then
    echo "Invalid code: $code. No corresponding activity found."
    exit 1
  fi

  echo "Running for code: $code, Activity: $activity_name"

  # run the custom_model_vqa to run eval on our fine tuned model
  echo "Running custom_model_vqa.py for attack category: ${code}-${activity_name}"
  python3 custom_model_vqa.py --attack-category "${code}-${activity_name}"

  # run answer_file_transform.py with the activity name
  echo "Running answer_file_transform.py for scenario name: $activity_name"
  echo "Running answer_file_transform.py for scenario number: $code"
  python3 answer_file_transform.py "$activity_name" "$code"

  # run MM_eval.py with the scenario number (code) in a list format
  echo "Running MM_eval.py for scenario number: $code"
  python3 MM_eval.py --scenario_numbers "$code"
}

if [[ $1 == "--query-code" ]]; then
  if [[ -z $2 ]]; then
    echo "You must provide a code when using --query-code flag"
    usage
  fi
  query_code=$2

  # make sure the query code is valid
  if [[ $query_code != "SD" ]]; then
    run_commands_for_code "$query_code"
  fi

elif [[ $1 == "--all" ]]; then
  # loop through all the codes if we turn on the all flag
  for code in "${!activity_map[@]}"; do
    run_commands_for_code "$code"
  done

else
  usage
fi
