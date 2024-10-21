#!/bin/bash

set -e

# usage
usage() {
  echo "Usage: $0 --query-code <code> --model-path <path> or $0 --all --model-path <path>"
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

# Default model path
model_path=""

# parrse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --query-code)
      query_code="$2"
      shift 2
      ;;
    --model-path)
      model_path="$2"
      shift 2
      ;;
    --all)
      all=true
      shift
      ;;
    *)
      usage
      ;;
  esac
done

# make sure we pass model
if [[ -z "$model_path" ]]; then
  echo "Error: --model-path argument is required."
  usage
fi

# if you only run one of the models
run_commands_for_code() {
  code=$1
  activity_name=${activity_map[$code]}

  if [ -z "$activity_name" ]; then
    echo "Invalid code: $code. No corresponding activity found."
    exit 1
  fi

  echo "Running for code: $code, Activity: $activity_name"

  # run the custom_model_vqa to run eval on our fine-tuned model
  echo "Running custom_model_vqa.py for attack category: ${code}-${activity_name}"
  python3 custom_model_vqa.py --attack-category "${code}-${activity_name}" --model-path "$model_path"

  #run answer_file_transform.py with the activity name
  echo "Running answer_file_transform.py for scenario name: $activity_name"
  echo "Running answer_file_transform.py for scenario number: $code"
  echo "Running answer_file_transform.py for model name: $model_path"
  python3 answer_file_transform.py "$activity_name" "$code" "$model_path"

  # run MM_eval.py with the scenario code
  echo "Running MM_eval.py for scenario number: $code"
  python3 MM_eval.py --scenario_numbers "$code" --model_name "$model_path"
}

# if --query-code flag is passed
if [[ ! -z "$query_code" ]]; then
  if [[ "$query_code" != "SD" ]]; then
    run_commands_for_code "$query_code"
  fi

# if --all flag is passed
elif [[ "$all" == true ]]; then
  for code in "${!activity_map[@]}"; do
    run_commands_for_code "$code"
  done

else
  usage
fi
