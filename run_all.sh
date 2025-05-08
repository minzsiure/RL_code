#!/bin/bash


TIMESTAMP=$(date +"%Y%m%d_%H%M")
DEFAULT_LOG_FILE="save_dirs_$TIMESTAMP.txt"


LOG_FILE="$DEFAULT_LOG_FILE"
INPUT_FILE=""
GPU_VISIBLE=""


while getopts "i:o:g:" opt; do
  case $opt in
    i)
      INPUT_FILE="$OPTARG"
      ;;
    o)
      LOG_FILE="$OPTARG"
      ;;
    g)
      GPU_VISIBLE="$OPTARG"
      ;;
    *)
      echo "Usage: $0 -i <commands_file> [-o <output_file>] [-g <gpu_id>]"
      exit 1
      ;;
  esac
done


if [[ -z "$INPUT_FILE" ]]; then
  echo "❌ Error: You must specify an input file with -i"
  echo "Usage: $0 -i <commands_file> [-o <output_file>] [-g <gpu_id>]"
  exit 1
fi


mkdir -p logs
> "$LOG_FILE"
echo "📝 Logging save_dirs to: $LOG_FILE"
echo "📂 Log files saved in: ./logs/"


LINE_NUM=0
while IFS= read -r CMD
do
    ((LINE_NUM++))
    echo "🔹 [$LINE_NUM] Running: $CMD"
    
    UNIQUE_PREFIX="$(date +%Y%m%d_%H%M%S)_$$"
    LOG_TMP="logs/log_${UNIQUE_PREFIX}_$(printf "%03d" $LINE_NUM).txt"

    
    if [[ -n "$GPU_VISIBLE" ]]; then
        CUDA_VISIBLE_DEVICES=$GPU_VISIBLE bash -c "$CMD" | tee "$LOG_TMP"
    else
        bash -c "$CMD" | tee "$LOG_TMP"
    fi

    
    # SAVE_DIR=$(grep -i "data saving dir:" "$LOG_TMP" | grep -oE "/[^ ]+" | head -n 1)
    SAVE_DIR=$(grep -i "data saving dir:" "$LOG_TMP" | sed -E 's/.*data saving dir:\s*//I' | head -n 1)

    echo "$CMD --> $SAVE_DIR" >> "$LOG_FILE"
    echo "✅ Saved: $SAVE_DIR"
    echo "-----------------------------------------"
done < "$INPUT_FILE"
