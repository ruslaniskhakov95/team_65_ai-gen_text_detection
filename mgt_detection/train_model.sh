# #!/bin/bash

# python pipeline/main.py --file_path 'data/data_updated.json' --out_path 'saved_updated' --num_labels 3 --model_name 'FacebookAI/xlm-roberta-base'

# Define the log file path
LOG_FILE="pipeline_log.txt"

python pipeline/main.py --file_path 'data/data_updated.json' --out_path 'saved_updated' --num_labels 3 --model_name 'FacebookAI/xlm-roberta-base' 2>&1 | tee -a "$LOG_FILE"

# Optionally, print a message indicating that the script has finished and the log file location
echo "Script execution completed. Check the log file at: $LOG_FILE"