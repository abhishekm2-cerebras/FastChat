#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --account=cerebras   # Specify your Slurm account here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=test-vicuna-evals
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --mail-user=abhishek.maiti@mbzuai.ac.ae
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/abhishek.maiti/slurm-output/slurm-%j-%x.out
#SBATCH --error=/home/abhishek.maiti/slurm-output/slurm-%j-%x.err

source /home/abhishek.maiti/bfcl/bin/activate # or any env which has HF changes incorporated.

export OPENAI_API_KEY=

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set your OpenAI API key before running this script."
    echo "You can do this by running: export OPENAI_API_KEY=your_api_key"
    exit 1
fi

# Define model variables
model1_path="<model_1_path>"
model1_id="model_id_1"

model2_path="<model_2_path>"
model2_id="model_id_2"

# Step 1: Generate model answers
python gen_model_answer.py --model-path "$model1_path" --model-id "$model1_id" --bench-name vicuna_ar_extended --num-gpus-per-model 1 --num-gpus-total 4
python gen_model_answer.py --model-path "$model2_path" --model-id "$model2_id" --bench-name vicuna_ar_extended --num-gpus-per-model 1 --num-gpus-total 4



# Step 2: Run judgments in parallel
/home/abhishek.maiti/venvs/vicuna/bin/python gen_judgment.py --model-list "$model1_id" "$model2_id" --mode pairwise-all --bench-name vicuna_ar_extended & 
/home/abhishek.maiti/venvs/vicuna/bin/python gen_judgment.py --model-list "$model1_id" "$model2_id" --mode single --bench-name vicuna_ar_extended &

# Wait for both background jobs to finish
wait

# Step 3: Show results
/home/abhishek.maiti/venvs/vicuna/bin/python show_result.py --model-list "$model1_id" "$model2_id" --bench-name vicuna_ar_extended --mode pairwise-all
/home/abhishek.maiti/venvs/vicuna/bin/python show_result.py --model-list "$model1_id" "$model2_id" --bench-name vicuna_ar_extended --mode single