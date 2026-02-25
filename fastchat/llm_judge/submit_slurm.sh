#!/bin/bash
#SBATCH --time=2-00:00:00
#SBATCH --account=cerebras   # Specify your Slurm account here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=vicuna-vllm
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -p gpumid
#SBATCH --mail-user=abhishek.maiti@mbzuai.ac.ae
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/abhishek.maiti/slurm-output/slurm-%j-%x.out
#SBATCH --error=/home/abhishek.maiti/slurm-output/slurm-%j-%x.err

source /home/abhishek.maiti/bfcl/bin/activate # or any env which has HF changes incorporated.

source .env

bench_names=(
    "vicuna_ar_extended"
    # "custom_human_preference_ar_bench"
)

for bench_name in "${bench_names[@]}"; do
    echo "Running for benchmark: $bench_name"
    

    # Check if OPENAI_API_KEY is set
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Error: OPENAI_API_KEY environment variable is not set."
        echo "Please set your OpenAI API key before running this script."
        echo "You can do this by running: export OPENAI_API_KEY=your_api_key"
        exit 1
    fi

    # Define model variables
    model1_path="XXXXX/lustre/scratch/users/abhishek.maiti/hf_ckpts/8b_LC_64K_all_newmix2_100pct"
    model1_id="allam-7-b-llama-2"

    model2_path="/lustre/scratch/users/abhishek.maiti/hf_ckpts/251204_8b_dpo_v0p5_excl_poetry_corruption_data_v1_bs160_beta0p1_lr_4e-5_351"
    model2_id="251204_8b_dpo_v0p5_excl_poetry_corruption_data_v1_bs160_beta0p1_lr_4e-5_351"

    # Step 1: Generate model answers
    # Check if the file $model1_id.jsonl exists in /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_ar_extended/model_answer
    if [ ! -f "/home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/$bench_name/model_answer/$model1_id.jsonl" ]; then
        python gen_model_answer.py --model-path "$model1_path" --model-id "$model1_id" --bench-name $bench_name --num-gpus-per-model 2 --num-gpus-total 2 --vllm
    fi

    if [ ! -f "/home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/$bench_name/model_answer/$model2_id.jsonl" ]; then    
        python gen_model_answer.py --model-path "$model2_path" --model-id "$model2_id" --bench-name $bench_name --num-gpus-per-model 2 --num-gpus-total 2 --vllm
    fi

if [ ! -f "/home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/$bench_name/model_answer/$model2_id.jsonl" ]; then    
    python gen_model_answer.py --model-path "$model2_path" --model-id "$model2_id" --bench-name $bench_name --num-gpus-per-model 1 --num-gpus-total 4 --vllm
fi

    # Step 2: Run judgments in parallel
    /home/abhishek.maiti/venvs/vicuna/bin/python gen_judgment.py --model-list "$model1_id" "$model2_id" --mode pairwise-all --bench-name $bench_name --judge-model gpt-5-chat-latest & 
    /home/abhishek.maiti/venvs/vicuna/bin/python gen_judgment.py --model-list "$model1_id" "$model2_id" --mode single --bench-name $bench_name --judge-model gpt-5-chat-latest &

    # Wait for both background jobs to finish
    wait

    # Step 3: Show results
    /home/abhishek.maiti/venvs/vicuna/bin/python show_result.py --model-list "$model1_id" "$model2_id" --bench-name $bench_name --mode pairwise-all --judge-model gpt-5-chat-latest
    /home/abhishek.maiti/venvs/vicuna/bin/python show_result.py --model-list "$model1_id" "$model2_id" --bench-name $bench_name --mode single --judge-model gpt-5-chat-latest
    
done

# # Step 4: Generate XL Sheet for $model1_id 
# /home/abhishek.maiti/venvs/vicuna/bin/python /home/abhishek.maiti/projects/utils/vicuna_json_to_xl.py --question_jsonl /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_ar_extended/question.jsonl --answer_jsonl /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_x_logic/model_answer/$model1_id.jsonl

# # Step 5 translate 
# python /home/abhishek.maiti/projects/utils/translate_ar_to_en.py --input_excel /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_x_logic/answer_excel/$model1_id.xlsx

