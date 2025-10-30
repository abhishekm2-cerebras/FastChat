#!/bin/sh
#SBATCH --time=2-00:00:00
#SBATCH --account=cerebras   # Specify your Slurm account here
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=vicuna-vllm
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -p gpumid
#SBATCH --mail-user=abhishek.maiti@mbzuai.ac.ae
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/abhishek.maiti/slurm-output/slurm-%j-%x.out
#SBATCH --error=/home/abhishek.maiti/slurm-output/slurm-%j-%x.err

source /home/abhishek.maiti/bfcl/bin/activate # or any env which has HF changes incorporated.

bench_name="vicuna_ar_extended"

source .env



# Define model variables
model1_path="XXXXX/lustre/scratch/users/sarath.chandran/verl/experiments/humaneval/ckpt_dir/jais2-8b_lr1e-6_bs16_epochs2_samples1000_rollouts8_baseckpt_PPO_ArmoRM_no_kl_loss_actor_bs16_critic_bs16/global_step_2772/actor/hf_model"
model1_id="250917_Jais2_8b_DPO_enhanced_format_continual_SFT_taxonomy_decontaminated1_replay_50pct_c2278_jais-2"

model2_path="XXXXX/lustre/scratch/users/ahmed.frikha/ckpts/250917_Jais2_8b_DPO_enhanced_format_continual_SFT_taxonomy_decontaminated1_replay_50pct_c2278/models--LLMer789--250917_Jais2_8b_DPO_enhanced_format_continual_SFT_taxonomy_decontaminated1_replay_50pct_c2278/snapshots/983bd7df1ac6ffc664d930e5f5fa9c12414a2787"
model2_id="allam-7-b-llama-2"

# Step 1: Generate model answers
# Check if the file $model1_id.jsonl exists in /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_ar_extended/model_answer
if [ ! -f "/home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/$bench_name/model_answer/$model1_id.jsonl" ]; then
    python gen_model_answer.py --model-path "$model1_path" --model-id "$model1_id" --bench-name $bench_name --num-gpus-per-model 1 --num-gpus-total 4 --vllm
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

# # Step 4: Generate XL Sheet for $model1_id 
# /home/abhishek.maiti/venvs/vicuna/bin/python /home/abhishek.maiti/projects/utils/vicuna_json_to_xl.py --question_jsonl /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_ar_extended/question.jsonl --answer_jsonl /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_x_logic/model_answer/$model1_id.jsonl

# # Step 5 translate 
# python /home/abhishek.maiti/projects/utils/translate_ar_to_en.py --input_excel /home/abhishek.maiti/projects/abhishekm2-cerebras/FastChat/fastchat/llm_judge/data/vicuna_x_logic/answer_excel/$model1_id.xlsx

