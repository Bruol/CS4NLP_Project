python main.py --dataset bbq --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --num_samples 2750
python main.py --dataset bbq --model_e google/gemini-2.5-pro --model_j openai/gpt-4o --num_samples 2750
python main.py --dataset bbq --model_e openai/o4-mini --model_j openai/gpt-4o --num_samples 2750
python main.py --dataset bbq --model_e groq/deepseek-r1-distill-llama --model_j openai/gpt-4o --num_samples 2750
python main.py --dataset stereoset --model_e google/gemini-2.5-pro --model_j openai/gpt-4o --num_samples 1000




python main.py --dataset stereoset --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --num_samples 1000
python main.py --dataset stereoset --model_e openai/o4-mini --model_j openai/gpt-4o --num_samples 1000
python main.py --dataset stereoset --model_e groq/deepseek-r1-distill-llama --model_j openai/gpt-4o --num_samples 1000

# Mitigation experiments
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --mitigation adbp
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --mitigation sfrp
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --mitigation cot
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --mitigation awareness
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --mitigation category

# Reasoning effort experiments
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --model_e_cot_length low
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j openai/gpt-4o --model_e_cot_length high
python main.py --dataset bbq --num_samples 2750 --model_e openai/o4-mini --model_j openai/gpt-4o --model_e_cot_length low
python main.py --dataset bbq --num_samples 2750 --model_e openai/o4-mini --model_j openai/gpt-4o --model_e_cot_length high
python main.py --dataset bbq --num_samples 2750 --model_e groq/deepseek-r1-distill-llama --model_j openai/gpt-4o --model_e_cot_length low
python main.py --dataset bbq --num_samples 2750 --model_e groq/deepseek-r1-distill-llama --model_j openai/gpt-4o --model_e_cot_length high

# Different model-j
python main.py --dataset bbq --num_samples 2750 --model_e google/gemini-2.5-flash --model_j google/gemini-2.5-flash
python main.py --dataset bbq --num_samples 2750 --model_e groq/deepseek-r1-distill-llama --model_j google/gemini-2.5-flash
python main.py --dataset bbq --num_samples 2750 --model_e openai/o4-mini --model_j google/gemini-2.5-flash
