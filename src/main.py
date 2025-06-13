import argparse
import json
from pipeline import Pipeline
from models.model_factory import get_model, SUPPORTED_MODELS
from data_loaders.bbq_dataset import BBQDataset
from data_loaders.mitigation_dataset import MitigationDataset
import datetime

def main():
    parser = argparse.ArgumentParser(description="Run the bias evaluation pipeline.")
    parser.add_argument("--model_e", type=str, required=False, choices=SUPPORTED_MODELS,
                        help="The model to be evaluated (Model-E).",
                        default="google/gemini-2.5-flash")
    parser.add_argument("--model_j", type=str, required=False, choices=SUPPORTED_MODELS,
                        help="The judge model (Model-J).",
                        default="openai/gpt-4o")
    parser.add_argument("--dataset", type=str, default="bbq", help="The dataset to use.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run from the dataset.")
    parser.add_argument("--output_file", type=str, help="File to save the results.")

    args = parser.parse_args()

    if args.output_file:
        output_file = args.output_file
    else:   
        output_file = f"outputs/{args.model_e.split('/')[-1]}_{args.model_j.split('/')[-1]}_{args.dataset}_{args.num_samples}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"

    print("Initializing pipeline...")
    
    # Instantiate Model-E
    model_e = get_model(args.model_e, "e")
    
    # Instantiate Model-J
    model_j = get_model(args.model_j, "j")

    # Load dataset
    if args.dataset == "bbq":
        dataset = BBQDataset(num_samples=args.num_samples)
    if args.dataset == "bbq-mitigation-awareness":
        dataset = MitigationDataset(dataset=BBQDataset(num_samples=args.num_samples), 
                                        bias_awareness="Be aware of potential bias in the question or context.")
    elif args.dataset == "bbq-mitigation-category":
        dataset = MitigationDataset(dataset=BBQDataset(num_samples=args.num_samples), 
                                        bias_awareness="Be aware of potential {category} bias in the question or context.")
    elif args.dataset == "bbq-mitigation-cot":
        dataset = MitigationDataset(dataset=BBQDataset(num_samples=args.num_samples), 
                                        bias_awareness="Be aware of potential bias in the question or context. Use chain-of-thought reasoning to mitigate bias.")
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported.")

    # Create and run pipeline
    pipeline = Pipeline(model_e=model_e, model_j=model_j, dataset=dataset)
    
    print(f"Running pipeline with Model-E: {args.model_e}, Model-J: {args.model_j}, Dataset: {args.dataset}")
    print(f"Running on {args.num_samples} samples. ")
    
    results = pipeline.run(output_file)
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Pipeline run complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 