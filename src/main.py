import argparse
import json
from src.pipeline import Pipeline
from src.models.model_factory import get_model, SUPPORTED_MODELS
from src.data_loaders.bbq_dataset import BBQDataset

def main():
    parser = argparse.ArgumentParser(description="Run the bias evaluation pipeline.")
    parser.add_argument("--model_e", type=str, required=True, choices=SUPPORTED_MODELS,
                        help="The model to be evaluated (Model-E).")
    parser.add_argument("--model_j", type=str, required=True, choices=SUPPORTED_MODELS,
                        help="The judge model (Model-J).")
    parser.add_argument("--dataset", type=str, default="bbq", help="The dataset to use.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run from the dataset.")
    parser.add_argument("--with_cot", action="store_true", help="Enable chain-of-thought for Model-E.")
    parser.add_argument("--output_file", type=str, default="results.json", help="File to save the results.")

    args = parser.parse_args()

    print("Initializing pipeline...")
    
    # Instantiate Model-E
    model_e = get_model(args.model_e, "e")
    
    # Instantiate Model-J
    model_j = get_model(args.model_j, "j")

    # Load dataset
    if args.dataset == "bbq":
        dataset = BBQDataset(num_samples=args.num_samples)
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported.")

    # Create and run pipeline
    pipeline = Pipeline(model_e=model_e, model_j=model_j, dataset=dataset)
    
    print(f"Running pipeline with Model-E: {args.model_e}, Model-J: {args.model_j}, Dataset: {args.dataset}")
    print(f"Running on {args.num_samples} samples. CoT enabled: {args.with_cot}")
    
    results = pipeline.run(with_cot=args.with_cot)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Pipeline run complete. Results saved to {args.output_file}")
    print("Example result:")
    print(json.dumps(results[0], indent=4))


if __name__ == "__main__":
    main() 