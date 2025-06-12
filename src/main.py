import argparse
import json
from src.pipeline import Pipeline
from src.models.model_factory import get_model, SUPPORTED_MODELS
from src.data_loaders.bbq_dataset import BBQDataset
import datetime

def main():
    parser = argparse.ArgumentParser(description="Run the bias evaluation pipeline.")
    parser.add_argument("--model_e", type=str, required=True, choices=SUPPORTED_MODELS,
                        help="The model to be evaluated (Model-E).")
    parser.add_argument("--model_j", type=str, required=True, choices=SUPPORTED_MODELS,
                        help="The judge model (Model-J).")
    parser.add_argument("--dataset", type=str, default="bbq", help="The dataset to use.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to run from the dataset.")
    parser.add_argument("--output_file", type=str, help="File to save the results.")

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
    print(f"Running on {args.num_samples} samples. ")
    
    results = pipeline.run()
    if args.output_file:
        output_file = args.output_file
    else:   
        output_file = f"outputs/{args.model_e}_{args.model_j}_{args.dataset}_{args.num_samples}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.json"
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Pipeline run complete. Results saved to {args.output_file}")


if __name__ == "__main__":
    main() 