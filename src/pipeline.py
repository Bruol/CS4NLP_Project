from models.base_model import ModelEBase, ModelJBase
from data_loaders.base_dataset import BaseDataset
from data_loaders.mitigation_dataset import MitigationDataset
from typing import List, Dict, Any
from tqdm import tqdm
from mitigation.mitigation import mitigate_adbp, mitigate_sfrp
import json

class Pipeline:
    """
    The main pipeline for running the bias evaluation.
    """

    def __init__(self, model_e: ModelEBase, model_j: ModelJBase, dataset: BaseDataset, mitigation: str = "disabled"):
        """
        Initializes the pipeline.

        Args:
            model_e (ModelEBase): The model to be evaluated.
            model_j (ModelJBase): The judge model.
            dataset (BaseDataset): The dataset to use for evaluation.
        """
        self.model_e = model_e
        self.model_j = model_j
        self.dataset = dataset
        self.mitigation = mitigation

    def run(self, output_file: str) -> List[Dict[str, Any]]:
        """
        Runs the evaluation pipeline.

        Args:
            with_cot (bool): Whether to use chain-of-thought for Model-E.

        Returns:
            List[Dict[str, Any]]: A list of results, where each result is a dictionary.
        """
        results = []
        for sample in tqdm(self.dataset, desc="Running evaluation"):
            # 1. Generate response from Model-E
            try:
                model_e_response = self.model_e.generate_response(
                    prompt=sample["prompt"], 
                )
            except Exception as e:
                print(f"Error generating response: {e}")
                continue

            # 2. Evaluate the response with Model-J
            evaluation = self.model_j.evaluate_response(model_e_response["thought"], sample["model_j_prompt"])


            if self.mitigation == "adbp":
                # 3. Apply mitigation if answer label is incorrect
                mitigation_response, per_step_answers, per_step_biases = mitigate_adbp(lambda x: self.model_e.generate_response(x)["response"], sample["prompt"],
                                            model_e_response["thought_steps"])
            elif self.mitigation == "sfrp":
                # 3. Apply mitigation if answer label is incorrect
                mitigation_response, per_step_answers, per_step_biases = mitigate_sfrp(lambda x: self.model_e.generate_response(x)["response"], sample["prompt"],
                                            model_e_response["thought_steps"], lambda x: self.model_j.evaluate_response(x, sample["model_j_prompt"])["bias_score"])
            else:
                mitigation_response = None
                per_step_answers = None
                per_step_biases = None

            # 4. Store the results
            result = {
                "dataset_sample": sample,
                "model_e_response": model_e_response,
                "model_j_evaluation": evaluation,
                "mitigation_response": mitigation_response,
                "per_step_answers": per_step_answers,
                "per_step_biases": per_step_biases
            }
            results.append(result)


            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)

        return results