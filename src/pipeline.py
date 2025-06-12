from src.models.base_model import ModelEBase, ModelJBase
from src.data_loaders.base_dataset import BaseDataset
from typing import List, Dict, Any
from tqdm import tqdm
from src.mitigation.adbp_mitigation import mitigate

class Pipeline:
    """
    The main pipeline for running the bias evaluation.
    """

    def __init__(self, model_e: ModelEBase, model_j: ModelJBase, dataset: BaseDataset):
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

    def run(self) -> List[Dict[str, Any]]:
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
            model_e_response = self.model_e.generate_response(
                prompt=sample["prompt"], 
            )

            # 2. Evaluate the response with Model-J
            evaluation = self.model_j.evaluate_response(model_e_response["thought"], sample)

            if model_e_response != sample["answer_label"]:
                # 3. Apply mitigation 
                mitigation_response = mitigate(lambda x: self.model_e.generate_response(x)["response"], sample["prompt"],
                                            model_e_response["thought_steps"])

            # 4. Store the results
            result = {
                "dataset_sample": sample,
                "model_e_response": model_e_response,
                "model_j_evaluation": evaluation,
                "mitigation_response": mitigation_response
            }
            results.append(result)

            

        return results 