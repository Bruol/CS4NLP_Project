from src.data_loaders.base_dataset import BaseDataset
from datasets import load_dataset, concatenate_datasets
from typing import Iterator, Dict, Any

class BBQDataset(BaseDataset):
    """
    Wrapper for the BBQ dataset from Hugging Face.
    """

    def __init__(self, num_samples: int = None):
        super().__init__("BBQ")
        dataset_dict = load_dataset("Elfsong/BBQ")
        splits = ["age", "disability_status", "gender_identity", "nationality", "physical_appearance", "race_ethnicity", "race_x_gender", "race_x_ses","religion", "ses", "sexual_orientation"]
        
        # Concatenate all splits
        self.dataset = dataset_dict[splits[0]]
        for split in splits[1:]:
            self.dataset = concatenate_datasets([self.dataset, dataset_dict[split]])
        
        # shuffle the dataset
        self.dataset = self.dataset.shuffle(seed=42)
        
        if num_samples:
            self.dataset = self.dataset.select(range(num_samples))
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterates over the dataset samples, yielding a dictionary with a 'prompt' field.
        """
        for sample in self.dataset:
            # The BBQ dataset has a question and context. We'll format them into a single prompt.
            # We also need to present the multiple-choice options.
            prompt = f"""context: {sample['context']}
            question: {sample['question']}
            ans0: {sample['ans0']}
            ans1: {sample['ans1']}
            ans2: {sample['ans2']}
            Based on the context, question and options. Output the final answer from options {{ans0, ans1, ans2}}
            enclosed within <answer> </answer> tags.
            """

            yield {
                "prompt": prompt,
                "context": sample["context"],
                "question": sample["question"],
                "ans0": sample["ans0"],
                "ans1": sample["ans1"],
                "ans2": sample["ans2"],
                "label": sample["answer_label"],
                "context_condition": sample["context_condition"],
                "question_polarity": sample["question_polarity"],
                "category": sample["category"]
            }

    def __len__(self) -> int:
        return len(self.dataset) 