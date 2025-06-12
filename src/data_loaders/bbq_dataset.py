from src.data_loaders.base_dataset import BaseDataset
from datasets import load_dataset
from typing import Iterator, Dict, Any

class BBQDataset(BaseDataset):
    """
    Wrapper for the BBQ dataset from Hugging Face.
    """

    def __init__(self, num_samples: int = None, split: str = "age"):
        super().__init__(f"BBQ_{split}")
        self.dataset = load_dataset("Elfsong/BBQ")[split]
        if num_samples:
            self.dataset = self.dataset.select(range(num_samples))
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterates over the dataset samples, yielding a dictionary with a 'prompt' field.
        """
        for sample in self.dataset:
            # The BBQ dataset has a question and context. We'll format them into a single prompt.
            # We also need to present the multiple-choice options.
            prompt = f"Context: {sample['context']}\n\n"
            prompt += f"Question: {sample['question']}\n\n"
            prompt += f"Options:\n"
            prompt += f"1. {sample['ans0']}\n"
            prompt += f"2. {sample['ans1']}\n"
            prompt += f"3. {sample['ans2']}\n\n"
            prompt += "Please choose the best answer and explain your reasoning."

            yield {
                "prompt": prompt,
                "context": sample["context"],
                "question": sample["question"],
                "ans0": sample["ans0"],
                "ans1": sample["ans1"],
                "ans2": sample["ans2"],
                "label": sample["target_label"],
                "context_condition": sample["context_condition"],
                "question_polarity": sample["question_polarity"],
                "category": sample["category"]
            }

    def __len__(self) -> int:
        return len(self.dataset) 