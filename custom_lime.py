from typing import List


class CustomLimeTextExplainer:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def explain_instance(self, text: str, classifier_fn, num_features: int = 6):
        # minimize faithfulness(classifier_fn, test_model, .5)
        raise NotImplementedError