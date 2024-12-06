from typing import List


class CustomLimeTextExplainer:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def explain_instance(self, text: str, classifier_fn, num_features: int = 6):
        # minimize faithfulness(classifier_fn, test_model, .5)
        # TODO: make a linear model through multiple dimensions, where each dimension is a word
        # do something like m_1 x_1 + m_2 x_2 + ... + m_n x_n + b
        # where m_i is the weight of the word x_i
        # set a weight to 0, then train a model and see how well it works
        # then set another weight to 0, and train a model and see how well it works
        # do so for every word until you can tell which words are the most important
        raise NotImplementedError