import math
import re

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression


def calculate_weights(distance_array, kernel_width=25):
    return np.sqrt(np.exp(-(distance_array ** 2) / (kernel_width ** 2))) # Converts distances to weights (0 to 1) using a Gaussian distribution


def find_distance(perturbed, original):
    # cosine similarity for distance
    dot_product = sum(perturbed_i * original_i for perturbed_i, original_i in zip(perturbed, original))

    perturbed_magnitude = math.sqrt(sum(perturbed)) # All the elements are binary, so I don't need to square them.
    original_magnitude = math.sqrt(sum(original))

    if perturbed_magnitude == 0 or original_magnitude == 0:
        # it breaks if there's a zero vector (divide by zero)
        return 1.0  # maximum distance if one vector is zero
    cosine_similarity = dot_product / (perturbed_magnitude * original_magnitude)

    return 1 - cosine_similarity


def feature_selection(perturbed_masks, perturbed_labels, weights, num_features):
    # Repeatedly adds features to the model to find which features are most relevant
    # This function I didn't change much. I don't see a ton of ways I could make it work differently.
    linreg = LinearRegression(fit_intercept=True)
    used_features = []

    # Find which num_features features are most important by looping over the regression training
    # (running it once gets the most important feature, but we need many)
    for i in range(num_features):
        max_score = -math.inf
        best_feature = 0

        # find which set of features makes the best model approximation
        for feature in range(perturbed_masks.shape[1]):
            if feature in used_features: # skip the previously chosen features
                continue

            chosen_features = perturbed_masks[:, used_features + [feature]]
            linreg.fit(chosen_features, perturbed_labels, sample_weight=weights)
            score = linreg.score(chosen_features, perturbed_labels, sample_weight=weights)
            if score > max_score:
                best_feature = feature
                max_score = score
        used_features.append(best_feature)
    return used_features


def create_linear_approximation(perturbed_masks, perturbed_labels, perturbed_distances, label, num_features):
    weights = calculate_weights(perturbed_distances)
    labels_column = perturbed_labels[:, label]  # the label column for the label we're trying to explain

    used_features = feature_selection(perturbed_masks,
                                           labels_column,
                                           weights,
                                           num_features)

    # Create a linear model using the selected features
    model = Ridge(alpha=1, fit_intercept=True)
    model.fit(perturbed_masks[:, used_features], # only the words that were selected to be important
              labels_column, sample_weight=weights)

    return sorted(zip(used_features, model.coef_), key=lambda x: np.abs(x[1]), reverse=True)


class SplitString:
    def __init__(self, text_instance):
        self.raw_text = text_instance
        self.nonword_matcher = re.compile(r'(%s)|$' % r'\W+')
        self.split_string = [s for s in self.nonword_matcher.split(text_instance) if s]

        self.words = []
        for i, word in enumerate(self.split_string):
            if word not in self.words and not self.nonword_matcher.match(word):
                self.words.append(word)

    def mask_words(self, indices_to_remove):
        masked_list = np.array(self.split_string).copy()
        masked_list[self.transform_positions(indices_to_remove)] = ''
        return "".join(masked_list)

    def transform_positions(self, word_indices):
        positions = [] # This is slow, but it works.
        for word_index in word_indices:
            for i, word in enumerate(self.split_string):
                if self.words[word_index] == word:
                    positions.append(i)
        return positions


class CustomLimeTextExplainer(object):
    def __init__(self):
        self.rng = np.random.default_rng()

    def explain_instance(self, text_instance, classifier_fn, num_features, class_id):
        split_string = SplitString(text_instance)
        data_mask, labels, distances = self.generate_perturbed_samples(classifier_fn, 5000, split_string)

        explanations = create_linear_approximation(data_mask, labels, distances, class_id, num_features)

        return [(split_string.words[result[0]], float(result[1])) for result in explanations]

    def generate_perturbed_samples(self, classifier_fn, num_samples, split_string: SplitString):
        document_size = len(split_string.words)
        sample: np.ndarray = self.rng.integers(low=1, high=document_size + 1, size=num_samples - 1)

        data_mask: np.ndarray = np.ones((num_samples, document_size), dtype=int)
        distances: np.array = np.zeros(num_samples)
        masked_words_list = [split_string.raw_text]
        for i, mask_size in enumerate(sample, start=1):
            indices_to_mask = self.rng.choice(range(document_size), size=mask_size, replace=False)
            masked_words_list.append(split_string.mask_words(indices_to_mask))
            data_mask[i, indices_to_mask] = 0
            distances[i] = find_distance(data_mask[i], data_mask[0])
        labels = classifier_fn(masked_words_list)
        return data_mask, labels, distances
