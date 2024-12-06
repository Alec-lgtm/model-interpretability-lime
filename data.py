from typing import Iterable, Sequence, Mapping
import torch
import os
import json

class IMDBData:
    """ A class to store data from the IMDB Dataset.
    """

    def __init__(self, data_path : str):
        self.labels = {"neg":0,
                       "pos":1}

        self.train = self.load_data(data_path + "train/{}.txt")
        self.dev = self.load_data(data_path + "dev/{}.txt")
        self.test = self.load_unlabeled_data(data_path + "test/test.txt")

    def load_data(self, data_path_fs : str) -> Iterable[tuple[Sequence[str], int]]:
        """ Load labeled data from the provided file path
        """

        out = []
        for label, y in self.labels.items():
            with open(data_path_fs.format(label)) as data_f:
                for line in data_f:
                    out.append((line.split(), y))
        return out

    def load_unlabeled_data(self, data_path :str) -> Iterable[Sequence[str]]:
        """ Load unlabeled (i.e., test) data from the provided file path.
        """
        out = []
        with open(data_path) as data_f:
            for line in data_f:
                out.append(line)
        return out

    def get_train_examples(self) -> Iterable[tuple[Sequence[str], int]]:
        """ Return an iterator over training examples (i.e., (data, label) pairs)
        """
        for example in self.train:
            yield example

    def get_dev_examples(self) -> Iterable[tuple[Sequence[str], int]]:
        """ Return an iterator over development examples (i.e., (data, label) pairs)
        """
        for example in self.dev:
            yield example

    def get_test_examples(self) -> Iterable[Sequence[str]]:
        """ Return an iterator over test examples (no labels!)
        """
        for example in self.test:
            yield example

class AuthorIDData:
    """ A class to store data for the author-id task
    """

    def __init__(self, data_path : str):
        self.problems = []
        self.train_probs = self.load_data(data_path + "train/")

        # Target labels will be "unknown" for test data!
        self.test_probs = self.load_data(data_path + "test/")

    def load_data(self, data_path : str) -> Iterable[tuple[Mapping[str, Sequence[Sequence[str]]],
                                                           Iterable[Sequence[str]],
                                                           Iterable[str]]]:
        """ Load data for a set of problems from the provided file path

            Parses file structure using the provided JSON files.
        """

        problems = []

        task_info = None
        with open(data_path + "collection-info.json") as in_f:
            task_info = json.load(in_f)

        for problem in task_info:
            problem_path = data_path + "{}/".format(problem["problem-name"])

            problem_info = None
            with open(problem_path + "problem-info.json") as in_f:
                problem_info = json.load(in_f)

            candidates = {}
            for candidate in problem_info["candidate-authors"]:
                author_path = problem_path + "{}/".format(candidate["author-name"])
                candidate_docs = []
                for filename in os.listdir(author_path):
                    with open(author_path + filename) as in_f:
                        candidate_docs.append(in_f.read().split())
                candidates[candidate["author-name"]] = candidate_docs

            unknown_path = problem_path + "{}/".format(problem_info["unknown-folder"])
            target_docs = []
            target_labels = []
            with open(problem_path + "ground-truth.json") as in_f:
                labels = json.load(in_f)["ground_truth"]

            for trial in labels:
                with open(unknown_path + trial["unknown-text"]) as in_f:
                    target_docs.append(in_f.read().split())
                target_labels.append(trial["true-author"])


            problems.append((candidates, target_docs, target_labels))
            return problems

    def get_train_problems(self):
        """ Return an iterator over training problems, each of which is a triplet
            (training data, target documents, labels)

            training data       - a dictionary matching candidate authors to documents written by them
            target documents    - a list of documents that should be labeled
            labels              - a list of the corresponding gold labels for target documents
        """

        for problem in self.train_problems:
            yield problem

    def get_test_problems(self):
        """ Return an iterator over test problems, each of which is a pair (training data, target documents)

            training data       - a dictionary matching candidate authors to documents written by them
            target documents    - a list of documents that should be labeled
        """

        for candidates, target_docs, _ in self.test_problems:
            yield candidates, target_docs   # For a test case, the labels are all "unknown"
