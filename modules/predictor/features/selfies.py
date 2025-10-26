"""Module for one-hot encoding of SELFIES representations of molecules."""
import selfies as sf

from modules.predictor.features.feature_factory import FeatureFactory


@FeatureFactory.register("one_hot_selfies")
class OneHotSelfiesEncoder:
    """
    One-hot encoder for SELFIES representations of molecules.
    Attributes:
        selfies_alphabet (dict): A dictionary mapping SELFIES characters to their corresponding indices.
        max_length (int): The maximum length of SELFIES strings in the dataset.
    Methods:
        fit_transform(smiles: list) -> list:
            Fit the encoder to the dataset and transform the SMILES strings to one-hot encoded SELFIES
            representations.
        get_feature_names_out() -> list:
            Get the names of the one-hot encoded SELFIES features.
    """

    def __init__(self):
        self.selfies_alphabet = {}
        self.max_length = 0

    def fit_transform(self, smiles):
        """
        Fit the encoder to the dataset and transform the SMILES strings to one-hot encoded SELFIES
        representations.
        :param smiles: list of SMILES strings.
        :return: list of one-hot encoded SELFIES representations."""
        for s in smiles:
            selfie = sf.encoder(s)
            len_selfies = len(list(sf.split_selfies(selfie)))
            if len_selfies > self.max_length:
                self.max_length = len_selfies
            for char in sf.split_selfies(selfie):
                if char not in self.selfies_alphabet:
                    self.selfies_alphabet[char] = len(self.selfies_alphabet)
        if "[nop]" not in self.selfies_alphabet:
            self.selfies_alphabet["[nop]"] = len(self.selfies_alphabet)
        selfies_batch = [sf.encoder(s) for s in smiles]
        one_hot_encoded = sf.batch_selfies_to_flat_hot(selfies_batch, self.selfies_alphabet, pad_to_len=self.max_length)
        return one_hot_encoded

    def get_feature_names_out(self):
        """
        Get the names of the one-hot encoded SELFIES features.
        :return: list of feature names."""
        feature_names = []
        for i in range(self.max_length):
            for char in self.selfies_alphabet.keys():
                feature_names.append(f"selfies_pos_{i}_{char}")
        return feature_names

    def get_feature_types(self):
        """
        Get the types of the one-hot encoded SELFIES features.
        :return: list of feature types."""
        return {"categorical": [], "binary": self.get_feature_names_out(), "numerical": []}
