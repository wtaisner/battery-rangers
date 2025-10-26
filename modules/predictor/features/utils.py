"""Utility functions for feature type generation."""


def fingerprint_feature_types(name: str, feature_names: list[str], **kwargs) -> dict:
    """
    Generate feature types for fingerprints.
    :param name: name of the fingerprint.
    :param feature_names: list of feature names.
    :param kwargs: additional arguments for the fingerprint generation.
    :return: dictionary with feature types.
    """
    if name == "ecfp":
        count = kwargs.get("count", False)
        if count:
            return {"categorical": [], "binary": [], "numerical": feature_names}
        return {"categorical": [], "binary": feature_names, "numerical": []}
    elif name == "maccs":
        count = kwargs.get("count", False)
        if count:
            return {"categorical": [], "binary": [], "numerical": feature_names}
        return {"categorical": [], "binary": feature_names, "numerical": []}
    elif name == "functional_groups":
        count = kwargs.get("count", False)
        if count:
            return {"categorical": [], "binary": [], "numerical": feature_names}
        return {"categorical": [], "binary": feature_names, "numerical": []}
    elif name == "atom_pair":
        count = kwargs.get("count", False)
        if count:
            return {"categorical": [], "binary": [], "numerical": feature_names}
        return {"categorical": [], "binary": feature_names, "numerical": []}
    elif name == "estate":
        variant = kwargs.get("variant", "sum")
        if variant == "bit":
            return {"categorical": [], "binary": feature_names, "numerical": []}
        else:
            return {"categorical": [], "binary": [], "numerical": feature_names}
    elif name == "physiochemical":
        count = kwargs.get("count", False)
        if count:
            return {"categorical": [], "binary": [], "numerical": feature_names}
        return {"categorical": [], "binary": feature_names, "numerical": []}
    elif name == "bcut":
        return {"categorical": [], "binary": [], "numerical": feature_names}
    else:
        raise ValueError(f"Fingerprint feature types for '{name}' is not defined.")
