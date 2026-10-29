"""Match concept identities and recover their WordNet hierarchy."""

import re
from collections import defaultdict

from representation.data import read_json
from representation.storage import FeatureStore


def normalize(word):
    """Normalize concept names by lowercasing and standardizing whitespace."""
    return " ".join(word.lower().replace("_", " ").split())


def load_wordnet(path=None):
    """Load WordNet with an optional corpus search directory."""
    import nltk
    from nltk.corpus import wordnet

    if path is not None:
        nltk.data.path.insert(0, str(path))
    wordnet.ensure_loaded()
    return wordnet


def is_wnid(name):
    """Check whether a name is a WordNet noun identifier."""
    return re.fullmatch(r"n\d{8}", name) is not None


def synset_id(synset):
    """Convert a noun synset to its WordNet identifier."""
    return f"n{synset.offset():08d}"


def parent_id(synset):
    """Return the identifier of the first direct hypernym in name order."""
    parents = sorted(synset.hypernyms(), key=lambda node: node.name())
    return synset_id(parents[0]) if parents else None


def make_concept_map(vision, language, wordnet_dir=None):
    """Choose a unique available noun alias for each ImageNet concept."""
    vision, language = FeatureStore(vision), FeatureStore(language)
    words = {normalize(word): word for word in language.ids}
    wn = load_wordnet(wordnet_dir)
    rows, used = [], set()
    for key in sorted(vision.ids):
        if not is_wnid(key):
            continue
        synset = wn.synset_from_pos_and_offset("n", int(key[1:]))
        aliases = [normalize(word) for word in synset.lemma_names()]
        alias = next((word for word in aliases if word in words and word not in used), None)
        if alias is not None:
            rows.append({"id": key, "word": words[alias], "parent": parent_id(synset)})
            used.add(alias)
    if not rows:
        raise ValueError("No ImageNet concepts match the supplied language vocabulary.")
    return rows


def match_items(a, b, level="sample", concept_map=None):
    """Return canonical IDs and row indices for matched samples or concepts."""
    a_idx = {key: i for i, key in enumerate(a.ids)}
    b_idx = {key: i for i, key in enumerate(b.ids)}
    if level == "sample":
        return [
            {"id": key, "index_a": a_idx[key], "index_b": b_idx[key]}
            for key in sorted(a_idx.keys() & b_idx.keys())
        ]
    if concept_map is None:
        raise ValueError("Concept comparisons require --concept-map.")
    rows, used_a, used_b = [], set(), set()
    for row in read_json(concept_map):
        keys = (row["id"], row["word"])
        ka = next((key for key in keys if key in a_idx), None)
        kb = next((key for key in keys if key in b_idx), None)
        if ka is not None and kb is not None:
            if ka in used_a or kb in used_b:
                raise ValueError("Concept mappings must assign unique feature rows in each model.")
            rows.append({"id": row["id"], "index_a": a_idx[ka], "index_b": b_idx[kb]})
            used_a.add(ka)
            used_b.add(kb)
    return sorted(rows, key=lambda row: row["id"])


def make_hierarchy(ids, mapping=None, wordnet_dir=None):
    """Map valid noun concepts to their first direct hypernym."""
    if mapping is not None:
        keys = set(ids)
        parents = {}
        for row in read_json(mapping):
            key = row["id"] if row["id"] in keys else row.get("word")
            if key in keys and row.get("parent"):
                parents[key] = row["parent"]
        return parents
    wn = load_wordnet(wordnet_dir)
    parents = {}
    for key in ids:
        if is_wnid(key):
            synset = wn.synset_from_pos_and_offset("n", int(key[1:]))
        else:
            senses = wn.synsets(normalize(key).replace(" ", "_"), pos=wn.NOUN)
            if not senses:
                continue
            synset = senses[0]
        parent = parent_id(synset)
        if parent is not None:
            parents[key] = parent
    return parents


def group_children(ids, parents, min_children=3):
    """Group concept row indices by parents with enough children."""
    groups = defaultdict(list)
    for i, key in enumerate(ids):
        if key in parents:
            groups[parents[key]].append(i)
    return {
        parent: indices
        for parent, indices in sorted(groups.items())
        if len(indices) >= min_children
    }
