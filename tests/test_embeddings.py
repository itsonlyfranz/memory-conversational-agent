import math

from memory_agent.embeddings import HashingEmbedder, cosine, tokenize


def test_embedding_is_deterministic_and_normalised(embedder):
    a = embedder.embed("User lives in Berlin")
    assert a == embedder.embed("User lives in Berlin")
    assert len(a) == embedder.dim
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, abs_tol=1e-3)


def test_related_text_scores_above_unrelated(embedder):
    q = embedder.embed("Where do I live?")
    related = embedder.embed("User lives in Berlin")
    unrelated = embedder.embed("User is allergic to peanuts")
    assert cosine(q, related) > cosine(q, unrelated) + 0.2


def test_morphological_variants_share_stems():
    assert tokenize("live") == tokenize("lives") == tokenize("living")
    assert tokenize("like") == tokenize("likes")


def test_empty_text_gives_zero_vector_and_zero_similarity(embedder):
    zero = embedder.embed("the a of")
    assert not any(zero)
    assert cosine(zero, embedder.embed("Berlin")) == 0.0


def test_cosine_handles_mismatched_lengths():
    assert cosine([1.0], [1.0, 0.0]) == 0.0
    assert HashingEmbedder(dim=64).dim == 64
