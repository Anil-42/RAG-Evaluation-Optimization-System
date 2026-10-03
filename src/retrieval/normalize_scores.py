
def normalize_scores(scores):
    maxval=max(scores)
    minval=min(scores)

    if maxval == minval:
        return [0.0] * len(scores)

    scores = [
        (score - minval) / (maxval - minval)
        for score in scores
    ]

    return scores


