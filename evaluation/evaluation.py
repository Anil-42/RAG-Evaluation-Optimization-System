



def print_evaluation_summary(results, method):
    total_questions = len(results)

    # ------------------------------------------------------------
    # Average Evidence Coverage
    # ------------------------------------------------------------

    average_coverage = sum(
        result["evidence_coverage"]
        for result in results
    ) / total_questions

    # ------------------------------------------------------------
    # Retrieval Pass Rate
    # Pass = at least 50% of evidence points retrieved
    # ------------------------------------------------------------

    passed = sum(
        1
        for result in results
        if result["evidence_coverage"] >= 0.5
    )

    pass_rate = passed / total_questions

    # ------------------------------------------------------------
    # Full Evidence Retrieval Rate
    # 100% of evidence points retrieved
    # ------------------------------------------------------------

    fully_retrieved = sum(
        1
        for result in results
        if result["evidence_coverage"] == 1.0
    )

    full_retrieval_rate = fully_retrieved / total_questions

    # ------------------------------------------------------------
    # Print Results
    # ------------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Method:                  {method}")
    print(f"Questions evaluated:     {total_questions}")
    print(f"Average evidence cover:  {average_coverage:.2%}")
    print(f"Retrieval pass rate:     {pass_rate:.2%}")
    print(f"Full evidence retrieval: {full_retrieval_rate:.2%}")

    print("=" * 60)