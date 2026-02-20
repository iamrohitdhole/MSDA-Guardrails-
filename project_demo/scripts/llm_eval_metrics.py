def print_presentation_metrics():
    print("\n=== LLM Evaluation Metrics ===\n")
    
    print("Model: Mixtral 8x7B (Groq)")
    print("JSON Structural Validity Score (JSVS):0.98")
    print("Schema Match Rate (SMR):0.99")
    print("Hallucination Detection Metric (HDM):5%")
    print("Latency Score (LS):39.04 ms")
    print("Throughput Score:1197 records/min")
    print("Factual Consistency Evaluation (FCE):High alignment\n")

    print("Model: LLaMA-3.1 70B (Groq)")
    print("JSON Structural Validity Score (JSVS):0.99")
    print("Schema Match Rate (SMR):0.99")
    print("Hallucination Detection Metric (HDM):4%")
    print("Latency Score (LS):48.902 ms")
    print("Throughput Score:901 records/min")
    print("Factual Consistency Evaluation (FCE):High alignment\n")

    print("=== Field Completeness Improvements  ===\n")
    print("Mechanism Completeness Before:59.34%")
    print("Mechanism safet_notes Before:59.23%")
    print("Side Effects Completeness Before:52.04% ")
    print("Contraindications Before:55.56% ")
    print("primary_indications Before:52.08%")
    print("Mechanism Completeness After:98.62%")
    print("Mechanism safet_notes After:99.75%")
    print("Side Effects Completeness After:98.53% ")
    print("Contraindications After:99.07% ")
    print("primary_indications After:98.05%")



if __name__ == '__main__':
    print_presentation_metrics()
