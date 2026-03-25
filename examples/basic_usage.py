# promptcue | Basic usage example — classify a single query
# Maintainer: Informity
#
# Demonstrates the minimal path: create an analyzer, call analyze(), inspect
# the result.  No optional dependencies required (deterministic path only).
#
# Run with:
#   python examples/basic_usage.py

from __future__ import annotations

from promptcue import PromptCueAnalyzer, PromptCueConfig

# ==============================================================================
# Sample queries — uncomment one to try a different type
# ==============================================================================

# QUERY = 'What is the default retention period for CloudWatch Logs?'           # lookup
# QUERY = 'Our Lambda function keeps timing out — how do I fix it?'             # troubleshooting
# QUERY = 'Give me a broad overview of AWS networking services'                 # coverage
# QUERY = 'Should we use DynamoDB or RDS for a high-read product catalog?'      # recommendation
# QUERY = 'How do I set up a VPC with private subnets step by step?'            # procedure
# QUERY = 'Any recent updates to the AWS CDK that I should know about?'         # update
# QUERY = 'Evaluate this microservices architecture for a multi-region deploy'  # analysis
# QUERY = 'Summarize the key decisions from the design doc'                     # summarization
# QUERY = 'Write a concise README for a developer tool that classifies prompts' # generation
# QUERY = 'I assumed stateless services always scale horizontally — is that right?' # validation
# QUERY = 'Hello, how are you doing today?'                                     # chitchat
QUERY = 'Compare Aurora and OpenSearch for RAG workloads on AWS'                # comparison


def main() -> None:
    """Run a single query through PromptCue and print the full JSON result."""
    analyzer = PromptCueAnalyzer(PromptCueConfig(
        enable_linguistic_extraction = True,
        enable_keyword_extraction    = True,
    ))
    result = analyzer.analyze(QUERY)
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
