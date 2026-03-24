# promptcue | Manual test run across all 9 query types
# Maintainer: Informity
#
# Run with:
#   python examples/demo_run.py

import logging
import os

# Suppress HuggingFace and transformers noise before importing promptcue.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

from promptcue import PromptCueAnalyzer, PromptCueConfig  # noqa: E402

QUERIES = [
    ('comparison',       'Compare Aurora and OpenSearch for RAG workloads on AWS'),
    ('lookup',           'What is the default timeout for an AWS Lambda function?'),
    ('troubleshooting',  'Our Lambda function keeps timing out — how do I fix it?'),
    ('coverage',         'Give me a broad overview of AWS networking services'),
    ('recommendation',   'Should we use DynamoDB or RDS for a high-read product catalog?'),
    ('procedure',        'How do I set up a VPC with private subnets step by step?'),
    ('update',           'Any recent updates to the AWS CDK that I should know about?'),
    ('analysis',         'Evaluate this microservices architecture for a multi-region deployment'),
    ('chitchat',         'Hello, how are you doing today?'),
]

SEP   = '─' * 72
WIDTH = 22


def _row(label: str, value: object) -> None:
    print(f'  {label:<{WIDTH}} {value}')


def main() -> None:
    analyzer = PromptCueAnalyzer(PromptCueConfig(
        enable_language_detection    = True,
        enable_linguistic_extraction = True,
        enable_keyword_extraction    = True,
    ))
    analyzer.warm_up()  # pre-load all models; avoids latency on first query

    print(f'\n{"PromptCue Demo Run":^{WIDTH + 50}}')
    print(f'{"— all 9 query types, enrichment enabled —":^{WIDTH + 50}}\n')

    for expected_type, query in QUERIES:
        result = analyzer.analyze(query)

        match = '✓' if result.primary_query_type == expected_type else '✗'

        print(SEP)
        print(f'  Query : {query}')
        print(SEP)
        label_line = f'{result.primary_query_type}  {match}  (expected: {expected_type})'
        _row('primary_query_type',  label_line)
        _row('scope',               result.scope)
        _row('confidence',          f'{result.confidence:.2f}')
        _row('ambiguity_score',     f'{result.ambiguity_score:.2f}')
        _row('classification_basis',result.classification_basis)

        active_routing = [k for k, v in result.routing_hints.items() if v]
        _row('routing_hints (on)',  ', '.join(active_routing) or '—')

        active_actions = [k for k, v in result.action_hints.items() if v]
        _row('action_hints (on)',   ', '.join(active_actions) or '—')

        if result.main_verbs:
            _row('main_verbs',      result.main_verbs)
        if result.noun_phrases:
            _row('noun_phrases',    result.noun_phrases[:4])
        if result.named_entities:
            _row('entities',        [(e.text, e.entity_type) for e in result.entities])
        if result.keywords:
            kw_str = ', '.join(f'{k.text} ({k.weight:.2f})' for k in result.keywords[:4])
            _row('keywords',        kw_str)
        print()

    print(SEP)
    print(f'\n  {len(QUERIES)} queries tested.\n')


if __name__ == '__main__':
    main()
