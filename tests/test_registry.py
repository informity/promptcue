import pytest

from promptcue.core.registry import PromptCueRegistry, PromptCueTypeDefinition
from promptcue.exceptions import PromptCueRegistryError


def test_registry_loads_defaults() -> None:
    registry    = PromptCueRegistry()
    query_types = registry.get_query_types()
    assert len(query_types) >= 12


def test_registry_contains_expected_labels() -> None:
    registry = PromptCueRegistry()
    labels   = {qt.label for qt in registry.get_query_types()}
    expected = {'coverage', 'lookup', 'comparison', 'recommendation', 'troubleshooting',
                'procedure', 'analysis', 'update', 'chitchat'}
    assert expected.issubset(labels)


def test_get_by_label_found() -> None:
    registry   = PromptCueRegistry()
    definition = registry.get_by_label('comparison')
    assert definition is not None
    assert definition.label == 'comparison'


def test_get_by_label_missing_returns_none() -> None:
    registry = PromptCueRegistry()
    assert registry.get_by_label('nonexistent') is None


def test_routing_hints_sourced_from_yaml() -> None:
    registry   = PromptCueRegistry()
    comparison = registry.get_by_label('comparison')
    assert comparison is not None
    assert comparison.routing_hints.get('needs_reasoning')   is True
    assert comparison.routing_hints.get('needs_retrieval')   is True
    assert comparison.routing_hints.get('needs_current_info') is False


def _defn(**kwargs) -> PromptCueTypeDefinition:
    """Build a minimal valid PromptCueTypeDefinition with sensible defaults for testing."""
    defaults = dict(
        label         = 'x',
        description   = 'desc',
        triggers      = ['trig'],
        examples      = ['ex'],
        routing_hints = {},
        scope         = 'focused',
        action_hints  = {},
    )
    return PromptCueTypeDefinition(**{**defaults, **kwargs})


def test_validation_rejects_missing_label() -> None:
    with pytest.raises(PromptCueRegistryError, match='non-empty string label'):
        PromptCueRegistry(definitions=[_defn(label='')])


def test_validation_rejects_duplicate_label() -> None:
    defn = _defn(label='dup')
    with pytest.raises(PromptCueRegistryError, match='Duplicate label'):
        PromptCueRegistry(definitions=[defn, defn])


def test_validation_rejects_missing_description() -> None:
    with pytest.raises(PromptCueRegistryError, match='non-empty description'):
        PromptCueRegistry(definitions=[_defn(description='')])


def test_validation_rejects_empty_examples() -> None:
    with pytest.raises(PromptCueRegistryError, match='at least one example'):
        PromptCueRegistry(definitions=[_defn(examples=[])])


def test_validation_rejects_non_bool_routing_hint() -> None:
    with pytest.raises(PromptCueRegistryError, match='must be a boolean'):
        PromptCueRegistry(definitions=[_defn(routing_hints={'needs_retrieval': 'yes'})])  # type: ignore[arg-type]


@pytest.mark.parametrize('content', ['', 'null', '- item'])
def test_registry_rejects_invalid_yaml_structure(tmp_path, content: str) -> None:
    p = tmp_path / 'bad.yaml'
    p.write_text(content)
    with pytest.raises(PromptCueRegistryError):
        PromptCueRegistry.from_yaml(p)
