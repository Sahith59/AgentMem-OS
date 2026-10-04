"""Forward-only retirement: no network, retries, or historical result rewriting."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.model_policy import RetiredModelError, require_active_model
from benchmarks.english_screen import runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks"))
    import socket
    def denied(*args, **kwargs):
        raise AssertionError('Network forbidden in retirement tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.mark.parametrize('model', [
    'gpt-4o', 'gpt-4o-mini', 'gpt-4o-2024-08-06',
    'gpt-4o-mini-2024-07-18', 'chatgpt-4o-latest',
    'openai/gpt-4o', 'openai:gpt-4o-mini', 'azure/gpt-4o-2024-08-06',
    ' OpenAI/GPT-4o ', 'ft:gpt-4o-mini:org:custom',
])
def test_retired_variants_blocked(model):
    with pytest.raises(RetiredModelError, match='No automatic replacement'):
        require_active_model(model)


@pytest.mark.parametrize('model', ['gpt-5.6-luna', 'openai/gpt-5.6-terra',
                                 'gpt-4.1-mini', 'ollama/qwen2.5:14b'])
def test_active_model_is_not_remapped(model):
    assert require_active_model(model) == model


def test_paid_package_blocks_before_validation_files_or_first_generation(tmp_path, monkeypatch):
    # Even the allowed first generation must not run if its later judge is retired.
    package = {'settings': {'generate': {'model': 'gpt-5.6-luna'},
                            'judge': {'model': 'gpt-4o'}}}
    def forbidden(*a, **kw):
        raise AssertionError('No validation, provider call, or checkpoint write allowed')
    monkeypatch.setattr(runner, 'validate', forbidden)
    out = tmp_path / 'new-run'
    with pytest.raises(RetiredModelError):
        runner.run(package, out, forbidden, 1, 'approval', 'paid')
    assert not out.exists()


def test_provider_blocks_before_sdk_request():
    # Bypass client construction: the boundary itself must reject old models.
    provider = runner.OpenAIProvider.__new__(runner.OpenAIProvider)
    with pytest.raises(RetiredModelError):
        provider({'model': 'openai/gpt-4o-mini'})


def _isolated_function(path, name, class_name=None, **namespace):
    """Execute the actual boundary with no legacy import-time DB/API side effects."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    fn = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    env = {'require_active_model': require_active_model, '__file__': str(ROOT / path), **namespace}
    exec(compile(module, str(ROOT / path), 'exec'), env)
    return env[name]


@pytest.mark.parametrize('path', ['benchmarks/qa_accuracy_eval.py',
                                  'benchmarks/real_baseline_eval.py',
                                  'benchmarks/oracle_ceiling_eval.py'])
def test_legacy_retry_boundaries_stop_before_sdk_import_or_retry(path):
    call = _isolated_function(path, '_chat_with_retry')
    with pytest.raises(RetiredModelError):
        call(None, 'gpt-4o', 'private prompt', 100)


def test_adapter_blocks_before_context_assembly():
    call = _isolated_function('llm/adapters.py', 'send_message', 'UniversalAdapter')
    with pytest.raises(RetiredModelError):
        call(SimpleNamespace(), 'sid', 'query', 'gpt-4o')


def test_extraction_and_supersession_block_before_request_or_retries(monkeypatch):
    monkeypatch.setenv('AGENTMEM_OS_EXTRACTION_API_MODEL', 'gpt-4o-mini')
    monkeypatch.setenv('AGENTMEM_OS_SUPERSESSION_API_MODEL', 'gpt-4o')
    import os
    extract = _isolated_function('llm/consolidation_v2.py', '_llm_api', 'ConsolidationV2', os=os)
    judge = _isolated_function('llm/supersession.py', '_llm', 'SupersessionJudge')
    for call in (extract, judge):
        with pytest.raises(RetiredModelError):
            call(SimpleNamespace(), 'private source text')


def test_summarizer_blocks_before_dependency_or_client_creation(monkeypatch):
    monkeypatch.setenv('MEMNAI_SUMMARIZER_MODEL', 'openai/gpt-4o-mini')
    call = _isolated_function('llm/summarizer.py', '_get_llm', 'SummarizationEngine')
    with pytest.raises(RetiredModelError):
        call(SimpleNamespace(llm=None))


@pytest.mark.parametrize('path,name', [
    ('benchmarks/adapters/mem0_subprocess_worker.py', '_get_memory'),
    ('benchmarks/adapters/langmem_subprocess_worker.py', '_get_manager'),
    ('benchmarks/adapters/graphiti_subprocess_worker.py', '_get_graphiti'),
])
def test_fixed_model_competitors_fail_before_sdk_state(path, name):
    call = _isolated_function(path, name)
    with pytest.raises(ValueError, match='GPT-4o family is retired'):
        call(*([] if name == '_get_graphiti' else ['namespace']))


def test_letta_blocks_before_server_or_reset():
    for name, args in [('setup', []), ('reset', ['namespace'])]:
        call = _isolated_function('benchmarks/adapters/letta_adapter.py', name, 'LettaAdapter')
        with pytest.raises(RetiredModelError):
            call(SimpleNamespace(_model='openai/gpt-4o-mini'), *args)


def test_legacy_mutating_rejudge_blocked_before_database(monkeypatch):
    # Rejudge can delete judgments; retired config must stop before DB imports.
    import os
    monkeypatch.setenv('AGENTMEM_OS_SUPERSESSION_API_MODEL', 'gpt-4o-mini')
    call = _isolated_function('benchmarks/rejudge_luna_corpus.py', 'main', os=os)
    with pytest.raises(ValueError, match='GPT-4o family is retired'):
        call()


def test_api_rejects_explicit_and_aliased_retired_models():
    class HTTPError(Exception):
        def __init__(self, status_code, detail):
            self.status_code, self.detail = status_code, detail
    call = _isolated_function('api/app.py', '_normalise_model',
                             _MODEL_ALIASES={'legacy': 'openai/gpt-4o-mini'},
                             HTTPException=HTTPError)
    for model in ['gpt-4o', 'legacy']:
        with pytest.raises(HTTPError) as error:
            call(model)
        assert error.value.status_code == 400
        assert 'retired' in error.value.detail
    assert call('gpt-5.6-luna') == 'gpt-5.6-luna'


def test_profile_extractor_blocks_before_local_request():
    call = _isolated_function('llm/profile_extractor.py', '_llm', 'ProfileExtractor')
    with pytest.raises(RetiredModelError):
        call(SimpleNamespace(model='gpt-4o'), 'private facts')


@pytest.mark.parametrize('path, name, owner, args', [
    ('llm/consolidation_engine.py', '_generate_cluster_summary', 'SleepConsolidationEngine', [[], 1]),
    ('llm/procedural_memory.py', '_generate_pattern_text', 'ProceduralMemory', ['question', 'answer', [('q', 'a')]]),
])
def test_summary_policy_error_cannot_become_template_fallback(path, name, owner, args):
    def denied(*a, **kw):
        require_active_model('gpt-4o')
    call = _isolated_function(path, name, owner, RetiredModelError=RetiredModelError)
    with pytest.raises(RetiredModelError):
        call(SimpleNamespace(summarizer=SimpleNamespace(compress=denied)), *args)


def test_compress_checks_model_before_entity_downloads():
    call = _isolated_function('llm/summarizer.py', 'compress', 'SummarizationEngine')
    def denied():
        require_active_model('gpt-4o-mini')
    with pytest.raises(RetiredModelError):
        call(SimpleNamespace(_get_llm=denied), [])


def test_cached_summary_model_is_checked():
    call = _isolated_function('llm/summarizer.py', '_get_llm', 'SummarizationEngine')
    with pytest.raises(RetiredModelError):
        call(SimpleNamespace(llm=SimpleNamespace(model='gpt-4o'), model_name='llama3.1'))


@pytest.mark.parametrize('name', ['judge', 'judge_api', 'main'])
def test_cross_language_evaluator_stops_before_retries_or_report_write(name):
    call = _isolated_function('benchmarks/xling_merge_judge_eval.py', name, MODEL='gpt-4o')
    with pytest.raises(ValueError, match='GPT-4o family is retired'):
        call(*([] if name == 'main' else ['a', 'b']))
