import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from fedotllm.constants import PACKAGE_PATH


class TemplatesConfig(BaseModel):
    code: str
    train: str
    evaluate: str
    predict: str


class AutoMLConfig(BaseModel):
    fix_tries: int = 5
    templates: TemplatesConfig
    predictor_init_kwargs: dict = Field(default_factory=dict)


class CachingConfig(BaseModel):
    enabled: bool = True
    dir_path: str = Field(default=str(Path(PACKAGE_PATH) / "cache"))


class LLMConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    caching: CachingConfig = Field(default_factory=CachingConfig)
    extra_headers: Dict[str, Any] = {}
    completion_params: Dict[str, Any] = {}


class EmbeddingsConfig(BaseModel):
    provider: str = "openai"
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    extra_headers: Dict[str, Any] = {}
    embedding_params: Dict[str, Any] = {}


class LangfuseConfig(BaseModel):
    host: str = "https://cloud.langfuse.com"
    public_key: Optional[str] = None
    secret_key: Optional[str] = None


class EvolveConfig(BaseModel):
    """Runtime knobs for scan -> reader -> verifier -> fixer.

    YAML and `load_config` overrides are the composition surface. `from_env`
    exists for the CLI boundary and existing tests that still set
    `FEDOTLLM_EVOLVE_*`.
    """

    only_files: str = ""
    verified_path: str = ""
    invariants_path: str = "/tmp/fedotllm_invariants.bridge.json"
    journal_path: str = ""
    semantic_mode: str = "automl"
    include_all_files: bool = True
    limit_files: int = 0
    reader_workers: int = 8
    reader_passes: int = 1
    reader_model: str = ""
    # Passes may use different models. Two passes of one model already agreed on
    # only 36 of 244 suspicions; two *different* models overlapped on 3 of 16.
    # A free model is therefore worth a pass of its own rather than a swap.
    reader_models: str = ""
    verifier_model: str = ""
    fixer_model: str = ""
    verifier_workers: int = 6
    limit_leads: int = 0
    max_fixer_candidates: int = 0
    max_unproven_fixes: int = 0
    static_semantic_verify: bool = False
    verify_timeout: int = 180
    verify_attempts: int = 3
    public_attempts: int = 2
    max_chars_per_file: int = 12000
    symbol_view: bool = True
    max_outline_chars: int = 18000
    max_outline_chars_grounded: int = 7000
    max_fix_tries: int = 3
    num_candidates: int = 3
    automl_gate: bool = True
    probe_gate: bool = True
    tuning_gate: bool = True
    value_gate: bool = True
    templates: bool = False
    triage_lint: bool = False
    invariants_first: bool = True
    max_hotspots: int = 40
    max_lint_findings: int = 40
    lint_rules: str = "B006,B007,B008,B020,B904,B905,F821,F841,S113,S608,RUF012"
    lint_cmd: str = ""
    max_tree_files: int = 220
    journal_enabled: bool = True
    journal_limit: int = 25
    max_failed_attempts: int = 4
    dep_chars: int = 1200
    max_deps: int = 8
    inline_chars: int = 500
    max_uses: int = 6
    repo_python: str = ""

    @classmethod
    def from_env(cls, base: "EvolveConfig | None" = None) -> "EvolveConfig":
        data = (base or cls()).model_dump()
        data.update(_evolve_env_overrides())
        return cls.model_validate(data)

    def reader_model_list(self, fallback: str) -> list[str]:
        """One model per reader pass, in order; falls back to a single model."""
        raw = (self.reader_models or "").strip()
        if not raw:
            return [self.reader_model or fallback]
        models = [item.strip() for item in raw.split(",") if item.strip()]
        return models or [self.reader_model or fallback]

    def file_filter(self) -> set[str] | None:
        raw = (self.only_files or "").strip()
        if not raw:
            return None
        import re

        return {
            item.strip().replace("\\", "/")
            for item in re.split(r"[,\n:]+", raw)
            if item.strip()
        }


def _env_flag(name: str) -> str | None:
    if name not in os.environ:
        return None
    return os.environ.get(name, "")


def _as_bool(raw: str, default: bool) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _evolve_env_overrides() -> dict[str, Any]:
    """Read only keys that are actually set, so YAML defaults survive."""
    out: dict[str, Any] = {}

    def text(env_name: str, field: str) -> None:
        raw = _env_flag(env_name)
        if raw is not None:
            out[field] = raw

    def boolean(env_name: str, field: str, default: bool) -> None:
        raw = _env_flag(env_name)
        if raw is not None:
            out[field] = _as_bool(raw, default)

    def integer(env_name: str, field: str) -> None:
        raw = _env_flag(env_name)
        if raw is not None:
            out[field] = int(raw)

    text("FEDOTLLM_EVOLVE_ONLY_FILES", "only_files")
    text("FEDOTLLM_VERIFIED", "verified_path")
    text("FEDOTLLM_INVARIANTS", "invariants_path")
    text("FEDOTLLM_EVOLVE_JOURNAL", "journal_path")
    text("FEDOTLLM_EVOLVE_SEMANTIC_MODE", "semantic_mode")
    text("FEDOTLLM_EVOLVE_READER_MODEL", "reader_model")
    text("FEDOTLLM_EVOLVE_READER_MODELS", "reader_models")
    text("FEDOTLLM_EVOLVE_VERIFIER_MODEL", "verifier_model")
    text("FEDOTLLM_EVOLVE_FIXER_MODEL", "fixer_model")
    text("FEDOTLLM_EVOLVE_LINT_RULES", "lint_rules")
    text("FEDOTLLM_EVOLVE_LINT_CMD", "lint_cmd")
    text("FEDOTLLM_REPO_PYTHON", "repo_python")
    boolean("FEDOTLLM_EVOLVE_INCLUDE_ALL_FILES", "include_all_files", True)
    boolean("FEDOTLLM_EVOLVE_STATIC_SEMANTIC_VERIFY", "static_semantic_verify", False)
    boolean("FEDOTLLM_EVOLVE_SYMBOL_VIEW", "symbol_view", True)
    boolean("FEDOTLLM_EVOLVE_AUTOML_GATE", "automl_gate", True)
    boolean("FEDOTLLM_EVOLVE_PROBE", "probe_gate", True)
    boolean("FEDOTLLM_EVOLVE_PROBE_GATE", "probe_gate", True)
    boolean("FEDOTLLM_EVOLVE_TUNING_GATE", "tuning_gate", True)
    boolean("FEDOTLLM_EVOLVE_VALUE_GATE", "value_gate", True)
    boolean("FEDOTLLM_EVOLVE_TRIAGE", "triage_lint", False)
    boolean("FEDOTLLM_EVOLVE_INVARIANTS_FIRST", "invariants_first", True)
    integer("FEDOTLLM_EVOLVE_LIMIT_FILES", "limit_files")
    integer("FEDOTLLM_EVOLVE_READER_WORKERS", "reader_workers")
    integer("FEDOTLLM_EVOLVE_READER_PASSES", "reader_passes")
    integer("FEDOTLLM_EVOLVE_VERIFIER_WORKERS", "verifier_workers")
    integer("FEDOTLLM_EVOLVE_LIMIT_LEADS", "limit_leads")
    integer("FEDOTLLM_EVOLVE_MAX_FIXER_CANDIDATES", "max_fixer_candidates")
    integer("FEDOTLLM_EVOLVE_MAX_UNPROVEN_FIXES", "max_unproven_fixes")
    integer("FEDOTLLM_EVOLVE_VERIFY_TIMEOUT", "verify_timeout")
    integer("FEDOTLLM_EVOLVE_VERIFY_ATTEMPTS", "verify_attempts")
    integer("FEDOTLLM_EVOLVE_PUBLIC_ATTEMPTS", "public_attempts")
    integer("FEDOTLLM_EVOLVE_MAX_CHARS", "max_chars_per_file")
    integer("FEDOTLLM_EVOLVE_MAX_OUTLINE", "max_outline_chars")
    integer("FEDOTLLM_EVOLVE_MAX_OUTLINE_GROUNDED", "max_outline_chars_grounded")
    integer("FEDOTLLM_EVOLVE_FIX_TRIES", "max_fix_tries")
    integer("FEDOTLLM_EVOLVE_CANDIDATES", "num_candidates")
    integer("FEDOTLLM_EVOLVE_MAX_HOTSPOTS", "max_hotspots")
    integer("FEDOTLLM_EVOLVE_MAX_LINT", "max_lint_findings")
    integer("FEDOTLLM_EVOLVE_MAX_TREE", "max_tree_files")
    integer("FEDOTLLM_EVOLVE_JOURNAL_LIMIT", "journal_limit")
    integer("FEDOTLLM_EVOLVE_MAX_FAILS", "max_failed_attempts")
    integer("FEDOTLLM_EVOLVE_DEP_CHARS", "dep_chars")
    integer("FEDOTLLM_EVOLVE_MAX_DEPS", "max_deps")
    integer("FEDOTLLM_EVOLVE_INLINE_CHARS", "inline_chars")
    integer("FEDOTLLM_EVOLVE_MAX_USES", "max_uses")
    raw_templates = _env_flag("FEDOTLLM_EVOLVE_TEMPLATES")
    if raw_templates is not None:
        out["templates"] = _as_bool(raw_templates, False)
    raw_journal_off = _env_flag("FEDOTLLM_EVOLVE_JOURNAL_OFF")
    if raw_journal_off is not None:
        out["journal_enabled"] = not _as_bool(raw_journal_off, False)
    return out


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    automl: AutoMLConfig = Field(default_factory=AutoMLConfig)
    evolve: EvolveConfig = Field(default_factory=EvolveConfig)
    session_id: Optional[str] = Field(default=None)
