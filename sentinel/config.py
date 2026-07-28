"""Runtime settings, plus every tunable number Sentinel uses.

Two reasons this file exists:

* No other module touches ``os.environ``, which is what makes "never hardcode
  secrets" checkable by reading one file.
* Every risk threshold and point weight lives here, so the scoring policy can
  be re-tuned without editing a single line of scoring logic.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BugfixDetection(BaseModel):
    """How Sentinel decides that a past commit was fixing a bug.

    Phase 2's SZZ labeling needs exactly this definition to find bug-fixing
    commits, so it lives here as data rather than inside ``git_reader`` — one
    definition, two consumers.

    Keywords are matched on word boundaries, so ``fix`` does not match
    ``prefix``. That is also why the inflections are listed explicitly.
    """

    keywords: tuple[str, ...] = (
        "fix",
        "fixes",
        "fixed",
        "fixing",
        "bug",
        "bugs",
        "bugfix",
        "patch",
        "hotfix",
        "defect",
        "regression",
        "crash",
        "broken",
    )

    #: Regexes for "this commit closes a tracked issue".
    #:
    #: An issue reference only counts when a fixing verb is attached. A bare
    #: `#1234` looks like a bug reference but is really a pull-request number in
    #: any repository that squash-merges: measured on `requests`, treating bare
    #: references as bug-fix evidence labelled 51% of all commits as bug fixes,
    #: against 16% for this stricter pattern. Over-labelling here does not just
    #: add noise, it corrupts every training label downstream.
    issue_patterns: tuple[str, ...] = (
        r"\b(?:clos(?:e|es|ed)|resolv(?:e|es|ed)|fix(?:es|ed)?)\b[^\n]{0,24}?"
        r"(?:#|\b[A-Z][A-Z0-9]+-)\d+",
    )


class DistributionSettings(BaseModel):
    """How the repository's own history is turned into relative thresholds.

    Absolute counts do not travel between repositories: eight past bug fixes is
    remarkable in a young service and unremarkable in a fifteen-year-old library,
    where scoring on absolutes makes almost every change look critical.
    """

    #: A file above this quantile of bug-fix counts is "unusually often fixed".
    very_hot_quantile: float = 0.90
    hot_quantile: float = 0.75
    churn_quantile: float = 0.90

    #: Below this many files with history, percentiles are noise rather than a
    #: distribution, and the absolute thresholds in `RiskWeights` are used.
    min_files: int = 20

    #: Escape hatch: set False to score on absolute thresholds everywhere.
    enabled: bool = True


class RiskWeights(BaseModel):
    """Thresholds and point values for the rule-based score.

    Points are deliberately coarse round numbers: the score has to be arguable
    in a code review, and nobody argues productively about 7.3 points.
    """

    # --- size of the edit to a single file ---
    file_large_lines: int = 300
    file_large_points: int = 25
    file_moderate_lines: int = 80
    file_moderate_points: int = 12

    # --- breadth of the change as a whole ---
    broad_files: int = 10
    broad_files_points: int = 10
    broad_lines: int = 600
    broad_lines_points: int = 10

    # --- bug history of the file (the strongest signal we have) ---
    # These counts are floors. When the repository has enough history, the
    # effective thresholds are raised to its own percentiles instead — see
    # `DistributionSettings` and `risk_rules.calibrate`.
    hot_bugfixes: int = 3
    hot_points: int = 20
    very_hot_bugfixes: int = 8
    very_hot_points: int = 35

    # --- raw churn, independent of bugs ---
    churn_commits: int = 25
    churn_points: int = 10

    # --- test coverage signal ---
    no_test_file_points: int = 20
    stale_test_points: int = 10

    # --- cyclomatic complexity of the changed file ---
    high_ccn: int = 10
    high_ccn_points: int = 12
    very_high_ccn: int = 20
    very_high_ccn_points: int = 22

    # --- how well the author knows this file ---
    new_to_file_points: int = 15
    low_ownership: float = 0.25
    low_ownership_points: int = 8

    # --- when the change is being shipped ---
    weekend_points: int = 10
    late_night_start_hour: int = 22
    late_night_end_hour: int = 6
    late_night_points: int = 12

    # --- score bands (inclusive lower bounds) ---
    medium_band_score: int = 35
    high_band_score: int = 65

    #: Hard ceiling. A score is a triage signal, not an unbounded sum.
    max_score: int = 100


class BlastRadiusSettings(BaseModel):
    """Limits for the dependency graph and the impact walk.

    Every number here exists to stop a true-but-useless answer. A hub module is
    imported by everything, so an uncapped walk reports "this change affects the
    entire repository", which tells a reviewer nothing they can act on.
    """

    enabled: bool = True

    #: How many hops of dependents to follow. Beyond about three, "depends on"
    #: stops meaning "will break".
    max_depth: int = 3

    #: How many names to print per list. Counts are always reported in full.
    max_listed: int = 12

    #: At this many direct dependents, a file is called a hub instead of having
    #: its dependents listed as though the list were reviewable.
    hub_dependents: int = 20

    #: Per-file impact is computed for at most this many changed files, riskiest
    #: first, so `--all` on a large repository stays responsive.
    max_analyzed_files: int = 40

    #: Guards on graph construction, which reads every source file once.
    max_files: int = 5000
    max_bytes_per_file: int = 400_000


class TrainingSettings(BaseModel):
    """Mining, training and evaluation knobs.

    `max_commits` exists because SZZ is the expensive half of this project: it
    runs a line-level `git blame` for every bug-fixing commit, and on a large
    repository that is minutes, not seconds. Bounding the window keeps `train`
    usable; the count actually used is always reported so a truncated run can
    never be mistaken for a full one.
    """

    max_commits: int = 1500

    #: Where the trained model and its metadata live, relative to the repo root.
    model_dir: str = ".sentinel"
    model_file: str = "model.txt"
    metadata_file: str = "model.meta.json"

    #: Fraction of the (time-ordered) commits used for training; the newest
    #: remainder becomes the test set.
    train_fraction: float = 0.75

    # --- LightGBM ---
    # Capacity is deliberately small. A repository yields a few hundred labelled
    # commits with a few dozen positives, and a wide model on that much data
    # memorises the training period instead of learning anything portable.
    seed: int = 42
    num_leaves: int = 7
    learning_rate: float = 0.05
    num_rounds: int = 300
    min_data_in_leaf: int = 30

    #: Stop boosting when a held-out slice stops improving. `num_rounds` is the
    #: ceiling, not the target: the useful number of rounds depends on how much
    #: history a repository has, and hardcoding it either underfits big repos or
    #: overfits small ones.
    early_stopping_rounds: int = 30

    #: Stop on average precision, not ROC-AUC. Positives are rare, and ROC-AUC
    #: is dominated by the huge pool of true negatives — measured on `requests`,
    #: stopping on ROC-AUC halted at 17 rounds and scored 0.407 held-out PR-AUC,
    #: while average precision ran to 43 rounds and scored 0.438.
    early_stopping_metric: str = "average_precision"

    #: Fraction of the available training data used to fit before early
    #: stopping; the newest remainder is the stopping-criterion slice.
    holdout_fraction: float = 0.85

    #: How many SHAP contributions the report shows for a model-scored change.
    top_shap_reasons: int = 6

    #: Parsing a blob per changed file per commit is the slowest part of feature
    #: extraction. Turn off to trade the complexity features for speed.
    include_complexity: bool = True


class Settings(BaseSettings):
    """Configuration read from the process environment, falling back to ``.env``.

    Every LLM field is optional: Sentinel's score comes from git history and
    static analysis, so a missing API key must degrade the explanation only,
    never the score.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Explicit aliases because these two variables use different naming
    # conventions in the deployment environment (one is NVIDIA's, one is ours).
    nvidia_api_key: str | None = Field(default=None, validation_alias="NVIDIA_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    sentinel_llm_api_key: str | None = Field(default=None, validation_alias="SENTINEL_LLM_API_KEY")
    llm_model: str = Field(
        default="meta/llama-3.3-70b-instruct",
        validation_alias="SENTINEL_LLM_MODEL",
    )

    #: NVIDIA's OpenAI-compatible endpoint. Overridable so the same client code
    #: can be pointed at a local mock during tests.
    llm_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias="SENTINEL_LLM_BASE_URL",
    )

    #: Hard limit on how long a scan will wait for prose it does not need. The
    #: explanation is optional polish; blocking a risk report on a third-party
    #: API would make the tool less useful than having no AI at all.
    #:
    #: 60s because a large hosted model under load genuinely takes tens of
    #: seconds for a few hundred tokens — measured against the NVIDIA endpoint,
    #: where `meta/llama-3.3-70b-instruct` exceeded 150s while a 49B model on the
    #: same key answered immediately. When one model is saturated, swapping
    #: `SENTINEL_LLM_MODEL` is the fix; waiting longer is not.
    llm_timeout_seconds: float = Field(
        default=60.0, validation_alias="SENTINEL_LLM_TIMEOUT"
    )
    #: One retry, not the client default of two: a scan should fail fast.
    llm_max_retries: int = Field(default=1, validation_alias="SENTINEL_LLM_MAX_RETRIES")
    llm_max_tokens: int = Field(default=800, validation_alias="SENTINEL_LLM_MAX_TOKENS")
    #: Low, not zero — explanations should be stable, not robotic.
    llm_temperature: float = Field(
        default=0.2, validation_alias="SENTINEL_LLM_TEMPERATURE"
    )

    #: How many of the riskiest files the report lists.
    report_top_files: int = Field(default=10, validation_alias="SENTINEL_REPORT_TOP_FILES")

    bugfix: BugfixDetection = Field(
        default_factory=BugfixDetection,
        validation_alias="SENTINEL_BUGFIX",
    )
    rules: RiskWeights = Field(
        default_factory=RiskWeights,
        validation_alias="SENTINEL_RULES",
    )
    distribution: DistributionSettings = Field(
        default_factory=DistributionSettings,
        validation_alias="SENTINEL_DISTRIBUTION",
    )
    training: TrainingSettings = Field(
        default_factory=TrainingSettings,
        validation_alias="SENTINEL_TRAINING",
    )
    blast: BlastRadiusSettings = Field(
        default_factory=BlastRadiusSettings,
        validation_alias="SENTINEL_BLAST",
    )

    @property
    def api_key(self) -> str | None:
        """The active LLM API key, checked in order of preference."""
        return self.sentinel_llm_api_key or self.nvidia_api_key or self.openai_api_key

    @property
    def llm_enabled(self) -> bool:
        """True when an AI explanation can be attempted at all."""
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, parsed once."""
    return Settings()
