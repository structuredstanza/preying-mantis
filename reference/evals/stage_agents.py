# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import tempfile
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.skills import load_skill_from_dir
from core.config import get_llm_kwargs, ResilientLiteLlm

LiteLlm = ResilientLiteLlm
from core.schemas import ReviewVerdict, CriticVerdict, ReproVerdict, ExecutiveReport
from tools import TOOLS

STAGE_OUTPUT_SCHEMAS = {
    'reviewer': ReviewVerdict,
    'critic': CriticVerdict,
    'reproducer': ReproVerdict,
    'reporter': ExecutiveReport,
}

STAGE_CONFIGS = {
    'planner': {
        'skill': 'mantis-plan',
        # The planner node's tools in workflow.json. Used as the predecessor
        # in research_eval.py's planned arms (fresh vs forked researcher).
        'tools': ['read_file', 'write_file', 'list_files', 'get_summary', 'get_threat_model', 'record_plan', 'get_plan'],
        'instruction': (
            'You are the Security Planner stage in the Mantis review pipeline. Produce a focused, '
            'prioritized vulnerability research plan for the application under the directory '
            "'app/': call list_files('app'), read the entry points and the modules they reach, and "
            'record the plan with record_plan: which files and functions to inspect, for which '
            'vulnerability classes, and why. Do not report findings yourself.'
        ),
    },
    'researcher': {
        'skill': 'mantis-researcher',
        # Same tool list as the researcher node in workflow.json, so the eval
        # measures the production toolset. research_eval.py can replace it
        # via tools_override for toolset A/B runs.
        'tools': ['read_file', 'write_file', 'list_files', 'get_summary', 'get_threat_model', 'get_plan', 'report_findings', 'get_findings', 'find_symbol', 'find_callers', 'find_callees', 'get_function_boundary'],
        'instruction': 'You are the researcher stage in the Mantis review pipeline. Perform an in-depth code audit of the target to discover real, exploitable vulnerabilities. Trace data flow from untrusted sources to dangerous sinks, including flows that cross file boundaries, and report every distinct finding via report_findings with a precise title, filepath, line_numbers, and a description naming the end-to-end mechanism. Prefer the structural navigation tools when they are available: find_symbol to locate definitions, get_function_boundary to read one function instead of a whole file, and find_callers/find_callees to walk call chains across files; fall back to read_file when they report the index is unavailable. Before reporting a finding, verify the claim structurally: confirm the sink is reachable by walking callers, and read the enclosing function rather than reasoning from memory.'
    },
    'deduplicator': {
        'skill': 'mantis-dedupe',
        'tools': ['read_file', 'write_file', 'get_findings', 'report_findings', 'dedupe_findings'],
        'instruction': 'You are the deduplicator stage in the Mantis review pipeline. Retrieve findings using get_findings, analyze duplicate clusters carefully, and merge true duplicates using dedupe_findings. Do NOT merge distinct vulnerabilities.'
    },
    'reviewer': {
        'skill': 'mantis-review',
        'tools': ['read_file', 'get_findings', 'get_threat_model', 'get_summary', 'find_symbol', 'find_callers', 'find_callees', 'get_function_boundary'],
        'instruction': 'You are the reviewer stage in the Mantis review pipeline. Evaluate findings against the active codebase and 13 triage rejection constraints. Check each finding mechanically before trusting it: use get_function_boundary to re-read the enclosing function and confirm the cited guard or missing check is quoted correctly, and find_callers to confirm the sink is actually reachable from the claimed source. Return a ReviewVerdict with route="confirmed" for true vulnerabilities or route="false_positive" for benign/mock/unreachable/hygiene code.'
    },
    'critic': {
        'skill': 'mantis-critic',
        'tools': ['read_file', 'get_findings', 'get_threat_model', 'find_symbol', 'find_callers', 'find_callees', 'get_function_boundary'],
        'instruction': 'You are the critic stage in the Mantis review pipeline. Assess technical exploit viability and untrusted attacker reachability. Use find_callers and get_function_boundary to challenge reachability: a sink with no callers, a guard the finding misquotes, or an input the attacker cannot control are grounds for rejection. Return a CriticVerdict with route="viable" if reachable and exploitable, or route="non_viable" if blocked by framework invariants or missing attack vectors.'
    },
    'calibrator': {
        'skill': 'mantis-calibrate',
        'tools': ['score_risk', 'calibrate_finding', 'get_findings', 'get_threat_model', 'read_file'],
        'instruction': 'You are the calibrator stage in the Mantis review pipeline. Evaluate finding severity, impact, likelihood, and sanity caps. Call calibrate_finding or score_risk to assign risk scores on the 0.1 - 10.0 scale and priority levels.'
    },
    'reproducer': {
        'skill': 'mantis-reproduce',
        'tools': ['run_sandbox', 'run_sandbox_with_evidence', 'get_findings', 'get_threat_model', 'read_file', 'write_file', 'get_function_boundary', 'find_callers'],
        'instruction': 'You are the reproducer stage in the Mantis review pipeline. Execute exploit PoCs in the sandbox. Return ReproVerdict with route="success" if the exploit triggers, or route="failed_repro" if not.'
    },
    'reflector': {
        'skill': 'mantis-reflect',
        'tools': ['read_file', 'write_file', 'get_findings', 'record_learning'],
        'instruction': 'You are the reflector stage in the Mantis review pipeline. Extract strategic cross-pass learnings and call record_learning with appropriate categories and tags.'
    },
    'reporter': {
        'skill': 'mantis-report',
        'tools': ['read_file', 'write_file', 'get_findings', 'get_plan', 'get_threat_model', 'get_summary', 'generate_report'],
        'instruction': 'You are the reporter stage in the Mantis review pipeline. Compile the final executive review packet and call generate_report with an ExecutiveReport object.'
    },
}

def build_stage_agent(stage_name: str, model_id: str = None, reasoning_effort: str = None, tools_override: list = None,
                      include_contents: str = None, before_model_callback: list = None, instruction_suffix: str = None,
                      instruction_override: str = None):
    """Builds one pipeline stage as a standalone ADK agent for benchmarking.

    ``tools_override`` replaces the stage's tool list for this build only
    (used by research_eval.py for toolset A/B runs). Names still resolve
    through tools.TOOLS, so an override cannot inject an unregistered tool.
    ``include_contents`` / ``before_model_callback`` / ``instruction_suffix``
    shape a stage that runs after another in one graph (research_eval.py's
    planned arms): production nodes run with include_contents="none", a
    forked stage carries the fork callback, and a stage that is not first
    gets its briefing in its instruction since it sees no seed message.
    ``instruction_override`` replaces the stage's own instruction (the skill
    guidance still follows it): a per-file campaign's planner must not be
    told to list and read the whole application.
    """
    model_id = model_id or os.environ.get('EVAL_MODEL', 'vertex_ai/gemini-3.8-flash')
    reasoning_effort = reasoning_effort or os.environ.get('EVAL_REASONING_EFFORT', 'low')
    
    _, llm_kwargs = get_llm_kwargs(model_id=model_id, reasoning_effort=reasoning_effort)
    
    cfg = STAGE_CONFIGS.get(stage_name, {'skill': f'mantis-{stage_name}', 'tools': list(TOOLS.keys()), 'instruction': f'You are the {stage_name} stage.'})
    tool_names = tools_override if tools_override is not None else cfg['tools']
    skill_path = Path(__file__).resolve().parent.parent.parent / cfg['skill']
    tools_list = [TOOLS[t] for t in tool_names if t in TOOLS]
    
    instruction = instruction_override if instruction_override else cfg['instruction']
    if skill_path.is_dir() and (skill_path / 'SKILL.md').exists():
        skill_obj = load_skill_from_dir(str(skill_path))
        if hasattr(skill_obj, 'instructions') and skill_obj.instructions:
            instruction = f"{instruction}\n\n[SKILL GUIDANCE]\n{skill_obj.instructions[:1000]}"
    if instruction_suffix:
        instruction = f"{instruction}\n\n{instruction_suffix}"

    output_schema = STAGE_OUTPUT_SCHEMAS.get(stage_name)
    from core.graph_loader import LiteralInstruction, _make_after_tool_guard_callback

    after_tool_cb = _make_after_tool_guard_callback(stage_name, tool_names, model=model_id)

    extra = {}
    if include_contents is not None:
        extra['include_contents'] = include_contents
    if before_model_callback:
        extra['before_model_callback'] = list(before_model_callback)
    return Agent(
        name=stage_name,
        model=LiteLlm(**llm_kwargs),
        instruction=LiteralInstruction(instruction),
        tools=tools_list,
        output_schema=output_schema,
        after_tool_callback=after_tool_cb,
        **extra,
    )


@contextmanager
def eval_run_context(jail_dir=None, db_path=""):
    """Installs a jailed RunContext for eval harnesses that invoke real tools.

    SECURITY (INV-4): the tools in TOOLS are host-boundary enforcing only while a
    RunContext is installed; they derive their jail from it. Eval harnesses must not
    invoke them bare — wrap the run in this context manager so every filesystem and
    sandbox operation is bounded by a disposable scratch jail rather than $CWD.
    """
    from core.context import RunContext, current_run_context

    tmp_dir = None
    if jail_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="mantis_eval_jail_")
        jail_dir = tmp_dir

    ctx, token = install_eval_run_context(jail_dir, db_path)
    try:
        yield ctx
    finally:
        current_run_context.reset(token)
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def install_eval_run_context(jail_dir, db_path="", target_file=""):
    """Installs a jailed RunContext and returns ``(ctx, reset_token)``.

    Used directly by run_eval.py, which owns its temp directories, and by the
    eval_run_context() context manager above. RunContext declares ``jail_dir: str``,
    so the value is normalized here rather than at each call site.
    """
    from core.context import RunContext, current_run_context

    jail_str = str(jail_dir)
    ctx = RunContext(
        jail_dir=jail_str,
        db_path=str(db_path or os.path.join(jail_str, "eval_knowledge.db")),
        target_file=target_file,
        run_id="eval",
    )
    return ctx, current_run_context.set(ctx)
