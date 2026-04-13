"""SWIFTbot: Subgroup Workflow for Identifying Fault-tolerant T-extensions.

(The 'bot' is flavor, not an acronym.)

A reusable agent-orchestrated pipeline that, for a given qudit dimension d,
discovers candidate finite subgroups C ⊂ SU(d), proposes diagonal and
non-diagonal extensions T, evaluates the QCO efficiency bound Q_T, and checks
QEC-code + magic-state-distillation compatibility.

Numerics are deterministic; LLM judgment is used only at stage boundaries.
"""

__version__ = "0.1.0"
