"""New LangGraph Agent.

This module defines a custom graph.
"""

from .graph_screening import graph as graph_screening
from .graph_screening import State as ScreeningState
from .graph_screening import ScreeningResult
from .graph_qa import graph as graph_qa
from .graph_qa import State as QAState
from .graph_qa import QAResult

