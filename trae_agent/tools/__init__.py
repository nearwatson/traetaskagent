# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tools module for Trae Agent."""

from typing import Type

from .base import Tool, ToolCall, ToolExecutor, ToolResult
from .bash_tool import BashTool
from .ckg_tool import CKGTool
from .edit_tool import TextEditorTool
from .json_edit_tool import JSONEditTool
from .mcp_tool import MCPTool, MCPToolFactory
from .sequential_thinking_tool import SequentialThinkingTool
from .task_done_tool import TaskDoneTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolCall",
    "ToolExecutor",
    "BashTool",
    "TextEditorTool",
    "JSONEditTool",
    "MCPTool",
    "MCPToolFactory",
    "SequentialThinkingTool",
    "TaskDoneTool",
    "CKGTool",
]

tools_registry: dict[str, Type[Tool]] = {
    "bash": BashTool,
    "str_replace_based_edit_tool": TextEditorTool,
    "json_edit_tool": JSONEditTool,
    "sequentialthinking": SequentialThinkingTool,
    "task_done": TaskDoneTool,
    "ckg": CKGTool,
    # MCP tools are handled dynamically via MCPToolFactory
}
