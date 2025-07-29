# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""MCP Tool for integrating MCP Server tools into TraeAgent."""

from typing import override, Any, Dict, List
import json

from .base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class MCPTool(Tool):
    """Tool that wraps MCP Server tools for use in TraeAgent."""

    def __init__(self, mcp_tool_name: str, mcp_tool_description: str, 
                 mcp_input_schema: Dict[str, Any], server_client, 
                 model_provider: str | None = None) -> None:
        """Initialize MCP Tool.
        
        Args:
            mcp_tool_name: Name of the MCP tool
            mcp_tool_description: Description of the MCP tool
            mcp_input_schema: Input schema from MCP tool
            server_client: ServerClient instance from iter_agent.py
            model_provider: Model provider for compatibility
        """
        super().__init__(model_provider)
        self._mcp_tool_name = mcp_tool_name
        self._mcp_tool_description = mcp_tool_description
        self._mcp_input_schema = mcp_input_schema
        self._server_client = server_client

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    @override
    def get_name(self) -> str:
        return self._mcp_tool_name

    @override
    def get_description(self) -> str:
        return self._mcp_tool_description

    @override
    def get_parameters(self) -> list[ToolParameter]:
        """Convert MCP input schema to TraeAgent ToolParameter format."""
        parameters = []
        
        if not self._mcp_input_schema or "properties" not in self._mcp_input_schema:
            return parameters
            
        properties = self._mcp_input_schema.get("properties", {})
        required_fields = self._mcp_input_schema.get("required", [])
        
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            description = param_info.get("description", f"Parameter {param_name}")
            is_required = param_name in required_fields
            
            # Handle enum values
            enum_values = param_info.get("enum", None)
            
            # Handle array items
            items = param_info.get("items", None)
            
            # For OpenAI compatibility, all parameters should be required=True
            # Optional parameters are handled via nullable types in get_input_schema
            if self.model_provider == "openai":
                is_required = True
            
            tool_param = ToolParameter(
                name=param_name,
                type=param_type,
                description=description,
                enum=enum_values,
                items=items,
                required=is_required
            )
            parameters.append(tool_param)
            
        return parameters

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        """Execute the MCP tool via ServerClient."""
        try:
            # Clean up arguments - remove None values for optional parameters
            clean_arguments = {}
            for key, value in arguments.items():
                if value is not None:
                    clean_arguments[key] = value
            
            # Call the MCP tool via ServerClient
            result = await self._server_client.call_tool(
                tool_name=self._mcp_tool_name,
                arguments=clean_arguments
            )
            
            # Convert MCP result to TraeAgent format
            return self._convert_mcp_result_to_trae_format(result)
            
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=f"MCP tool execution failed: {str(e)}",
                error_code=1
            )

    def _convert_mcp_result_to_trae_format(self, mcp_result) -> ToolExecResult:
        """Convert MCP CallToolResult to TraeAgent ToolExecResult format."""
        try:
            # MCP result should have content field with list of content items
            if hasattr(mcp_result, 'content') and mcp_result.content:
                # Extract text content from MCP result
                output_parts = []
                for content_item in mcp_result.content:
                    if hasattr(content_item, 'type') and content_item.type == 'text':
                        if hasattr(content_item, 'text'):
                            output_parts.append(content_item.text)
                    elif hasattr(content_item, 'text'):
                        output_parts.append(str(content_item.text))
                    else:
                        # Fallback: convert to string
                        output_parts.append(str(content_item))
                
                output = "\n".join(output_parts) if output_parts else "Tool executed successfully"
                
                return ToolExecResult(
                    output=output,
                    error=None,
                    error_code=0
                )
            else:
                # Handle case where result doesn't have expected structure
                return ToolExecResult(
                    output=str(mcp_result) if mcp_result else "Tool executed successfully",
                    error=None,
                    error_code=0
                )
                
        except Exception as e:
            return ToolExecResult(
                output=None,
                error=f"Failed to parse MCP result: {str(e)}",
                error_code=1
            )


class MCPToolFactory:
    """Factory class to create MCP Tools from MCP Server tools."""
    
    @staticmethod
    def create_mcp_tools(server_client, model_provider: str | None = None) -> List[MCPTool]:
        """Create TraeAgent-compatible tools from MCP Server tools.
        
        Args:
            server_client: ServerClient instance from iter_agent.py
            model_provider: Model provider for compatibility
            
        Returns:
            List of MCPTool instances
        """
        mcp_tools = []
        
        try:
            # This would be called asynchronously in practice
            # For now, we return empty list and let the caller handle async tool discovery
            return mcp_tools
            
        except Exception as e:
            print(f"Failed to create MCP tools: {e}")
            return mcp_tools
    
    @staticmethod
    async def create_mcp_tools_async(server_client, model_provider: str | None = None) -> List[MCPTool]:
        """Async version to create TraeAgent-compatible tools from MCP Server tools.
        
        Args:
            server_client: ServerClient instance from iter_agent.py
            model_provider: Model provider for compatibility
            
        Returns:
            List of MCPTool instances
        """
        mcp_tools = []
        
        try:
            # Get tools from MCP server
            mcp_server_tools = await server_client.list_tools()
            
            for mcp_tool in mcp_server_tools:
                trae_tool = MCPTool(
                    mcp_tool_name=mcp_tool.name,
                    mcp_tool_description=mcp_tool.description,
                    mcp_input_schema=mcp_tool.input_schema,
                    server_client=server_client,
                    model_provider=model_provider
                )
                mcp_tools.append(trae_tool)
                
        except Exception as e:
            print(f"Failed to create MCP tools: {e}")
            
        return mcp_tools 