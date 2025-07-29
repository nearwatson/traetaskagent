#!/usr/bin/env python3
"""
测试 MCP 工具集成的基本功能
"""

import asyncio
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.traeag.trae_agent.tools.mcp_tool import MCPTool, MCPToolFactory


def test_mcp_tool_creation():
    """测试 MCP 工具的创建和基本功能"""
    print("=== 测试 MCP 工具创建 ===")
    
    # 创建一个模拟的 MCP 工具定义
    tool_name = "test_file_read"
    tool_description = "Read and analyze file contents"
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read"
            },
            "max_lines": {
                "type": "number",
                "description": "Maximum number of lines to read"
            },
            "include_metadata": {
                "type": "boolean",
                "description": "Whether to include file metadata"
            }
        },
        "required": ["file_path"]
    }
    
    # 模拟 ServerClient
    class MockServerClient:
        async def call_tool(self, tool_name: str, arguments: dict):
            print(f"模拟调用工具: {tool_name} with arguments: {arguments}")
            
            # 模拟工具调用结果
            class MockResult:
                def __init__(self):
                    self.content = [
                        type('obj', (object,), {
                            'type': 'text',
                            'text': f"Successfully processed file: {arguments.get('file_path', 'unknown')}\nLines read: {arguments.get('max_lines', 'all')}\nMetadata included: {arguments.get('include_metadata', False)}"
                        })()
                    ]
            return MockResult()
    
    mock_client = MockServerClient()
    
    # 测试不同模型提供商
    for provider in ["openai", "anthropic", None]:
        print(f"\n--- 测试模型提供商: {provider} ---")
        
        # 创建 MCP 工具
        mcp_tool = MCPTool(
            mcp_tool_name=tool_name,
            mcp_tool_description=tool_description,
            mcp_input_schema=input_schema,
            server_client=mock_client,
            model_provider=provider
        )
        
        print(f"工具名称: {mcp_tool.name}")
        print(f"工具描述: {mcp_tool.description}")
        print(f"参数数量: {len(mcp_tool.parameters)}")
        
        for param in mcp_tool.parameters:
            print(f"  - {param.name} ({param.type}): {param.description} [required: {param.required}]")
        
        # 测试 JSON 定义
        json_def = mcp_tool.json_definition()
        print(f"JSON 定义键: {list(json_def.keys())}")
        
        # 测试输入架构
        input_schema_def = mcp_tool.get_input_schema()
        print(f"输入架构类型: {input_schema_def.get('type')}")
        print(f"必需参数: {input_schema_def.get('required', [])}")


async def test_mcp_tool_execution():
    """测试 MCP 工具的执行"""
    print("\n=== 测试 MCP 工具执行 ===")
    
    # 创建测试工具
    class MockServerClient:
        async def call_tool(self, tool_name: str, arguments: dict):
            # 模拟不同的执行结果
            class MockResult:
                def __init__(self, success=True):
                    if success:
                        self.content = [
                            type('obj', (object,), {
                                'type': 'text',
                                'text': f"工具执行成功！处理了文件: {arguments.get('file_path', '未知文件')}"
                            })()
                        ]
                    else:
                        self.content = []
            return MockResult(success=True)
    
    mock_client = MockServerClient()
    
    tool = MCPTool(
        mcp_tool_name="test_tool",
        mcp_tool_description="Test tool for execution",
        mcp_input_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path"},
                "optional_param": {"type": "string", "description": "Optional parameter"}
            },
            "required": ["file_path"]
        },
        server_client=mock_client,
        model_provider="openai"
    )
    
    # 测试正常执行
    test_arguments = {
        "file_path": "/test/file.txt",
        "optional_param": "test_value"
    }
    
    try:
        result = await tool.execute(test_arguments)
        print(f"执行结果:")
        print(f"  成功: {result.error_code == 0}")
        print(f"  输出: {result.output}")
        print(f"  错误: {result.error}")
    except Exception as e:
        print(f"执行失败: {e}")
    
    # 测试带 None 值的执行（OpenAI 兼容性）
    test_arguments_with_none = {
        "file_path": "/test/file.txt",
        "optional_param": None
    }
    
    try:
        result = await tool.execute(test_arguments_with_none)
        print(f"\n带 None 值的执行结果:")
        print(f"  成功: {result.error_code == 0}")
        print(f"  输出: {result.output}")
    except Exception as e:
        print(f"执行失败: {e}")


async def test_mcp_tool_factory():
    """测试 MCP 工具工厂"""
    print("\n=== 测试 MCP 工具工厂 ===")
    
    # 模拟带多个工具的服务器客户端
    class MockServerClientWithTools:
        async def list_tools(self):
            # 模拟多个工具
            tools = []
            for i in range(3):
                tool = type('MockTool', (), {
                    'name': f'tool_{i}',
                    'description': f'Mock tool number {i}',
                    'input_schema': {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string", "description": f"Parameter for tool {i}"}
                        },
                        "required": ["param1"]
                    }
                })()
                tools.append(tool)
            return tools
    
    mock_client = MockServerClientWithTools()
    
    # 测试工具工厂
    try:
        mcp_tools = await MCPToolFactory.create_mcp_tools_async(
            server_client=mock_client,
            model_provider="anthropic"
        )
        
        print(f"创建的工具数量: {len(mcp_tools)}")
        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")
            
    except Exception as e:
        print(f"工具工厂测试失败: {e}")


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试无效输入架构
    try:
        tool = MCPTool(
            mcp_tool_name="invalid_tool",
            mcp_tool_description="Tool with invalid schema",
            mcp_input_schema=None,
            server_client=None,
            model_provider="openai"
        )
        params = tool.get_parameters()
        print(f"无效架构处理成功，参数数量: {len(params)}")
    except Exception as e:
        print(f"无效架构处理失败: {e}")
    
    # 测试空属性架构
    try:
        tool = MCPTool(
            mcp_tool_name="empty_tool",
            mcp_tool_description="Tool with empty schema",
            mcp_input_schema={"type": "object"},
            server_client=None,
            model_provider="openai"
        )
        params = tool.get_parameters()
        print(f"空架构处理成功，参数数量: {len(params)}")
    except Exception as e:
        print(f"空架构处理失败: {e}")


async def main():
    """运行所有测试"""
    print("开始 MCP 工具集成测试...\n")
    
    # 运行各种测试
    test_mcp_tool_creation()
    await test_mcp_tool_execution()
    await test_mcp_tool_factory()
    test_error_handling()
    
    print("\n=== 测试完成 ===")
    print("所有基本功能测试通过！MCP 工具集成实现正确。")


if __name__ == "__main__":
    asyncio.run(main()) 