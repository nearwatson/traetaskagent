#!/usr/bin/env python3
"""
示例：在 TraeTaskAgent 中使用 MCP Tools

这个示例展示了如何配置和使用 MCP 服务器工具与 TraeTaskAgent。
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from agents.traeag.trae_agent.agent.traetaskagent import TraeTaskAgent


async def example_with_mcp_tools():
    """示例：使用 MCP 工具的 TraeTaskAgent"""
    
    # 1. 配置 MCP 服务器
    # 这里是一个示例配置，实际使用时需要根据你的 MCP 服务器配置
    mcp_servers_config = {
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {}
        },
        "git": {
            "type": "stdio", 
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-git"],
            "env": {}
        },
        # 如果你有 SSE 服务器，可以这样配置：
        # "custom_sse_server": {
        #     "type": "sse",
        #     "url": "http://localhost:8000/sse"
        # }
    }
    
    # 2. 创建 TraeTaskAgent 实例
    agent = TraeTaskAgent(
        servers=mcp_servers_config,
        llm_client=None,  # 会使用默认配置
        config=None,
        db_manager=None
    )
    
    try:
        # 3. 初始化 agent（这会初始化 MCP 服务器）
        await agent.initialize()
        
        # 4. 查看可用的 MCP 工具
        mcp_tools = agent.get_available_mcp_tools()
        print(f"可用的 MCP 工具数量: {len(mcp_tools)}")
        for tool in mcp_tools:
            print(f"- {tool.name}: {tool.description}")
        
        # 5. 处理任务
        task_message = """
        请帮我分析当前目录的文件结构，并创建一个简单的 README.md 文件。
        """
        
        session_id = "example_session_001"
        
        # 设置任务参数
        task_kwargs = {
            'project_path': os.getcwd(),
            'working_dir': os.getcwd(),
            'max_steps': 15
        }
        
        # 执行任务
        response = await agent.process_message(
            message=task_message,
            session_id=session_id,
            **task_kwargs
        )
        
        # 6. 打印结果
        print(f"\n任务执行结果:")
        print(f"- 类型: {response.get('type')}")
        print(f"- 成功: {response.get('success', False)}")
        print(f"- 步骤数: {len(response.get('steps', []))}")
        print(f"- 最终消息: {response.get('final_message')}")
        
        if response.get('patch_available'):
            print(f"- 生成了补丁文件")
            
        # 打印详细步骤信息
        for i, step in enumerate(response.get('steps', []), 1):
            print(f"\n步骤 {i}:")
            print(f"  类型: {step.get('type')}")
            print(f"  内容: {step.get('content', '')[:200]}...")
            if step.get('tool_calls'):
                print(f"  工具调用: {len(step['tool_calls'])} 个")
                for tool_call in step['tool_calls']:
                    print(f"    - {tool_call.get('name')}")
    
    except Exception as e:
        print(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 7. 清理资源
        await agent.cleanup_servers()


def example_mcp_tool_creation():
    """示例：手动创建 MCP 工具"""
    from agents.traeag.trae_agent.tools.mcp_tool import MCPTool
    
    # 创建一个模拟的 MCP 工具定义
    tool_name = "read_file"
    tool_description = "Read contents of a file"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read"
            }
        },
        "required": ["path"]
    }
    
    # 模拟 ServerClient（实际使用时会是真实的客户端）
    class MockServerClient:
        async def call_tool(self, tool_name: str, arguments: dict):
            # 模拟工具调用结果
            class MockResult:
                def __init__(self):
                    self.content = [
                        type('obj', (object,), {
                            'type': 'text',
                            'text': f"File content for {arguments.get('path', 'unknown')}"
                        })()
                    ]
            return MockResult()
    
    mock_client = MockServerClient()
    
    # 创建 MCP 工具
    mcp_tool = MCPTool(
        mcp_tool_name=tool_name,
        mcp_tool_description=tool_description,
        mcp_input_schema=input_schema,
        server_client=mock_client,
        model_provider="openai"
    )
    
    print(f"创建的 MCP 工具:")
    print(f"- 名称: {mcp_tool.name}")
    print(f"- 描述: {mcp_tool.description}")
    print(f"- 参数数量: {len(mcp_tool.parameters)}")
    
    for param in mcp_tool.parameters:
        print(f"  - {param.name} ({param.type}): {param.description}")


if __name__ == "__main__":
    print("=== MCP Tool 创建示例 ===")
    example_mcp_tool_creation()
    
    print("\n=== TraeTaskAgent 与 MCP Tools 集成示例 ===")
    # 注意：这个示例需要实际的 MCP 服务器才能运行
    # asyncio.run(example_with_mcp_tools())
    print("要运行完整示例，请取消注释上面的 asyncio.run() 调用") 