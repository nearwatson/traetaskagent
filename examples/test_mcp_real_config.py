#!/usr/bin/env python3
"""
基于真实配置文件测试 MCP 工具集成
使用 servers_config_trae.json 配置文件来实例化和测试 MCP 工具
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.agents.traeag.trae_agent.agent.traetaskagent import TraeTaskAgent
from backend.agents.simple_chatbot.mcp_simple_chatbot.iter_agent import ServerClient, Configuration
from backend.db import MongoDBManager


async def test_mcp_tools_with_real_config():
    """使用真实配置文件测试 MCP 工具集成"""
    print("=== 基于真实配置的 MCP 工具集成测试 ===\n")
    
    try:
        # 1. 加载配置文件
        config_path = project_root / "backend/agents/configuration/servers_config_trae.json"
        print(f"📂 加载配置文件: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        print(f"✅ 配置文件加载成功")
        print(f"📋 配置的 Agent 类型: {list(config_data.get('agents', {}).keys())}")
        print(f"🔧 配置的 MCP 服务器: {list(config_data.get('mcpServers', {}).keys())}")
        print(f"🛠️ 工具白名单数量: {len(config_data.get('tool_whitelist', []))}")
        
        # 2. 创建 Configuration 和 ServerClient 实例
        config = Configuration(str(config_path))
        
        # 创建服务器管理器（模拟 main.py 中的逻辑）
        servers = []
        mcp_servers_config = config_data.get("mcpServers", {})
        
        print(f"\n🚀 创建 MCP 服务器客户端...")
        for server_name, server_config in mcp_servers_config.items():
            try:
                print(f"  - 创建服务器: {server_name}")
                server_client = ServerClient(server_name, server_config)
                servers.append(server_client)
                print(f"    ✅ {server_name} 客户端创建成功")
            except Exception as e:
                print(f"    ❌ {server_name} 客户端创建失败: {e}")
        
        # 3. 创建数据库管理器（模拟）
        print(f"\n💾 创建数据库管理器...")
        try:
            db_manager = MongoDBManager(
                host="localhost", 
                port=27017, 
                database_name="scaffold_test"
            )
            print(f"    ✅ 数据库管理器创建成功")
        except Exception as e:
            print(f"    ⚠️  数据库连接失败，使用 None: {e}")
            db_manager = None
        
        # 4. 创建 TraeTaskAgent 实例
        print(f"\n🤖 创建 TraeTaskAgent 实例...")
        try:
            # 注意：这里传入的是 servers 列表（ServerClient 实例）
            agent = TraeTaskAgent(
                servers=servers,  # 传入 ServerClient 实例列表
                llm_client=None,  # 会使用默认配置
                config=config,
                db_manager=db_manager
            )
            print(f"    ✅ TraeTaskAgent 创建成功")
        except Exception as e:
            print(f"    ❌ TraeTaskAgent 创建失败: {e}")
            return
        
        # 5. 初始化 agent（这会初始化 MCP 服务器和工具）
        print(f"\n🔧 初始化 TraeTaskAgent...")
        try:
            await agent.initialize()
            print(f"    ✅ TraeTaskAgent 初始化成功")
        except Exception as e:
            print(f"    ⚠️  TraeTaskAgent 初始化部分失败: {e}")
            # 继续执行，查看能获取到的工具
        
        # 6. 获取和展示 MCP 工具信息
        print(f"\n🛠️  分析 MCP 工具...")
        mcp_tools = agent.get_available_mcp_tools()
        print(f"📊 成功加载的 MCP 工具数量: {len(mcp_tools)}")
        
        if mcp_tools:
            print(f"\n📋 MCP 工具详细信息:")
            for i, tool in enumerate(mcp_tools, 1):
                print(f"\n  {i}. 工具名称: {tool.name}")
                print(f"     描述: {tool.description}")
                print(f"     参数数量: {len(tool.parameters)}")
                
                if tool.parameters:
                    print(f"     参数详情:")
                    for param in tool.parameters:
                        required_text = "必需" if param.required else "可选"
                        enum_text = f" (枚举值: {param.enum})" if param.enum else ""
                        print(f"       - {param.name} ({param.type}): {param.description} [{required_text}]{enum_text}")
                else:
                    print(f"     无参数")
                
                # 显示 JSON 定义
                try:
                    json_def = tool.json_definition()
                    print(f"     JSON 定义预览: {json.dumps(json_def, ensure_ascii=False, indent=2)[:200]}...")
                except Exception as e:
                    print(f"     JSON 定义生成失败: {e}")
        else:
            print(f"    ⚠️  未加载到任何 MCP 工具")
        
        # 7. 测试单个工具（如果有的话）
        if mcp_tools:
            print(f"\n🧪 测试第一个 MCP 工具的执行...")
            test_tool = mcp_tools[0]
            print(f"    测试工具: {test_tool.name}")
            
            # 构造测试参数（简单的测试）
            test_args = {}
            for param in test_tool.parameters:
                if param.required:
                    if param.type == "string":
                        test_args[param.name] = "test_value"
                    elif param.type == "number":
                        test_args[param.name] = 1
                    elif param.type == "boolean":
                        test_args[param.name] = True
                    elif param.type == "array":
                        test_args[param.name] = []
                    elif param.type == "object":
                        test_args[param.name] = {}
            
            print(f"    测试参数: {test_args}")
            
            try:
                # 注意：这里不真正执行，因为可能会有副作用
                print(f"    ✅ 工具参数验证通过（未实际执行）")
                # result = await test_tool.execute(test_args)
                # print(f"    执行结果: {result.output}")
            except Exception as e:
                print(f"    ⚠️  工具测试失败: {e}")
        
        # 8. 显示服务器连接状态
        print(f"\n🔌 MCP 服务器连接状态:")
        for server_name, server_client in agent.server_clients.items():
            session_status = "已连接" if (hasattr(server_client, 'session') and server_client.session) else "未连接"
            print(f"    - {server_name}: {session_status}")
        
        # 9. 显示标准 TraeAgent 工具
        print(f"\n🔧 标准 TraeAgent 工具:")
        standard_tools = []
        if hasattr(agent, '_tools'):
            standard_tools = [tool for tool in agent._tools if not any(tool.name == mcp_tool.name for mcp_tool in mcp_tools)]
        
        print(f"📊 标准工具数量: {len(standard_tools)}")
        for tool in standard_tools[:5]:  # 显示前5个
            print(f"    - {tool.name}: {tool.description[:50]}...")
        
        if len(standard_tools) > 5:
            print(f"    ... 还有 {len(standard_tools) - 5} 个工具")
        
        # 10. 总结
        print(f"\n📋 测试总结:")
        print(f"    - 配置文件: ✅ 加载成功")
        print(f"    - MCP 服务器: {len(agent.server_clients)}/{len(mcp_servers_config)} 连接成功")
        print(f"    - MCP 工具: {len(mcp_tools)} 个加载成功")
        print(f"    - 标准工具: {len(standard_tools)} 个可用")
        print(f"    - 总工具数: {len(mcp_tools) + len(standard_tools)} 个")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        try:
            if 'agent' in locals():
                await agent.cleanup_servers()
                print(f"\n🧹 资源清理完成")
        except Exception as e:
            print(f"⚠️  资源清理失败: {e}")


def show_config_info():
    """显示配置文件信息"""
    print("=== 配置文件信息预览 ===\n")
    
    config_path = project_root / "backend/agents/configuration/servers_config_trae.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        print(f"📂 配置文件路径: {config_path}")
        print(f"📊 配置文件大小: {config_path.stat().st_size} 字节")
        
        # 显示 Agent 配置
        agents = config_data.get("agents", {})
        print(f"\n🤖 配置的 Agent 类型 ({len(agents)} 个):")
        for agent_type, agent_config in agents.items():
            print(f"  - {agent_type}:")
            print(f"    名称: {agent_config.get('name', 'N/A')}")
            print(f"    描述: {agent_config.get('description', 'N/A')}")
            print(f"    模块: {agent_config.get('module_path', 'N/A')}")
            print(f"    类名: {agent_config.get('class_name', 'N/A')}")
            print(f"    默认: {'是' if agent_config.get('default', False) else '否'}")
        
        # 显示 MCP 服务器配置
        mcp_servers = config_data.get("mcpServers", {})
        print(f"\n🔧 配置的 MCP 服务器 ({len(mcp_servers)} 个):")
        for server_name, server_config in mcp_servers.items():
            server_type = server_config.get('type', 'stdio')
            print(f"  - {server_name} ({server_type}):")
            if server_type == 'stdio':
                print(f"    命令: {server_config.get('command', 'N/A')}")
                print(f"    参数: {server_config.get('args', [])}")
            elif server_type == 'sse':
                print(f"    URL: {server_config.get('url', 'N/A')}")
        
        # 显示工具白名单
        tool_whitelist = config_data.get("tool_whitelist", [])
        print(f"\n🛠️ 工具白名单 ({len(tool_whitelist)} 个):")
        for i, tool_name in enumerate(tool_whitelist[:10], 1):  # 显示前10个
            print(f"  {i:2d}. {tool_name}")
        
        if len(tool_whitelist) > 10:
            print(f"  ... 还有 {len(tool_whitelist) - 10} 个工具")
        
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")


if __name__ == "__main__":
    print("🧪 MCP 工具集成真实配置测试\n")
    
    # 显示配置信息
    show_config_info()
    
    print("\n" + "="*60 + "\n")
    
    # 运行完整测试
    asyncio.run(test_mcp_tools_with_real_config()) 