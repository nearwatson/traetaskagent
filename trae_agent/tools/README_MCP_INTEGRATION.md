# MCP 工具集成到 TraeAgent

这个文档说明如何将 MCP (Model Context Protocol) 服务器工具集成到 TraeAgent 中。

## 概述

我们新增了 `MCPTool` 类和 `MCPToolFactory`，使得 TraeAgent 能够：

1. 连接到 MCP 服务器（stdio 和 SSE 类型）
2. 动态加载 MCP 服务器提供的工具
3. 将 MCP 工具无缝集成到 TraeAgent 的工具生态系统中
4. 在任务执行过程中使用 MCP 工具

## 核心组件

### MCPTool 类

`MCPTool` 是一个适配器类，它：
- 继承自 TraeAgent 的 `Tool` 基类
- 将 MCP 工具的输入/输出格式转换为 TraeAgent 兼容的格式
- 处理参数类型映射和验证
- 支持 OpenAI 和其他模型提供商的参数格式

### MCPToolFactory 类

`MCPToolFactory` 负责：
- 从 MCP 服务器动态发现可用工具
- 批量创建 `MCPTool` 实例
- 处理异步工具初始化

### TraeTaskAgent 增强

`TraeTaskAgent` 现在支持：
- MCP 服务器的初始化和清理
- MCP 工具的动态加载
- 标准 TraeAgent 工具和 MCP 工具的统一管理

## 使用方法

### 1. 配置 MCP 服务器

```python
mcp_servers_config = {
    "filesystem": {
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/work/dir"],
        "env": {}
    },
    "git": {
        "type": "stdio", 
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-git"],
        "env": {}
    },
    "custom_sse_server": {
        "type": "sse",
        "url": "http://localhost:8000/sse"
    }
}
```

### 2. 创建和初始化 TraeTaskAgent

```python
from backend.agents.traeag.trae_agent.agent.traetaskagent import TraeTaskAgent

agent = TraeTaskAgent(
    servers=mcp_servers_config,
    llm_client=None,  # 使用默认配置
    config=None,
    db_manager=None
)

# 初始化（这会连接 MCP 服务器并加载工具）
await agent.initialize()
```

### 3. 查看可用工具

```python
# 查看 MCP 工具
mcp_tools = agent.get_available_mcp_tools()
print(f"MCP 工具数量: {len(mcp_tools)}")

for tool in mcp_tools:
    print(f"- {tool.name}: {tool.description}")
```

### 4. 执行任务

```python
task_message = "请帮我分析项目文件结构并创建文档"
session_id = "session_001"

response = await agent.process_message(
    message=task_message,
    session_id=session_id,
    project_path="/path/to/project",
    working_dir="/path/to/project"
)
```

### 5. 清理资源

```python
# 任务完成后清理
await agent.cleanup_servers()
```

## 工具类型映射

MCP 工具的参数类型会自动转换为 TraeAgent 兼容的格式：

| MCP 类型 | TraeAgent 类型 | 说明 |
|----------|----------------|------|
| string   | string         | 字符串类型 |
| number   | number         | 数值类型 |
| boolean  | boolean        | 布尔类型 |
| array    | array          | 数组类型，保留 items 定义 |
| object   | object         | 对象类型 |

## OpenAI 兼容性

为了与 OpenAI 模型兼容，工具在 `model_provider="openai"` 时会：
- 将所有参数标记为 required=True
- 对可选参数使用 nullable 类型（如 `["string", "null"]`）
- 为对象类型添加 `additionalProperties: false`

## 错误处理

集成包含完整的错误处理：
- MCP 服务器连接失败时的优雅降级
- 工具执行错误的详细报告
- 资源清理的异常安全

## 支持的 MCP 服务器类型

1. **stdio 服务器**: 通过标准输入/输出通信
2. **SSE 服务器**: 通过 Server-Sent Events 通信

## 示例 MCP 服务器

常见的 MCP 服务器包括：
- `@modelcontextprotocol/server-filesystem`: 文件系统操作
- `@modelcontextprotocol/server-git`: Git 版本控制
- `@modelcontextprotocol/server-sqlite`: SQLite 数据库操作
- `@modelcontextprotocol/server-brave-search`: Web 搜索

## 故障排除

### 常见问题

1. **MCP 服务器连接失败**
   - 检查命令路径和参数是否正确
   - 确认环境变量配置
   - 查看日志中的详细错误信息

2. **工具调用失败**
   - 验证参数类型和格式
   - 检查 MCP 服务器是否正常运行
   - 确认工具权限设置

3. **性能问题**
   - 考虑并行工具调用设置
   - 优化 MCP 服务器配置
   - 监控内存和 CPU 使用

### 调试技巧

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

查看工具调用详情：
```python
for step in response['steps']:
    if step.get('tool_calls'):
        for call in step['tool_calls']:
            print(f"工具: {call['name']}, 参数: {call['arguments']}")
```

## 扩展和定制

### 添加自定义 MCP 工具处理

可以继承 `MCPTool` 类来添加特定的处理逻辑：

```python
class CustomMCPTool(MCPTool):
    def _convert_mcp_result_to_trae_format(self, mcp_result):
        # 自定义结果转换逻辑
        return super()._convert_mcp_result_to_trae_format(mcp_result)
```

### 添加工具过滤

可以在 `MCPToolFactory` 中添加工具过滤逻辑：

```python
@staticmethod
async def create_filtered_mcp_tools(server_client, tool_filter=None):
    all_tools = await MCPToolFactory.create_mcp_tools_async(server_client)
    if tool_filter:
        return [tool for tool in all_tools if tool_filter(tool)]
    return all_tools
```

## 性能考虑

1. **初始化时间**: MCP 服务器初始化可能需要几秒钟
2. **并发限制**: 某些 MCP 服务器可能有并发调用限制
3. **内存使用**: 大量工具可能增加内存消耗
4. **网络延迟**: SSE 服务器调用受网络影响

## 安全考虑

1. **权限控制**: 确保 MCP 工具只能访问授权的资源
2. **输入验证**: 所有工具参数都会进行类型验证
3. **错误信息**: 错误信息不会泄露敏感系统信息
4. **资源限制**: 考虑设置执行时间和资源使用限制 