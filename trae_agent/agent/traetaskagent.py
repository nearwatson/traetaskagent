"""TraeTaskAgent - Trae Agent adapted for web backend integration."""

import asyncio
import json
import os
from dotenv import load_dotenv
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, override

from .trae_agent import TraeAgent
from ..utils.config import Config, load_config
from ..utils.llm_basics import LLMMessage
from .agent_basics import AgentExecution
from ..tools.mcp_tool import MCPToolFactory

from logger import logger

def load_env():
    # project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    # dotenv_path = os.path.join(project_root, '.env')
    # load_dotenv(dotenv_path)
    load_dotenv()

class TraeTaskAgent(TraeAgent):
    """TraeTaskAgent - TraeAgent adapted for web backend integration."""

    @staticmethod
    def _create_default_trae_config() -> Config:
        """Create default Trae Config."""
        load_env()
        # project_root = Path(__file__).parent.parent.parent.parent.parent.parent
        config_path = "agents/traeag/trae_config.json"
        with open(config_path, 'r') as f:
            trae_config_dict = json.load(f)
        for provider, prov_config in trae_config_dict['model_providers'].items():
            key_name = provider.upper() + '_API_KEY'
            if 'api_key' in prov_config:
                prov_config['api_key'] = os.getenv(key_name, prov_config['api_key'])
        return Config(trae_config_dict)

    def __init__(self, servers, llm_client, config, db_manager):
        """Initialize TraeTaskAgent with server management."""
        # TraeTaskAgent 总是使用自己的专用配置
        # 因为外部的 config 和 llm_client 可能不兼容 TraeAgent 的接口需求
        trae_config = self._create_default_trae_config()
        
        # 调用父类初始化，使用专用配置和 llm_client=None
        super().__init__(trae_config, llm_client=None)
        
        # 设置实例变量
        self.servers = servers
        self.db_manager = db_manager
        self.current_session_id = None
        
        # 新增：WebSocket连接和实时反馈相关
        self.websocket_manager = None
        self.websocket_connection = None
        self.step_callback = None
        
        # MCP 工具相关
        self.mcp_tools = []
        self.server_clients = {}  # 存储 MCP 服务器客户端实例
        
        # 导入必需模块
        import tempfile
        os = __import__('os')
        
        # 创建 temp dir
        self.temp_dir = tempfile.mkdtemp(prefix="trae_agent_")
        
        # Initialize trajectory recording
        self.trajectory_path = None

    
    async def initialize(self):
        """Initialize the agent and MCP servers."""
        # Initialize MCP servers if available
        await self._initialize_mcp_servers()
    
    async def cleanup_servers(self):
        """Cleanup resources."""
        # Cleanup MCP servers
        await self._cleanup_mcp_servers()
    
    def set_websocket_callback(self, websocket_manager, websocket_connection, session_id: str):
        """设置WebSocket连接和回调，用于实时反馈"""
        self.websocket_manager = websocket_manager
        self.websocket_connection = websocket_connection
        self.current_session_id = session_id
        # 获取稳健的连接对象
        self.robust_connection = None
        if hasattr(websocket_manager, 'session_connections'):
            self.robust_connection = websocket_manager.session_connections.get(session_id)
        if not self.robust_connection and hasattr(websocket_manager, 'connections'):
            # 尝试从用户连接中获取
            for conn in websocket_manager.connections.values():
                if conn.websocket == websocket_connection:
                    self.robust_connection = conn
                    break
    
    async def _send_step_update(self, step_data: dict):
        """发送步骤更新到前端"""
        success = False
        
        try:
            import json
            from datetime import datetime
            
            # 构造实时步骤更新消息
            update_message = {
                "type": "step_update",
                "session_id": self.current_session_id,
                "step_data": step_data,
                "timestamp": datetime.now().isoformat()
            }
            
            message_json = json.dumps(update_message, ensure_ascii=False)
            
            # 优先使用稳健连接
            if self.robust_connection:
                success = await self.robust_connection.send_message(message_json)
                if success:
                    print(f"通过稳健连接发送步骤更新: {step_data.get('description', 'unknown')}")
                else:
                    print(f"稳健连接发送失败，连接状态: {self.robust_connection.state}")
            
            # 如果稳健连接失败，尝试传统方式（向后兼容）
            if not success and self.websocket_manager and self.websocket_connection:
                try:
                    success = await self.websocket_manager.send_personal_message(
                        message_json, 
                        self.websocket_connection
                    )
                    if success:
                        print(f"通过传统连接发送步骤更新: {step_data.get('description', 'unknown')}")
                except Exception as e:
                    print(f"传统连接发送也失败: {e}")
            
            if not success:
                print(f"所有连接方式都失败，步骤更新丢失: {step_data.get('description', 'unknown')}")
                
        except Exception as e:
            print(f"发送步骤更新时发生异常: {e}")
            import traceback
            traceback.print_exc()
    
    async def process_message(self, message: str, session_id: str, **kwargs) -> Dict:
        """
        Process a message and return structured response compatible with web-backend .
        
        Args:
            message: User message
            session_id: Session ID
            **kwargs: Additional parameters (project_path, file_path, trajectory_file, etc.)
            
        Returns:
            Dict: Structured response compatible with existing web backend
        """
        try:
            logger.info(f"TraeTaskAgent 开始处理消息: {message[:100]}...")
            
            # 存储当前session_id以供中断检查使用
            self.current_session_id = session_id
            
            # 从kwargs中提取TraeTaskAgent特定参数
            project_path = kwargs.get('project_path')
            file_path = kwargs.get('file_path')
            working_dir = kwargs.get('working_dir')
            max_steps = kwargs.get('max_steps', 20)
            must_patch = kwargs.get('must_patch', False)
            patch_path = kwargs.get('patch_path')
            trajectory_file = kwargs.get('trajectory_file')
            
            # Extract optional parameters from message metadata
            message_data = kwargs.get('message_data', {})
            metadata = message_data.get('metadata', {})
            
            # Extract task-specific parameters
            project_path = kwargs.get('project_path') or metadata.get('project_path', os.getcwd())
            file_path = kwargs.get('file_path') or metadata.get('file_path')
            trajectory_file = kwargs.get('trajectory_file') or metadata.get('trajectory_file')
            must_patch = kwargs.get('must_patch') or metadata.get('must_patch', False)
            patch_path = kwargs.get('patch_path') or metadata.get('patch_path')
            working_dir = kwargs.get('working_dir') or metadata.get('working_dir', project_path)
            max_steps = kwargs.get('max_steps') or metadata.get('max_steps', 20)
            
            # Parse message for task parameters if they're embedded
            task_content = self._parse_task_content(message, metadata)
            
            # If file_path is provided, read task from file
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    task = f.read()
            else:
                task = task_content
            
            # Setup trajectory recording with temp directory
            if trajectory_file:
                # 如果trajectory_file不是绝对路径，需要特殊处理
                if not os.path.isabs(trajectory_file):
                    # 检查是否只是一个目录名（不包含文件扩展名）
                    if not trajectory_file.endswith('.json'):
                        # 如果只是目录名，在该目录下创建一个唯一的json文件
                        trajectory_dir = os.path.join(working_dir, trajectory_file)
                        os.makedirs(trajectory_dir, exist_ok=True)
                        trajectory_filename = f"trajectory_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        trajectory_file = os.path.join(trajectory_dir, trajectory_filename)
                    else:
                        # 如果是文件名，放在working_dir下
                        trajectory_file = os.path.join(working_dir, trajectory_file)
                # 如果是绝对路径，直接使用，但要确保是json文件
                elif not trajectory_file.endswith('.json'):
                    # 如果是绝对路径但指向目录，在该目录下创建文件
                    if os.path.isdir(trajectory_file):
                        trajectory_filename = f"trajectory_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        trajectory_file = os.path.join(trajectory_file, trajectory_filename)
                    else:
                        # 如果是绝对路径但不是json文件，添加扩展名
                        trajectory_file += f"_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # 确保目录存在
                trajectory_dir = os.path.dirname(trajectory_file)
                os.makedirs(trajectory_dir, exist_ok=True)
                
                trajectory_path = self.setup_trajectory_recording(trajectory_file)
            else:
                # Generate default trajectory file in temp directory
                os.makedirs(os.path.join(tempfile.gettempdir(), "trae_trajectories"), exist_ok=True)
                default_trajectory = os.path.join(
                    tempfile.gettempdir(), 
                    "trae_trajectories",
                    f"trajectory_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                trajectory_path = self.setup_trajectory_recording(default_trajectory)
            
            # Validate and setup working directory
            original_cwd = os.getcwd()
            logger.info(f"original_cwd: {original_cwd}")
            if working_dir and os.path.isdir(working_dir):
                logger.info(f"chdir to {working_dir}")
                os.chdir(working_dir)
                actual_project_path = working_dir
            else:
                actual_project_path = project_path
            
            try:
                # Prepare task arguments
                task_args = {
                    "project_path": actual_project_path,
                    "issue": task,
                    "must_patch": "true" if must_patch else "false",
                }
                
                if patch_path:
                    if not os.path.isabs(patch_path):
                        patch_path = os.path.join(tempfile.gettempdir(), "trae_patches", patch_path)
                        os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                    task_args["patch_path"] = patch_path
                
                # Update max_steps if provided
                if max_steps and max_steps != 20:
                    self._max_steps = max_steps
                
                # Create new task with MCP tools
                await self.new_task_with_mcp_tools(task, task_args)
                
                # Execute task with real-time feedback
                execution: AgentExecution = await self.execute_task_with_realtime_feedback()
                
                # Convert execution result to web backend format
                response = self._convert_execution_to_web_response(execution, trajectory_path, task_args)
                
                return response
                
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return {
                "type": "error_response",
                "steps": [
                    {
                        "step": 1,
                        "type": "error",
                        "content": f"TraeAgent 执行时出现错误: {str(e)}",
                        "description": "系统错误",
                        "error_detail": error_detail
                    }
                ],
                "final_message": f"抱歉，执行任务时出现了错误: {str(e)}",
                "has_tool_calls": False,
                "waiting_for_approval": False,
                "error": str(e),
                "trajectory_file": trajectory_file,
                "error_detail": error_detail
            }
    
    async def execute_task_with_realtime_feedback(self) -> "AgentExecution":
        """Execute task with real-time step feedback to frontend."""
        import time
        from .agent_basics import AgentExecution, AgentStep, AgentState
        from ..utils.llm_basics import LLMMessage
        
        start_time = time.time()
        execution = AgentExecution(task=self._task, steps=[])
        
        try:
            messages = self._initial_messages
            step_number = 1

            while step_number <= self._max_steps:
                # 检查中断标志
                if hasattr(self, 'current_session_id') and self.current_session_id:
                    interrupt_flag = self.db_manager.get_session_interrupt_flag(self.current_session_id)
                    if interrupt_flag:
                        logger.info(f"检测到session {self.current_session_id} 的中断标志，停止任务执行")
                        
                        # 清除中断标志
                        self.db_manager.clear_session_interrupt_flag(self.current_session_id)
                        
                        # 发送中断状态更新
                        await self._send_step_update({
                            "step": step_number,
                            "type": "interrupted",
                            "content": "任务已被用户中断",
                            "description": f"步骤 {step_number}: 任务执行被用户中断",
                            "status": "interrupted"
                        })
                        
                        # 设置执行结果为中断状态
                        execution.final_result = f"Task interrupted by user at step {step_number}"
                        execution.success = False
                        break

                step = AgentStep(step_number=step_number, state=AgentState.THINKING)

                try:
                    # 发送"思考中"状态更新
                    await self._send_step_update({
                        "step": step_number,
                        "type": "thinking",
                        "content": f"步骤 {step_number}: 正在思考...",
                        "description": f"步骤 {step_number}: AI正在分析和思考",
                        "status": "in_progress"
                    })

                    step.state = AgentState.THINKING
                    self._update_cli_console(step)

                    # Get LLM response
                    llm_response = self._llm_client.chat(
                        messages, self._model_parameters, self._tools
                    )
                    step.llm_response = llm_response

                    # 发送LLM响应更新
                    await self._send_step_update({
                        "step": step_number,
                        "type": "thinking",
                        "content": llm_response.content,
                        "description": f"步骤 {step_number}: AI思考完成",
                        "status": "completed"
                    })

                    self._update_cli_console(step)
                    self._update_llm_usage(llm_response, execution)

                    if self.llm_indicates_task_completed(llm_response):
                        if self._is_task_completed(llm_response):
                            self._llm_complete_response_task_handler(
                                llm_response, step, execution, messages
                            )
                            
                            # 发送完成状态更新
                            await self._send_step_update({
                                "step": step_number,
                                "type": "complete",
                                "content": "任务执行完成",
                                "description": f"步骤 {step_number}: 任务成功完成",
                                "status": "completed"
                            })
                            break
                        else:
                            step.state = AgentState.THINKING
                            messages = [
                                LLMMessage(role="user", content=self.task_incomplete_message())
                            ]
                    else:
                        # Check if the response contains a tool call
                        tool_calls = llm_response.tool_calls
                        
                        # Process the tool calls and get results
                        messages = await self._tool_call_handler_with_feedback(tool_calls, step, step_number, llm_response)

                    # Record agent step
                    self._record_handler(step, messages)
                    self._update_cli_console(step)

                    execution.steps.append(step)
                    step_number += 1

                except Exception as e:
                    step.state = AgentState.ERROR
                    step.error = str(e)

                    # 发送错误状态更新
                    await self._send_step_update({
                        "step": step_number,
                        "type": "error",
                        "content": f"执行错误: {str(e)}",
                        "description": f"步骤 {step_number}: 执行出现错误",
                        "status": "error"
                    })

                    self._update_cli_console(step)
                    self._record_handler(step, messages)
                    
                    execution.steps.append(step)
                    break

            if step_number > self._max_steps and not execution.success:
                execution.final_result = "Task execution exceeded maximum steps without completion."
                
                # 发送超时状态更新
                await self._send_step_update({
                    "step": step_number,
                    "type": "failed",
                    "content": "任务执行超过最大步数限制",
                    "description": "任务未能在规定步数内完成",
                    "status": "failed"
                })

        except Exception as e:
            execution.final_result = f"Agent execution failed: {str(e)}"
            
            # 发送失败状态更新
            await self._send_step_update({
                "step": step_number if 'step_number' in locals() else 1,
                "type": "error",
                "content": f"代理执行失败: {str(e)}",
                "description": "系统级错误",
                "status": "error"
            })

        execution.execution_time = time.time() - start_time

        # Display final summary
        if 'step' in locals():
            self._update_cli_console(step)

        return execution
    
    async def _tool_call_handler_with_feedback(self, tool_calls, step, step_number, llm_response=None):
        """带实时反馈的工具调用处理器"""
        from ..utils.llm_basics import LLMMessage
        from .agent_basics import AgentState
        import json
        
        messages = []
        if not tool_calls or len(tool_calls) <= 0:
            messages = [
                LLMMessage(
                    role="user",
                    content="It seems that you have not completed the task.",
                )
            ]
            return messages
        
        # Add assistant message with tool calls to conversation history first
        # This ensures proper tool_use -> tool_result pairing for Anthropic models via OpenRouter
        if tool_calls and llm_response:
            for tool_call in tool_calls:
                tool_call_message = LLMMessage(
                    role="assistant",
                    content=llm_response.content or "",
                    tool_call=tool_call
                )
                messages.append(tool_call_message)

        step.state = AgentState.CALLING_TOOL
        step.tool_calls = tool_calls
        self._update_cli_console(step)

        if self._model_parameters.parallel_tool_calls:
            tool_results = await self._tool_caller.parallel_tool_call(tool_calls)
        else:
            tool_results = await self._tool_caller.sequential_tool_call(tool_calls)
        
        step.tool_results = tool_results
        self._update_cli_console(step)
        
        # 发送工具执行完成更新，包含完整的工具调用和结果信息
        await self._send_step_update({
            "step": step_number,
            "type": "tool_result",
            "content": f"工具执行完成，共 {len(tool_results)} 个结果",
            "description": f"步骤 {step_number}: 工具执行完成",
            # 保留工具调用信息以便前端同时显示参数和结果
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "id": getattr(call, 'id', f"call_{step_number}_{i}")
                }
                for i, call in enumerate(tool_calls)
            ],
            "tool_results": [
                {
                    "name": result.name,
                    "result": str(result.result)[:1000] if result.result else "",  # 增加长度限制
                    "success": result.success,
                    "error": result.error
                }
                for result in tool_results
            ],
            "status": "completed"
        })
        
        for tool_result in tool_results:
            # Add tool result to conversation
            message = LLMMessage(role="user", tool_result=tool_result)
            messages.append(message)

        reflection = self.reflect_on_result(tool_results)
        if reflection:
            step.state = AgentState.REFLECTING
            step.reflection = reflection

            # Display reflection
            self._update_cli_console(step)

            messages.append(LLMMessage(role="assistant", content=reflection))

        return messages
    
    def _parse_task_content(self, message: str, metadata: Dict) -> str:
        """Parse task content from message, extracting any embedded parameters."""
        # Look for common task patterns and extract parameters
        lines = message.strip().split('\n')
        task_lines = []
        
        for line in lines:
            # Skip lines that look like parameter specifications
            if any(line.strip().startswith(prefix) for prefix in [
                'project_path:', 'file_path:', 'working_dir:', 'must_patch:', 'max_steps:'
            ]):
                # Extract parameter value and store in metadata
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'project_path':
                        metadata['project_path'] = value
                    elif key == 'file_path':
                        metadata['file_path'] = value
                    elif key == 'working_dir':
                        metadata['working_dir'] = value
                    elif key == 'must_patch':
                        metadata['must_patch'] = value.lower() in ['true', '1', 'yes']
                    elif key == 'max_steps':
                        try:
                            metadata['max_steps'] = int(value)
                        except ValueError:
                            pass
            else:
                task_lines.append(line)
        
        return '\n'.join(task_lines).strip() or message
    
    def _convert_execution_to_web_response(self, execution: AgentExecution, trajectory_path: str, task_args: Dict = None) -> Dict:
        """Convert TraeAgent execution result to web backend response format."""
        steps = []
        
        for i, step in enumerate(execution.steps, 1):
            # Map agent states to frontend-compatible types
            state_mapping = {
                "thinking": "thinking",
                "calling_tool": "tool_calling", 
                "reflecting": "thinking",
                "completed": "complete",
                "error": "error",
                "idle": "thinking"
            }
            
            step_type = state_mapping.get(step.state.value, step.state.value)
            
            step_data = {
                "step": i,
                "type": step_type,
                "content": "",
                "description": f"步骤 {i}: {step.state.value}"
            }
            
            # Add LLM response content
            if step.llm_response:
                step_data["content"] = step.llm_response.content
                
            # Add tool calls information
            if step.tool_calls:
                step_data["tool_calls"] = [
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "id": getattr(call, 'id', f"call_{i}")
                    }
                    for call in step.tool_calls
                ]
                
            # Add tool results
            if step.tool_results:
                step_data["tool_results"] = [
                    {
                        "name": result.name,
                        "result": result.result,
                        "success": result.success,
                        "error": result.error
                    }
                    for result in step.tool_results
                ]
            
            steps.append(step_data)
        
        # Determine response type
        response_type = "error_response" if not execution.success else "complete_response"
        if execution.success and any(step.tool_calls for step in execution.steps):
            response_type = "tool_response"
        
        # Check for patch generation
        patch_content = None
        patch_available = False
        if hasattr(self, 'get_git_diff'):
            try:
                patch_content = self.get_git_diff()
                patch_available = bool(patch_content.strip())
            except Exception:
                pass
        
        response = {
            "type": response_type,
            "steps": steps,
            "final_message": execution.final_result or "任务执行完成",
            "has_tool_calls": any(step.tool_calls for step in execution.steps),
            "waiting_for_approval": False,  # TraeAgent 不需要工具确认
            "execution_time": getattr(execution, 'execution_time', 0),
            "success": execution.success,
            "input_tokens": execution.total_tokens.input_tokens if execution.total_tokens.input_tokens else 0,
            "output_tokens": execution.total_tokens.output_tokens if execution.total_tokens.output_tokens else 0,
            "trajectory_file": trajectory_path,
            "patch_available": patch_available
        }
        
        # Add patch-related information if available
        if patch_available and patch_content:
            response["patch_content"] = patch_content
            if task_args and task_args.get("patch_path"):
                response["patch_file"] = task_args["patch_path"]
                
        # Add task parameters for reference
        if task_args:
            response["task_parameters"] = {
                "project_path": task_args.get("project_path"),
                "must_patch": task_args.get("must_patch") == "true",
                "patch_path": task_args.get("patch_path")
            }
        
        return response
    
    async def _initialize_mcp_servers(self):
        """Initialize MCP servers and load tools."""
        try:
            # Import ServerClient - 注意这里 servers 可能已经是 ServerClient 实例或配置字典
            from agents.simple_chatbot.mcp_simple_chatbot.iter_agent import ServerClient
            
            # Initialize MCP servers if servers config is available
            if hasattr(self, 'servers') and self.servers:
                # 检查 servers 是 ServerClient 实例列表还是配置字典
                if isinstance(self.servers, list):
                    # 如果是 ServerClient 实例列表（从 main.py 传入）
                    for server_client in self.servers:
                        try:
                            logger.info(f"正在初始化 MCP 服务器: {server_client.name} (类型: {getattr(server_client, 'server_type', 'unknown')})")
                            
                            # 服务器可能已经初始化过了，这里重新初始化确保状态正确
                            if not hasattr(server_client, 'session') or server_client.session is None:
                                await server_client.initialize()
                            
                            # Store server client
                            self.server_clients[server_client.name] = server_client
                            
                            # Get MCP tools from this server
                            mcp_tools = await MCPToolFactory.create_mcp_tools_async(
                                server_client=server_client,
                                model_provider=self._llm_client.provider.value
                            )
                            
                            # Add to our tool list
                            self.mcp_tools.extend(mcp_tools)
                            
                            logger.info(f"初始化 MCP 服务器成功: {server_client.name}, 加载了 {len(mcp_tools)} 个工具")
                            
                        except Exception as e:
                            import traceback
                            logger.error(f"初始化 MCP 服务器失败 {server_client.name}: {e}")
                            logger.error(f"错误详情: {traceback.format_exc()}")
                            # 继续处理其他服务器，不要因为一个服务器失败就停止
                            
                elif isinstance(self.servers, dict):
                    # 如果是配置字典
                    for server_name, server_config in self.servers.items():
                        try:
                            # Create and initialize server client
                            server_client = ServerClient(server_name, server_config)
                            await server_client.initialize()
                            
                            # Store server client
                            self.server_clients[server_name] = server_client
                            
                            # Get MCP tools from this server
                            mcp_tools = await MCPToolFactory.create_mcp_tools_async(
                                server_client=server_client,
                                model_provider=self._llm_client.provider.value
                            )
                            
                            # Add to our tool list
                            self.mcp_tools.extend(mcp_tools)
                            
                            logger.info(f"初始化 MCP 服务器成功: {server_name}, 加载了 {len(mcp_tools)} 个工具")
                            
                        except Exception as e:
                            logger.error(f"初始化 MCP 服务器失败 {server_name}: {e}")
                        
        except Exception as e:
            logger.error(f"MCP 服务器初始化过程出错: {e}")
    
    async def _cleanup_mcp_servers(self):
        """Cleanup MCP servers."""
        for server_name, server_client in self.server_clients.items():
            try:
                await server_client.cleanup()
                logger.info(f"清理 MCP 服务器: {server_name}")
            except Exception as e:
                logger.error(f"清理 MCP 服务器失败 {server_name}: {e}")
        
        self.server_clients.clear()
        self.mcp_tools.clear()
    
    def get_available_mcp_tools(self) -> List:
        """Get list of available MCP tools."""
        return self.mcp_tools
    
    async def new_task_with_mcp_tools(self, task: str, extra_args: dict[str, str] | None = None, tool_names: list[str] | None = None):
        """Create a new task with both TraeAgent tools and MCP tools."""
        from ..tools import tools_registry
        from ..tools.base import Tool
        from ..agent.trae_agent import TraeAgentToolNames
        
        self._task: str = task

        if tool_names is None:
            tool_names = TraeAgentToolNames

        # Get the model provider from the LLM client
        provider = self._llm_client.provider.value
        
        # Create standard TraeAgent tools
        from ..tools.base import ToolExecutor
        standard_tools: list[Tool] = [
            tools_registry[tool_name](model_provider=provider) for tool_name in tool_names
        ]
        
        # Add MCP tools
        all_tools = standard_tools + self.mcp_tools
        
        self._tools: list[Tool] = all_tools
        self._tool_caller = ToolExecutor(self._tools)

        # Setup initial messages (same as parent class)
        from ..utils.llm_basics import LLMMessage
        from .agent_basics import AgentError
        
        self._initial_messages: list[LLMMessage] = []
        self._initial_messages.append(LLMMessage(role="system", content=self.get_system_prompt()))

        user_message = ""
        if not extra_args:
            raise AgentError("Project path and issue information are required.")
        if "project_path" not in extra_args:
            raise AgentError("Project path is required")

        self.project_path = extra_args.get("project_path", "")
        user_message += f"[Project root path]:\n{self.project_path}\n\n"

        if "issue" in extra_args:
            user_message += f"[Problem statement]: We're currently solving the following issue within our repository. Here's the issue text:\n{extra_args['issue']}\n"
        
        # Set optional attributes
        optional_attrs_to_set = ["base_commit", "must_patch", "patch_path"]
        for attr in optional_attrs_to_set:
            if attr in extra_args:
                setattr(self, attr, extra_args[attr])

        self._initial_messages.append(LLMMessage(role="user", content=user_message))

        # If trajectory recorder is set, start recording
        if self._trajectory_recorder:
            self._trajectory_recorder.start_recording(
                task=task,
                provider=self._llm_client.provider.value,
                model=self._model_parameters.model,
                max_steps=self._max_steps,
            )
            
        logger.info(f"任务创建完成，总共加载 {len(all_tools)} 个工具 (标准: {len(standard_tools)}, MCP: {len(self.mcp_tools)})")
    
    async def approve_tools(self, approved_call_ids: List[str], user_id: str) -> Dict:
        """
        TraeAgent 不需要工具确认，但为了兼容性保留此方法。
        """
        return {
            "type": "info_response",
            "final_message": "TraeAgent 的工具调用不需要用户确认，会自动执行。",
            "has_tool_calls": False,
            "waiting_for_approval": False
        }
    
    async def reject_tools(self, rejected_call_ids: List[str], user_id: str) -> Dict:
        """
        TraeAgent 不需要工具确认，但为了兼容性保留此方法。
        """
        return {
            "type": "info_response", 
            "final_message": "TraeAgent 的工具调用不需要用户确认，无法拒绝已执行的工具。",
            "has_tool_calls": False,
            "waiting_for_approval": False
        }
    
    def get_patch_content(self) -> Optional[str]:
        """获取生成的补丁内容."""
        try:
            if hasattr(self, 'get_git_diff'):
                return self.get_git_diff()
            return None
        except Exception:
            return None