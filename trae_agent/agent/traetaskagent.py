"""TraeTaskAgent - Trae Agent adapted for web backend integration."""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, override

from .trae_agent import TraeAgent
from ..utils.config import Config, load_config
from ..utils.llm_basics import LLMMessage
from .agent_basics import AgentExecution

from logger import logger

class TraeTaskAgent(TraeAgent):
    """TraeTaskAgent - TraeAgent adapted for web backend integration."""

    def __init__(self, servers, llm_client, config, db_manager):
        """
        Initialize TraeTaskAgent for web backend integration.
        
        Args:
            servers: MCP servers list (compatibility with existing agents)
            llm_client: LLM client instance
            config: Configuration object (compatible with existing agent config)
            db_manager: Database manager instance
        """
        # Create Trae Config from existing config
        self.web_config = config
        self.db_manager = db_manager
        self.servers = servers
        self.web_llm_client = llm_client
        self.current_session_id = None
        
        # Initialize Trae Agent with converted config
        trae_config = self._create_trae_config()
        super().__init__(config=trae_config)
        
    def _create_trae_config(self) -> Config:
        """Create Trae Config from web config."""
        # Default Trae config structure
        trae_config_dict = {
            "default_provider": "openrouter",
            "max_steps": 32,
            "enable_lakeview": False,
            "model_providers": {
                "openrouter": {
                "api_key": "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",
                "base_url": "https://openrouter.ai/api/v1",
                "model": "anthropic/claude-sonnet-4",
                "max_tokens": 32768,
                "temperature": 0.5,
                "top_p": 1,
                "top_k": 0,
                "max_retries": 10,
                "parallel_tool_calls": False
                },
                "anthropic": {
                    "model": "claude-sonnet-4-20250514",
                    "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                    "base_url": "https://api.anthropic.com",
                    "max_tokens": 4096,
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 0,
                    "max_retries": 10,
                    "parallel_tool_calls": False
                },
                "openai": {
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.5,
                    "top_p": 1,
                    "max_retries": 10,
                    "parallel_tool_calls": False
                }
            }
        }
        
        return Config(trae_config_dict)
    
    async def initialize(self):
        """Initialize the agent."""
        # TraeAgent doesn't need explicit initialization like MCP agents
        pass
    
    async def cleanup_servers(self):
        """Cleanup resources."""
        # TraeAgent doesn't have servers to cleanup
        pass
    
    async def process_message(self, message: str, session_id: str, **kwargs) -> Dict:
        """
        Process a message and return structured response compatible with web backend.
        
        Args:
            message: User message
            session_id: Session ID
            **kwargs: Additional parameters (project_path, file_path, trajectory_file, etc.)
            
        Returns:
            Dict: Structured response compatible with existing web backend
        """
        try:
            self.current_session_id = session_id
            
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
                if not os.path.isabs(trajectory_file):
                    # Create in temp directory if relative path
                    trajectory_file = os.path.join(tempfile.gettempdir(), "trae_trajectories", trajectory_file)
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
                
                # Create new task
                self.new_task(task, task_args)
                
                # Execute task
                execution: AgentExecution = await self.execute_task()
                
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