"""ArbAgent - 投研分析Agent用于材料分析、信息获取、检索和推理."""

import asyncio
import json
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from logger import logger
from .mongodb_rag import MongoDBRAG, SimpleMongoDBRetriever

class ProcessingStep:
    """处理步骤记录"""
    def __init__(self, step_type: str, content: str, description: str = ""):
        self.step = 0
        self.type = step_type
        self.content = content
        self.description = description
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "type": self.type,
            "content": self.content,
            "description": self.description,
            "timestamp": self.timestamp
        }


class ArbAgent:
    """
    ArbAgent - 投研分析Agent
    
    核心功能：
    1. 材料分析和信息提取
    2. 数据检索和知识查询
    3. 推理和洞察生成
    4. 支持工具调用和迭代处理
    """
    
    def __init__(self, servers: list, llm_client, config, db_manager, user_id: str = None):
        """初始化ArbAgent"""
        self.servers = servers
        self.llm_client = llm_client
        self.config = config
        self.db_manager = db_manager
        self.user_id = user_id
        
        # 会话状态
        self.current_session_id: str = None
        self.session_user_id: str = None
        self.processing_steps: List[ProcessingStep] = []
        self.step_counter: int = 0
        self.current_iteration: int = 0
        
        # 消息历史
        self.messages: list[dict[str, str]] = []
        self.initialized: bool = False
        
        # 处理状态
        self.is_processing: bool = False
        
        # Initialize RAG system (use simple retriever by default)
        self.rag_system = None
        self.use_rag = True  # Enable RAG by default
        
        logger.info(f"ArbAgent initialized for user {user_id}")
    
    async def initialize(self):
        """初始化Agent资源"""
        if self.initialized:
            return
        
        try:
            logger.info("🚀 开始初始化 ArbAgent...")
            
            # 初始化MCP服务器连接
            for server in self.servers:
                try:
                    await server.start()
                    tools = await server.list_tools()
                    logger.info(f"✅ ArbAgent连接到服务器 {server.name}, 可用工具: {len(tools)}")
                except Exception as e:
                    logger.warning(f"⚠️ ArbAgent连接服务器 {server.name} 失败: {e}")
            
            # Initialize RAG system with dedicated MongoDB connection
            if self.use_rag:
                try:
                    logger.info("🔍 初始化 RAG 系统...")
                    
                    # Import MongoDBHandler for RAG (different from chat history DB)
                    from db.mongodb_handler import MongoDBHandler
                    
                    # Create dedicated MongoDB connection for RAG (zsxq database, tmt collection)
                    rag_db = MongoDBHandler(
                        host="localhost",
                        port=27018,  # RAG database port
                        database_name="zsxq",  # RAG database
                        username=os.getenv('MONGO_ROOT_USERNAME', 'admin'),
                        password=os.getenv('MONGO_ROOT_PASSWORD', 'admin123'),
                        auto_create_indexes=False  # Don't create indexes on init
                    )
                    
                    # Connect to the database
                    if not rag_db.connect():
                        logger.warning("⚠️ RAG 数据库连接失败")
                        self.rag_system = None
                    else:
                        # Use SimpleMongoDBRetriever (doesn't require vector embeddings)
                        self.rag_system = SimpleMongoDBRetriever(
                            db_manager=rag_db,
                            llm_client=self.llm_client,
                            collection_name="tmt"
                        )
                        logger.info("✅ RAG 系统初始化完成")
                except Exception as e:
                    logger.warning(f"⚠️ RAG 系统初始化失败: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    self.rag_system = None
            
            self.initialized = True
            logger.info("✅ ArbAgent 初始化完成")
            
        except Exception as e:
            logger.error(f"❌ ArbAgent 初始化失败: {e}")
            raise
    
    async def cleanup_servers(self):
        """清理服务器连接"""
        logger.info("清理 ArbAgent 服务器连接...")
        for server in self.servers:
            try:
                await server.stop()
                logger.info(f"✅ 关闭服务器 {server.name}")
            except Exception as e:
                logger.warning(f"⚠️ 关闭服务器 {server.name} 失败: {e}")
    
    def _add_step(self, step_type: str, content: str, description: str = ""):
        """添加处理步骤"""
        self.step_counter += 1
        step = ProcessingStep(step_type, content, description)
        step.step = self.step_counter
        self.processing_steps.append(step)
        logger.debug(f"添加步骤 {self.step_counter}: {step_type} - {description}")
    
    def _create_success_response(self, final_message: str) -> dict:
        """创建成功响应"""
        return {
            "type": "success_response",
            "final_message": final_message,
            "has_tool_calls": False,
            "waiting_for_approval": False,
            "steps": [step.to_dict() for step in self.processing_steps],
            "recursion_depth": self.current_iteration,
            "pending_tools": [],
            "auto_approved_tools": []
        }
    
    def _create_error_response(self, error_message: str, error_details: str = "") -> dict:
        """创建错误响应"""
        return {
            "type": "error_response",
            "final_message": error_message,
            "has_tool_calls": False,
            "waiting_for_approval": False,
            "steps": [step.to_dict() for step in self.processing_steps],
            "error": error_details,
            "recursion_depth": self.current_iteration
        }
    
    def _parse_query_intent(self, user_message: str) -> Dict[str, any]:
        """
        Parse user query to extract intent and parameters.
        
        Returns:
            Dict with intent type and extracted parameters
        """
        message_lower = user_message.lower()
        
        # Check for retrieval/summarization intent
        retrieval_keywords = ["检索", "查询", "搜索", "获取", "总结", "摘要", "分析"]
        is_retrieval = any(keyword in message_lower for keyword in retrieval_keywords)
        
        # Extract date (支持多种日期格式)
        date_patterns = [
            r"(\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?)",  # 2025-11-24 or 2025年11月24日
            r"今天|today",
            r"昨天|yesterday",
        ]
        
        date = None
        for pattern in date_patterns:
            match = re.search(pattern, user_message)
            if match:
                if "今天" in match.group() or "today" in match.group():
                    date = datetime.now().strftime("%Y-%m-%d")
                elif "昨天" in match.group() or "yesterday" in match.group():
                    date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                else:
                    date_str = match.group(1)
                    # Normalize date format
                    date = date_str.replace("年", "-").replace("月", "-").replace("日", "")
                break
        
        # Extract provider
        provider = None
        provider_patterns = [
            r"provider[为是:\s]+([^\s,，。]+)",
            r"来源[为是:\s]+([^\s,，。]+)",
            r"作者[为是:\s]+([^\s,，。]+)",
        ]
        
        for pattern in provider_patterns:
            match = re.search(pattern, user_message)
            if match:
                provider = match.group(1).strip()
                break
        
        # Check for stock recommendation keywords
        wants_stocks = any(word in message_lower for word in ["股票", "推荐", "标的", "个股"])
        
        return {
            "is_retrieval": is_retrieval,
            "date": date,
            "provider": provider,
            "wants_stocks": wants_stocks,
            "original_query": user_message
        }
    
    async def _handle_rag_query(self, intent: Dict[str, any]) -> Dict[str, any]:
        """
        Handle RAG-based retrieval and summarization query.
        
        Args:
            intent: Parsed query intent
            
        Returns:
            RAG query results
        """
        if not self.rag_system:
            return {
                "success": False,
                "message": "RAG 系统未初始化"
            }
        
        try:
            self._add_step("rag_retrieval", f"检索条件: date={intent.get('date')}, provider={intent.get('provider')}", "RAG检索")
            
            # Perform retrieval and summarization
            result = await self.rag_system.summarize(
                date=intent.get("date"),
                provider=intent.get("provider"),
                custom_query=None,  # Use default summarization prompt
                limit=100
            )
            
            if result["success"]:
                self._add_step("rag_summary", f"生成摘要，处理了 {result['num_documents']} 篇文档", "RAG摘要生成")
            else:
                self._add_step("rag_error", result.get("summary", "检索失败"), "RAG错误")
            
            return result
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {
                "success": False,
                "message": f"RAG 查询失败: {str(e)}",
                "error": str(e)
            }
    
    async def process_message(self, user_message: str, session_id: str = None, **kwargs) -> dict:
        """
        处理用户消息 - 核心消息处理函数
        
        TODO: 实现完整的消息处理逻辑，包括：
        - LLM调用
        - 工具执行
        - 迭代推理
        - 结果生成
        
        当前为测试版本，返回随机文本回复
        """
        try:
            # 确保已初始化
            if not self.initialized:
                await self.initialize()
            
            # 设置会话信息
            self.current_session_id = session_id
            self.session_user_id = self.user_id
            self.processing_steps = []
            self.step_counter = 0
            self.current_iteration = 0
            self.is_processing = True
            
            logger.info(f"🔍 ArbAgent 开始处理消息: {user_message[:100]}...")
            
            # 添加用户消息步骤
            self._add_step("user_input", user_message, "用户输入")
            
            # Parse user intent
            intent = self._parse_query_intent(user_message)
            logger.info(f"📋 解析查询意图: {intent}")
            
            # Check if this is a RAG query
            if intent["is_retrieval"] and self.rag_system and (intent["date"] or intent["provider"]):
                logger.info("🔎 检测到 RAG 检索请求，使用 RAG 系统处理...")
                
                self._add_step(
                    "intent_analysis",
                    f"检测到检索查询: date={intent['date']}, provider={intent['provider']}",
                    "意图分析"
                )
                
                # Handle RAG query
                rag_result = await self._handle_rag_query(intent)
                
                if rag_result["success"]:
                    # Format RAG response
                    final_message = f"""📊 **ArbAgent 投研分析报告**

**查询条件**:
- 📅 日期: {intent['date'] or '未指定'}
- 📝 来源: {intent['provider'] or '未指定'}
- 📄 检索文档数: {rag_result.get('num_documents', 0)}

---

{rag_result['summary']}

---
*ArbAgent - 基于 MongoDB RAG 的投研分析*
*处理时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
                    
                    self._add_step("final_response", final_message, "生成 RAG 分析报告")
                    logger.info(f"✅ RAG 查询完成")
                    
                    return self._create_success_response(final_message)
                else:
                    # RAG query failed
                    error_msg = rag_result.get("summary", "RAG 查询失败")
                    final_message = f"""⚠️ **检索失败**

{error_msg}

**查询条件**:
- 日期: {intent['date'] or '未指定'}
- 来源: {intent['provider'] or '未指定'}

请检查查询条件是否正确，或尝试其他查询。
"""
                    
                    self._add_step("final_response", final_message, "RAG 查询失败")
                    return self._create_success_response(final_message)
            
            # Fallback to original test implementation for non-RAG queries
            logger.info("💬 非 RAG 查询，使用默认处理流程...")
            
            # 临时测试实现：生成随机回复
            test_responses = [
                f"我已经收到您的投研分析请求：'{user_message}'。正在分析相关材料和数据...",
                f"根据您的问题：'{user_message}'，我需要检索相关的市场数据和研究报告。",
                f"关于'{user_message}'这个问题，我会从多个维度进行分析，包括基本面、技术面和市场情绪。",
                f"您询问的'{user_message}'涉及到深度的投研分析，让我为您提供专业的洞察...",
                f"我理解您对'{user_message}'的关注。作为投研分析Agent，我会综合多方面信息给您答案。",
            ]
            
            # 随机选择一个回复
            random_response = random.choice(test_responses)
            logger.info(f"🎲 生成测试回复: {random_response[:50]}...")
            
            # 添加AI思考步骤
            self._add_step("ai_thinking", random_response, "ArbAgent 分析思考")
            
            # 模拟一些处理步骤
            await asyncio.sleep(0.5)  # 模拟处理延迟
            
            self._add_step(
                "analysis", 
                "正在分析用户查询的意图和关键信息点...", 
                "意图分析"
            )
            
            await asyncio.sleep(0.3)
            
            self._add_step(
                "retrieval", 
                "模拟检索相关材料和数据源...", 
                "信息检索"
            )
            
            # 生成最终回复
            final_message = f"""📊 **ArbAgent 投研分析报告**

**您的查询**: {user_message}

**分析结果**:
{random_response}

---
**投研洞察**:
- 📈 市场趋势：基于当前数据，市场呈现稳定态势
- 💡 关键发现：需要关注相关指标的变化
- 🎯 建议：建议持续跟踪相关数据

**提示**: 如需检索具体的投研材料，请使用以下格式：
- "检索今天的 provider 为 XXX 的内容"
- "查询 2025-11-24 provider 为 180k 的内容，总结主要观点和股票推荐"

---
*ArbAgent v0.2 - 投研分析专家 (支持 RAG 检索)*
"""
            
            self._add_step("final_response", final_message, "生成最终分析报告")
            
            logger.info(f"✅ ArbAgent 处理完成，生成 {len(self.processing_steps)} 个步骤")
            
            return self._create_success_response(final_message)
            
        except Exception as e:
            logger.error(f"❌ ArbAgent 处理消息时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._create_error_response(
                f"抱歉，处理您的投研分析请求时出现了错误: {str(e)}",
                str(e)
            )
        finally:
            self.is_processing = False
    
    async def approve_tools(self, approved_call_ids: List[str], user_id: str = None) -> dict:
        """
        批准工具调用
        
        TODO: 实现工具审批逻辑
        """
        logger.info(f"ArbAgent 收到工具批准请求: {approved_call_ids}")
        return self._create_success_response("工具批准功能开发中...")
    
    async def reject_tools(self, rejected_call_ids: List[str], user_id: str = None) -> dict:
        """
        拒绝工具调用
        
        TODO: 实现工具拒绝逻辑
        """
        logger.info(f"ArbAgent 收到工具拒绝请求: {rejected_call_ids}")
        return self._create_success_response("工具拒绝功能开发中...")