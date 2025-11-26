"""MongoDB RAG Module using llama_index for document retrieval and analysis."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from logger import logger

# Optional llama_index imports - only needed for advanced RAG features
try:
    from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.anthropic import Anthropic
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    logger.warning("llama_index not fully installed. Advanced RAG features will be disabled. SimpleMongoDBRetriever will still work.")
    LLAMA_INDEX_AVAILABLE = False

# MongoDB vector store is optional
try:
    from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
    MONGODB_VECTOR_AVAILABLE = True
except ImportError:
    logger.info("llama_index MongoDB vector store not available")
    MONGODB_VECTOR_AVAILABLE = False


class MongoDBRAG:
    """
    MongoDB-based RAG system using llama_index.
    
    Provides document indexing, retrieval, and query capabilities
    for investment research materials stored in MongoDB.
    
    Note: Requires llama_index to be installed.
    """
    
    def __init__(
        self,
        db_manager,
        collection_name: str = "tmt",
        embedding_model: str = "local",
        llm_provider: str = "openai",
        llm_config: dict = None
    ):
        """
        Initialize MongoDB RAG system.
        
        Args:
            db_manager: MongoDB handler instance
            collection_name: MongoDB collection to query
            embedding_model: "local" for HuggingFace or "openai"
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_config: LLM configuration dict
        """
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                "llama_index is required for MongoDBRAG. "
                "Install with: pip install llama-index llama-index-embeddings-huggingface llama-index-llms-anthropic"
            )
        
        self.db_manager = db_manager
        self.collection_name = collection_name
        self.llm_config = llm_config or {}
        
        # Initialize embedding model
        if embedding_model == "openai":
            self.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            # Use local HuggingFace model (default)
            self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-zh-v1.5"  # Chinese embedding model
            )
        
        # Initialize LLM
        if llm_provider == "anthropic":
            self.llm = Anthropic(
                model=self.llm_config.get("model", "claude-sonnet-4-20250514"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 4096)
            )
        else:
            # Default to OpenAI
            self.llm = OpenAI(
                model=self.llm_config.get("model", "gpt-4o"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 4096)
            )
        
        # Configure llama_index settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Will be initialized on first use
        self.index = None
        self.query_engine = None
        
        logger.info(f"MongoDBRAG initialized with {embedding_model} embeddings and {llm_provider} LLM")
    
    def _build_documents_from_query(
        self,
        filter_dict: Dict[str, Any] = None,
        limit: int = None
    ) -> List[Document]:
        """
        Build llama_index Documents from MongoDB query results.
        
        Args:
            filter_dict: MongoDB query filter
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of llama_index Document objects
        """
        # Query MongoDB
        results = self.db_manager.find_many(
            collection_name=self.collection_name,
            filter_dict=filter_dict or {},
            limit=limit
        )
        
        logger.info(f"Retrieved {len(results)} documents from MongoDB")
        
        # Convert to llama_index Documents
        documents = []
        for doc in results:
            # Use the OCR-processed content for better text quality
            content = doc.get("dsocred_content") or doc.get("ocred_content") or doc.get("content", "")
            
            # Build metadata
            metadata = {
                "date": doc.get("date", ""),
                "time": doc.get("time", ""),
                "provider": doc.get("provider", ""),
                "url": doc.get("url", ""),
                "uri": doc.get("uri", ""),
                "filename": doc.get("filename", "")
            }
            
            # Create Document
            llama_doc = Document(
                text=content,
                metadata=metadata,
                id_=str(doc.get("_id", ""))
            )
            documents.append(llama_doc)
        
        return documents
    
    def build_index_from_query(
        self,
        filter_dict: Dict[str, Any] = None,
        limit: int = None
    ) -> VectorStoreIndex:
        """
        Build a vector index from MongoDB query results.
        
        Args:
            filter_dict: MongoDB query filter
            limit: Maximum number of documents to retrieve
            
        Returns:
            VectorStoreIndex for querying
        """
        logger.info(f"Building index from MongoDB query: {filter_dict}")
        
        # Get documents from MongoDB
        documents = self._build_documents_from_query(filter_dict, limit)
        
        if not documents:
            logger.warning("No documents found for indexing")
            return None
        
        # Build index
        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
        logger.info(f"Index built successfully with {len(documents)} documents")
        return self.index
    
    def query(
        self,
        query_text: str,
        filter_dict: Dict[str, Any] = None,
        limit: int = None,
        similarity_top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query_text: Query text
            filter_dict: MongoDB filter to pre-filter documents
            limit: Maximum documents to index
            similarity_top_k: Number of similar chunks to retrieve
            
        Returns:
            Dict with response and source information
        """
        try:
            # Build or rebuild index if needed
            if self.index is None or filter_dict is not None:
                self.build_index_from_query(filter_dict, limit)
            
            if self.index is None:
                return {
                    "success": False,
                    "response": "没有找到相关文档",
                    "sources": []
                }
            
            # Update query engine if needed
            if self.query_engine is None:
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=similarity_top_k,
                    response_mode="tree_summarize"
                )
            
            # Execute query
            logger.info(f"Executing RAG query: {query_text[:100]}...")
            response = self.query_engine.query(query_text)
            
            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    source_info = {
                        "content": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                        "score": node.score,
                        "metadata": node.node.metadata
                    }
                    sources.append(source_info)
            
            return {
                "success": True,
                "response": str(response),
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "response": f"查询失败: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def retrieve_and_summarize(
        self,
        date: str = None,
        provider: str = None,
        query: str = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Retrieve documents by date and provider, then generate summary.
        
        Args:
            date: Date filter (e.g., "2025-11-24")
            provider: Provider name filter
            query: Custom query for summarization
            limit: Maximum documents to retrieve
            
        Returns:
            Summary and analysis results
        """
        # Build filter
        filter_dict = {}
        if date:
            filter_dict["date"] = date
        if provider:
            filter_dict["provider"] = provider
        
        logger.info(f"Retrieving and summarizing documents: date={date}, provider={provider}")
        
        # Build index from filtered documents
        self.build_index_from_query(filter_dict, limit)
        
        if self.index is None:
            return {
                "success": False,
                "summary": f"未找到 date={date}, provider={provider} 的相关文档",
                "num_documents": 0
            }
        
        # Generate summary query
        if query is None:
            query = f"""请分析以下投资研究材料，提供详细的总结报告：

1. **主要观点和核心论点**：总结材料中的关键观点和主要论述
2. **股票推荐**：列出所有提及的股票代码、公司名称和推荐理由
3. **市场趋势分析**：归纳对市场趋势的判断和预测
4. **投资建议**：提炼具体的投资建议和策略
5. **风险提示**：总结提及的主要风险点

请确保信息准确、结构清晰、重点突出。"""
        
        # Execute query
        result = self.query(query, filter_dict=filter_dict, limit=limit)
        
        if result["success"]:
            return {
                "success": True,
                "summary": result["response"],
                "sources": result["sources"],
                "num_documents": len(self._build_documents_from_query(filter_dict, limit)),
                "filter": filter_dict
            }
        else:
            return {
                "success": False,
                "summary": result.get("response", "生成摘要失败"),
                "error": result.get("error", "Unknown error"),
                "filter": filter_dict
            }


class SimpleMongoDBRetriever:
    """
    Simplified MongoDB retriever without vector indexing.
    Uses direct MongoDB queries and LLM for summarization.
    """
    
    def __init__(self, db_manager, llm_client=None, collection_name: str = "tmt"):
        """
        Initialize simple retriever.
        
        Args:
            db_manager: MongoDB handler instance
            llm_client: LLM client for summarization
            collection_name: MongoDB collection name
        """
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.collection_name = collection_name
        
        logger.info("SimpleMongoDBRetriever initialized")
    
    def retrieve(
        self,
        date: str = None,
        provider: str = None,
        limit: int = 100
        ) -> List[Dict[str, Any]]:
        """
        Retrieve documents from MongoDB.
        
        Args:
            date: Date filter
            provider: Provider filter
            limit: Maximum documents
            
        Returns:
            List of documents
        """
        filter_dict = {}
        if date:
            filter_dict["date"] = date
        if provider:
            filter_dict["provider"] = provider
        
        documents = self.db_manager.find_many(
            collection_name=self.collection_name,
            filter_dict=filter_dict,
            limit=limit
        )
        
        logger.info(f"Retrieved {len(documents)} documents with filter: {filter_dict}")
        return documents
    
    def format_documents(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents for LLM consumption."""
        formatted = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("dsocred_content") or doc.get("ocred_content") or doc.get("content", "")
            formatted.append(f"""
### 文档 {i}
- **来源**: {doc.get('provider', 'Unknown')}
- **日期**: {doc.get('date', 'Unknown')} {doc.get('time', '')}
- **链接**: {doc.get('url', 'N/A')}

**内容**:
{content[:2000]}{'...' if len(content) > 2000 else ''}
---
""")
        return "\n".join(formatted)
    
    async def summarize(
        self,
        date: str = None,
        provider: str = None,
        custom_query: str = None,
        limit: int = 50
        ) -> Dict[str, Any]:
        """
        Retrieve and summarize documents.
        
        Args:
            date: Date filter
            provider: Provider filter
            custom_query: Custom summarization instructions
            limit: Maximum documents
            
        Returns:
            Summary results
        """
        # Retrieve documents
        documents = self.retrieve(date=date, provider=provider, limit=limit)
        
        if not documents:
            return {
                "success": False,
                "summary": f"未找到符合条件的文档 (date={date}, provider={provider})",
                "num_documents": 0
            }
        
        # Format documents
        formatted_docs = self.format_documents(documents)
        
        # Build prompt
        if custom_query is None:
            custom_query = f"""请分析以下投资研究材料（共{len(documents)}篇），提供详细的总结报告：

1. **主要观点和核心论点**：总结材料中的关键观点和主要论述
2. **股票推荐**：列出所有提及的股票代码、公司名称和推荐理由
3. **市场趋势分析**：归纳对市场趋势的判断和预测
4. **投资建议**：提炼具体的投资建议和策略
5. **风险提示**：总结提及的主要风险点

请确保信息准确、结构清晰、重点突出。"""
        
        prompt = f"""{custom_query}

{formatted_docs}
"""
        
        # Generate summary using LLM
        try:
            if self.llm_client:
                # Use configured LLM client - build messages for get_response
                messages = [
                    {"role": "user", "content": prompt}
                ]
                summary = self.llm_client.get_response(messages)
            else:
                # Fallback: return formatted documents
                summary = f"检索到 {len(documents)} 篇文档，但未配置 LLM 进行摘要。\n\n{formatted_docs[:1000]}"
            
            return {
                "success": True,
                "summary": summary,
                "num_documents": len(documents),
                "filter": {"date": date, "provider": provider}
            }
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                "success": False,
                "summary": f"生成摘要时出错: {str(e)}",
                "num_documents": len(documents),
                "error": str(e)
            }

