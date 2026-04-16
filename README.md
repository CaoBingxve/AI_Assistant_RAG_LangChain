# AI Assistant RAG LangChain

## 项目功能

这是一个基于 LangChain 和 HuggingFace 的 RAG（检索增强生成）聊天机器人项目，主要功能包括：

- **文档问答**：基于本地文档内容进行智能问答，禁止编造信息
- **向量存储**：使用 FAISS 向量库存储文档嵌入，实现高效语义检索
- **LangGraph Agent**：基于 LangChain 1.2.15 的官方推荐 create_agent() 方法，构建智能代理
- **多模型支持**：使用 DeepSeek 大语言模型进行文本生成，HuggingFace 向量化模型进行文档嵌入

## 技术栈

- **语言**：Python 3.11+
- **核心框架**：LangChain 1.2.15, LangGraph 0.2.0
- **向量库**：FAISS
- **大语言模型**：DeepSeek
- **向量化模型**：HuggingFace all-MiniLM-L6-v2
- **工具**：uv (依赖管理)

## 项目结构

```
AI_Assistant_RAG_LangChain/
├── rag-demo/
│   ├── rag_chatbot_hf.py  # RAG 核心模块
│   ├── agent_base.py      # LangGraph Agent 主程序
│   ├── test1.txt          # 测试文档
├── .env                   # 环境变量配置
├── pyproject.toml         # 项目配置
├── requirements.txt       # 依赖文件
└── README.md              # 项目说明
```

## 运行步骤

### 1. 环境配置

1. **安装依赖**：
   ```bash
   uv install
   ```

2. **配置环境变量**：
   在 `.env` 文件中添加 DeepSeek API Key：
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

### 2. 运行项目

1. **进入项目目录**：
   ```bash
   cd rag-demo
   ```

2. **运行 Agent**：
   ```bash
   python agent_base.py
   ```

3. **开始对话**：
   - 输入问题进行问答
   - 输入 `q` 退出

## 核心模块说明

### rag_chatbot_hf.py

- **功能**：封装 RAG 核心功能，包括文档加载、文本切分、向量库构建和问答生成
- **特点**：使用 HuggingFace 向量化模型，支持自定义文档路径和分块参数

### agent_base.py

- **功能**：基于 LangGraph 构建智能代理，集成 RAG 功能
- **特点**：使用 LangChain 1.2.15 官方推荐的 create_agent() 方法，支持工具调用

## 测试

1. **替换测试文档**：
   修改 `test1.txt` 文件内容，添加自定义文档

2. **运行测试**：
   ```bash
   python agent_base.py
   ```

3. **验证问答**：
   输入与文档相关的问题，验证回答是否基于文档内容

## 依赖管理

项目使用 uv 进行依赖管理，依赖文件包括：
- `pyproject.toml`：项目配置和依赖声明
- `requirements.txt`：固定版本依赖列表

## 注意事项

- 确保网络连接正常，以便下载 HuggingFace 模型
- DeepSeek API Key 需要在 `.env` 文件中正确配置
- 文档内容应清晰明确，以便获得准确的问答结果
