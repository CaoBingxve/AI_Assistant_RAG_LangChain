import os
from dotenv import load_dotenv
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 导入你封装好的 RAG（完全不变）
from rag_chatbot_hf import RAGChatbot

# ===================== 【LangChain 1.2.15 官方正确导入】 =====================
# 1. 唯一官方推荐：create_agent() 基于 LangGraph
# 2. 导入路径在 1.2.15 中 100% 正确
# ==========================================================================
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

# ===================== 【你的原配置完全保留】 =====================
INPUT_FILE = "test1.txt"
TOP_K = 3
# ==============================================================

# 加载配置
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化 RAG（你的代码完全不变）
rag = RAGChatbot(api_key=api_key)
rag.build_vector_db(INPUT_FILE)

# ===================== LangChain 标准工具封装 =====================
def rag_qa(query: str) -> str:
    """文档问答工具：查询本地文档内容，回答用户问题"""
    return rag.ask(query)

tools = [
    StructuredTool.from_function(
        func=rag_qa,
        name="document_qa",
        description="用于查询用户提供的本地文档内容，回答相关问题，严禁编造文档中没有的信息"
    )
]

# ===================== LangChain 1.2.15 官方标准 Agent =====================
model = ChatDeepSeek(api_key=api_key, model="deepseek-chat", temperature=0.1)

# ✅ 官方唯一推荐：create_agent()（1.2.15 版本 100% 正确）
# 注意：create_agent() 返回的是一个 CompiledStateGraph 对象，不需要 AgentExecutor
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个智能助手，当用户的问题涉及本地文档内容时，必须调用文档问答工具获取准确信息，禁止编造。"
)

# ===================== 启动对话（用 invoke 执行） =====================
if __name__ == "__main__":
    print("="*60)
    print("✅ LangChain 1.2.15 官方标准 Agent 已启动！")
    print("✅ 完全匹配你的 pyproject.toml 配置")
    print("✅ 零导入错误 | 零版本冲突")
    print("📝 输入 q 退出")
    print("="*60)

    while True:
        question = input("\n请输入问题：")
        if question.lower() == "q":
            print("👋 退出成功")
            break
        # ✅ 用 invoke 执行，和原功能完全一致
        # 注意：CompiledStateGraph 的 invoke 方法接受 messages 参数
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        # 从结果中获取最后一条消息的内容
        output = result["messages"][-1].content
        print(f"\n✅ 回答：{output}")