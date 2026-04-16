import os
from dotenv import load_dotenv
# 强制国内镜像，解决网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# LangChain 核心工具
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# HuggingFace 官方向量化（核心区别）
from langchain_huggingface import HuggingFaceEmbeddings
# ===================== 【可配置区】全在这里改 =====================
# 文档路径
INPUT_FILE = "test1.txt"
# 文本分块大小
CHUNK_SIZE = 300
# 分块重叠长度
CHUNK_OVERLAP = 50
# 向量库路径
CACHE_DIR = "./model_cache"
# 检索返回条数
TOP_K = 3
# ====================== HuggingFace 版 RAG 封装 ======================
class RAGChatbot:
    def __init__(self, api_key: str, model_name: str = "deepseek-chat", temperature: float = 0.1):
        """
        RAG 聊天机器人初始化（HuggingFace 版）
        :param api_key: DeepSeek API Key
        """
        # 1. 初始化大模型
        self.llm = ChatDeepSeek(
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
        # 2. 初始化 HuggingFace 向量化模型（超轻量、快速、通用）
        self.embeddings = self._init_embedding_model()
        # 3. 向量库
        self.db = None

    def _init_embedding_model(self):
        """HuggingFace 向量化模型（一行加载，代码极简）"""
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # 全球最常用轻量向量模型
            model_kwargs={'device': 'cpu'}  # 笔记本 CPU 直接跑
        )
        return embeddings

    def load_document(self, file_path: str =INPUT_FILE, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """加载并切分文档（和之前完全通用）"""
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = splitter.split_documents(documents)
        print(f"✅ 文档加载完成，共切分为 {len(texts)} 段")
        return texts

    def build_vector_db(self, file_path: str = INPUT_FILE):
        """一键构建向量库（通用）"""
        texts = self.load_document(file_path)
        self.db = FAISS.from_documents(texts, self.embeddings)
        print("✅ 向量库构建完成！")

    def search(self, query: str, k: int = 3):
        """语义检索（通用）"""
        if not self.db:
            raise Exception("❌ 请先调用 build_vector_db() 构建向量库")
        return self.db.similarity_search(query, k=k)

    def ask(self, query: str) -> str:
        """RAG 问答（通用）"""
        # 1. 检索相关文档
        docs = self.search(query)
        context = "\n".join([d.page_content for d in docs])

        # 2. 构造提示词
        prompt = f"""
        请严格根据以下文档内容回答问题，不许编造信息。
        文档内容：{context}
        问题：{query}
        回答：
        """

        # 3. AI 生成答案
        response = self.llm.invoke(prompt)
        return response.content

# ====================== 测试代码 ======================
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # 初始化 HuggingFace 版 RAG
    bot = RAGChatbot(api_key=api_key)

    # 构建向量库
    bot.build_vector_db(INPUT_FILE)

    # 提问
    question = "文档里写了什么内容？"
    answer = bot.ask(question)

    # 输出
    print("\n" + "="*50)
    print(f"❓ 问题：{question}")
    print(f"🤖 AI 回答：{answer}")
    print("="*50)