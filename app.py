import streamlit as st
import os
import sys

# 引用 langchain 组件
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 配置页面 ---
st.set_page_config(page_title="法律 AI 助手", page_icon="⚖️")
st.title("⚖️ 专业法律 AI 顾问")
st.caption("基于 DeepSeek-7B 微调 + RAG 检索增强 | 支持民法典与劳动合同法")

# --- 核心配置 ---
DB_PATH = "law_vector_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_MODEL_NAME = "my-lawyer:latest"

# 解决代理问题（如果需要）
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:33210'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:33210'

# --- 关键：使用缓存加载模型 ---
# 加上 @st.cache_resource 装饰器，
# 这样每次发消息时，不用重新加载 4GB 的数据库和模型，速度飞快！
@st.cache_resource
def load_chain():
    print("正在初始化模型资源...")
    
    # 1. 加载 Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}, # 显存不够用 cpu，够用写 cuda
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. 加载向量库
    if not os.path.exists(DB_PATH):
        st.error(f"找不到数据库 {DB_PATH}，请先运行向量库.py！")
        return None
        
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    # 增加分数过滤，避免乱答
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}
    )
    
    # 3. 加载 LLM
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)
    
    # 4. Prompt
    template = """
    你是一名专业的中国法律顾问。请必须严格依据下方的【参考法律条文】来回答用户的问题。
    
    严禁使用自带知识编造！如果参考条文中没有相关内容，或者分数过低，请直接回答“抱歉，当前法律知识库中未收录相关法条，无法回答。”

    【参考法律条文】：
    {context}

    【用户问题】：
    {question}

    【专业回答】：
    请一步步分析法律依据，最后给出结论。
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Chain
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# 加载链条
chain = load_chain()

# --- 聊天界面逻辑 ---

# 1. 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "您好！我是您的专属法律顾问。请问有什么关于劳动法或民法典的问题？"}]

# 2. 显示聊天历史
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 3. 处理用户输入
if prompt_text := st.chat_input("请输入法律问题..."):
    # 显示用户的话
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.chat_message("user").write(prompt_text)

    # 显示 AI 的回答（带加载动画）
    if chain:
        with st.chat_message("assistant"):
            with st.spinner("正在检索法条并思考..."):
                try:
                    response = chain.invoke(prompt_text)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"发生错误: {e}")