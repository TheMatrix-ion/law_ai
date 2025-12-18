import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:33210'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:33210'
# --- 配置区 ---
DATA_FOLDER = "law_data" 
DB_SAVE_PATH = "law_vector_db"
EMBEDDING_MODEL = "BAAI/bge-m3" 

def create_db():
    print(f"1. 正在扫描 '{DATA_FOLDER}' ...")
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"找不到文件夹: {DATA_FOLDER}")

    # 1. 加载阶段：只负责读文件和存元数据
    all_raw_docs = []
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    
    for file_name in files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            
            # 提取来源名
            source_name = os.path.splitext(file_name)[0]
            
            for doc in docs:
                # 关键：把来源存进 metadata，先别改 page_content
                doc.metadata["source_name"] = source_name
            
            all_raw_docs.extend(docs)
        except Exception as e:
            print(f"   [警告] 加载 {file_name} 失败: {e}")

    # 2. 切分阶段
    print("2. 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "第"]
    )
    split_docs = text_splitter.split_documents(all_raw_docs)
    
    # 3. 🔥🔥 注入阶段：切完后再给每个碎片盖章 🔥🔥
    print("3. 正在给每个切片注入来源标签...")
    for chunk in split_docs:
        # 从 metadata 里取回刚才存的名字
        source = chunk.metadata.get("source_name", "未知法律")
        # 把它硬编码进正文开头
        chunk.page_content = f"【法律来源：{source}】\n{chunk.page_content}"

    print(f"   处理完成！共生成 {len(split_docs)} 个带标签的切片。")

    # 4. 向量化阶段
    print(f"4. 正在构建向量库 ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}, # 显存不够改成 'cpu'
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(DB_SAVE_PATH):
        shutil.rmtree(DB_SAVE_PATH) # 删除旧库，防止混淆

    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(DB_SAVE_PATH)
    print(f">>> 恭喜！数据库已重建，每个切片都带上了身份证！")

if __name__ == "__main__":
    create_db()