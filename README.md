# ⚖️ Law-GLM-RAG: 基于 DeepSeek-7B 的垂直领域法律 AI 助手

## 📖 项目介绍
本项目是一个运行在本地的法律垂直领域 AI 助手。为了解决通用大模型在法律咨询场景下的**知识幻觉**和**时效性滞后**问题，项目采用了 **SFT（指令微调）** + **RAG（检索增强生成）** 的技术架构。

核心功能包括：
1.  **专业微调**：基于 DeepSeek-7B 进行 LoRA 微调，增强法律文书写作与逻辑推理能力。
2.  **精准检索**：基于 FAISS 向量库与 BGE-M3 模型，实现了对《民法典》、《劳动合同法》的精准检索。
3.  **源头注入**：独创的 Metadata Injection 机制，解决了多部法律文档检索时的来源混淆问题。

## 🚀 Quick Start

### 1. 环境准备
```bash
# 克隆仓库
git clone [https://github.com/TheMatrix-ion/Law-RAG-Assistant.git](https://github.com/TheMatrix-ion/Law-RAG-Assistant.git)

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型准备
本项目依赖本地 Ollama 运行大模型, 请安装 Ollama, 再拉取模型

### 3. 构建知识库
```bash
python build_vector_db.py
```

### 4. 启动应用
```bash
streamlit run app.py
```
**注意**：本项目使用的 `my-lawyer` 是我私有微调的模型（GGUF格式，约4GB）。为方便演示，您可以在 `app.py` 中将 `OLLAMA_MODEL_NAME` 修改为通用的 `deepseek-llm:7b` 或 `qwen:7b`，RAG 功能依然可用。

## 贡献与联系
如有问题，欢迎提交 Issue 或联系 alexroe888@gmail.com
