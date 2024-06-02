# DermAssist

## Architecture
![DermAssist Architecture](./media/dermassist_arch.png)

## Vector Store Pipeline
![Vector_Store_Pipeline](./media/dermassist_data.png)

### Steps to run DermAssist locally

Clone the repository:
```commandline
https://github.com/chinmaysharmacs10/DermAssist.git
cd DermAssist
```

Create virtual environment:
```commandline
Conda:
conda create --name dermassist

Pip:
python -m venv dermassist
```

Install necessary packages:
```commandline
pip install requirements.txt
```

Download ollama client from: [https://ollama.com](https://ollama.com/)

Pull Llama3-8b model: 
```commandline
ollama run llama3
```

In rag_system.py, enter your `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT` to enable tracing via [LangSmith](https://www.langchain.com/langsmith).

To start StreamLit server:
```commandline
streamlit run dermassist_streamlit.py
```