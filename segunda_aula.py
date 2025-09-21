#AULA 02 - CONSTRUINDO A BASE DE CONHECIMENTO COM RAG.

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from primeira_aula import GOOGLE_API_KEY, llm_triagem
from typing import Dict, List
from colorama import Fore, init
from pathlib import Path
import time, re, pathlib

#-------------FORMATADORES------------- 
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

init(autoreset=True) 

# --- 1. CARREGAMENTO DOS DOCUMENTOS ---
# Lê os arquivos PDF de uma pasta e adiciona ao conjunto de documentos para busca.
docs = []

for n in Path(
    r"C:\Users\slbma\Documents\ImersaoAluraAgentesIAPython\Project_Gemini_AI\RagPolitica"
).glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"{Fore.LIGHTBLACK_EX}Arquivo {n.name} carregado com sucesso")
    except Exception as e:
        print(f"{Fore.RED}Erro ao carregar o arquivo {n.name}: {e}")

print(f"{Fore.LIGHTBLACK_EX}Total de documentos carregados: {len(docs)}\n\n+")

# --- 2. DIVISÃO DOS DOCUMENTOS (CHUNKING) ---
# Quebra os documentos em partes menores para facilitar a indexação e recuperação.
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# --- 3. CRIAÇÃO DE EMBEDDINGS E VETORSTORE ---
# Gera representações numéricas dos chunks de texto usando modelo de embeddings.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    api_key=GOOGLE_API_KEY
)

# Armazena os embeddings em um banco vetorial FAISS para busca por similaridade.
vectorstores = FAISS.from_documents(chunks, embeddings)

# --- 4. CONFIGURAÇÃO DO RETRIEVER ---
# Define como buscar os documentos mais relevantes com base na pergunta do usuário.
retriever = vectorstores.as_retriever(search_type="similarity_score_threshold", 
                                      search_kwargs={"score_threshold": 0.3, "k": 4}
)

# --- 5. PROMPT E CHAIN DO RAG ---
# Define o comportamento do assistente.
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
        "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
        "Responda SOMENTE com base no contexto fornecido. "
        "Se não houver base suficiente, responda apenas 'Não sei'.",),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}"),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}"), # Reforço para o modelo considerar o contexto.
])

# Cria a cadeia de execução que conecta LLM + prompt + documentos.
document_chain = create_stuff_documents_chain(llm_triagem, prompt=prompt_rag)

# --- 6. FUNÇÃO PRINCIPAL DO RAG ---
# Orquestra a busca de documentos e a geração de resposta pelo modelo de linguagem.
def perguntar_politica_RAG(pergunta: str) -> Dict:
    # Recupera documentos relevantes
    docs_relacionados = retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", 
                "citacoes": [], 
                "contexto_encontrado": False}

    # Gera resposta com base na pergunta + contexto encontrado
    answer = document_chain.invoke({"input": pergunta, 
                                    "context": docs_relacionados})
    txt = (answer or "").strip()

    # Se o modelo não encontrou resposta confiável
    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", 
                "citacoes": [], 
                "contexto_encontrado": False}

    # Retorna resposta com citações dos documentos usados
    return {"answer": txt, 
            "citacoes": formatar_citacoes(docs_relacionados, pergunta), 
            "contexto_encontrado": True}

# --- TESTES ---
if __name__ == "__main__":
    testes = ["Posso reembolsar a internet?",
            "Quero mais 5 dias de trabalho remoto. Como faço?",
            "Posso reembolsar cursos ou treinamentos da Alura?",
            "Quantas capivaras tem no Rio Pinheiros?"]
    for msg_teste in testes:
        resposta = perguntar_politica_RAG(msg_teste)
        print(f"{Fore.CYAN}PERGUNTA: {msg_teste}")
        print(f"{Fore.CYAN}RESPOSTA: {resposta['answer']}") 
        
        if resposta['contexto_encontrado']:
            print(f"{Fore.BLUE}CITAÇÕES:")
            for c in resposta['citacoes']:
                print(f"{Fore.BLUE} - Documento: {c['documento']}, Página: {c['pagina']}")
                print(f"{Fore.BLUE}   Trecho: {c['trecho']}")
        print(".--.--.--.--.--.--.--.--.--.--.")
        time.sleep(1)
