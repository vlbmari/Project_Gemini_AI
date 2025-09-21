#AULA 01 - CLASSIFICAÇÃO E INTEGRAÇÃO POR IA.

import os 
from dotenv import load_dotenv  
from langchain_google_genai import ChatGoogleGenerativeAI  
from pydantic import BaseModel, Field
from typing import Literal, List, Dict 
from langchain_core.messages import SystemMessage, HumanMessage
from colorama import Fore,init 
import time

#--- 1. INICIALIZAÇÃO ---
# Ativa suporte a cores no terminal.
init(autoreset=True) 
# Carrega variáveis de ambiente do arquivo .env.
load_dotenv()
# Obtém a chave da API do Google.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#--- MODELOS DE LINGUAGEM ---
# Inicializa modelo LLM para testes gerais.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.0,  
)
# Exemplo de teste simples.
# resp_test = llm.invoke("Quantos dias de ferias eu tenho na minha empresa?")
# print(resp_test.content)

#--- 2. PROMPT DE TRIAGEM ---
# Define as regras que a IA deve seguir para classificar mensagens do usuário.
# O modelo deve responder apenas com um JSON estruturado (decisão, urgência e campos faltantes).
TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou sem detalhes suficientes.\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou abertura explícita de chamado.\n'
    "Analise a mensagem e decida a ação mais apropriada."
)

#--- 3. MODELO DE SAÍDA ---
# Estrutura de dados que padroniza a resposta da IA.
# Garante que toda saída siga o formato JSON esperado.
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER","PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA","MEDIA","ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

#--- 4. PIPELINE DE TRIAGEM ---
# Inicializa modelo dedicado à triagem.
llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

# Força saída estruturada no formato TriagemOut.
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

# Função principal de triagem: classifica a mensagem do usuário.
def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT), 
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()

#--- TESTES ---
if __name__ == "__main__":
    testes = ["Posso reembolsar a internet?",
            "Quero mais 5 dias de trabalho remoto. Como faço?",
            "Posso reembolsar cursos ou treinamentos da Alura?",
            "Qual é a palavra chave da aula de hoje?",
            "Quantas capivaras tem no Rio Pinheiros?"]

    for msg_teste in testes:
        print(f"{Fore.CYAN}Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}")
        print(".-.-.-.-.-.-.-.-.-.-.")
        time.sleep(1)
