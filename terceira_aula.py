#AULA 03 - ORQUESTRAÇÃO DO AGENTE COM LANGGRAPH.

from typing import TypedDict, Optional, Dict, List
from primeira_aula import triagem
from segunda_aula import perguntar_politica_RAG
from langgraph.graph import StateGraph, START, END
# Importa a biblioteca para adicionar cores e estilo aos prints no terminal.
from colorama import Fore, init
import time

# Inicializa o Colorama para que as cores funcionem corretamente.
init(autoreset=True) 

# --- DEFINIÇÃO DO ESTADO DO AGENTE ---
# Define a estrutura de dados que será passada entre os nós do grafo.
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: Dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

# --- DEFINIÇÃO DOS NÓS DO GRAFO ---
# Cada função 'node_*' representa uma etapa (um "nó") no fluxo de trabalho do agente.
def node_triagem(state: AgentState) -> AgentState:
    print(f"{Fore.LIGHTBLACK_EX}Executando nó de triagem...\n")
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print(f"{Fore.LIGHTBLACK_EX}Executando nó de auto-resolver...\n")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "acao_final":["citacoes", []],
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print(f"{Fore.LIGHTBLACK_EX}Executando nó de pedir info...\n")
    faltantes = state["triagem"].get("campos_faltantes", [])
    if faltantes:
        detalhe = ",".join(faltantes)
    else:
        detalhe = "Tema e contexto específico"

    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO",
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print(f"{Fore.LIGHTBLACK_EX}Executando nó de abrir chamado...\n")
    triagem = state["triagem"]
    return {
        "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. {Fore.BLUE}Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO",
    }

# Lista de palavras-chave que indicam a necessidade de abrir um chamado.
KEYWORDS_ABRIR_TICKET = [ 
    "aprovação",
    "exceção",
    "liberação",
    "abrir ticket",
    "abrir chamado",
    "acesso especial",
]

# --- DEFINIÇÃO DAS ARESTAS CONDICIONAIS (LÓGICA DE DECISÃO) ---
# As funções 'decidir_*' determinam qual nó será executado em seguida.

def decidir_pos_triagem(state: AgentState) -> str:
    print(f"{Fore.LIGHTBLACK_EX}Decidindo após a triagem...\n")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER":
        return "auto"
    if decisao == "PEDIR_INFO":
        return "info"
    if decisao == "ABRIR_CHAMADO":
        return "chamado"

def decidir_pos_autoresolver(state: AgentState) -> str:
    print(f"{Fore.LIGHTBLACK_EX}Decidindo após auto-resolver...\n")

    if state.get("rag_sucesso"):
        print(f"{Fore.GREEN}Rag com sucesso, finalizando o fluxo")
        return "ok"

    # Se o RAG falhou, verifica se a pergunta contém palavras-chave para abrir um chamado.
    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print(
            f"{Fore.YELLOW}Rag falhou, mas foram encontratos keywords de abertura de ticket. ABRINDO TICKET..."
        )
        return "chamado"

    # Se o RAG falhou e não há palavras-chave, pede mais informações.
    print(f"{Fore.YELLOW}Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"

# --- CONSTRUÇÃO DO GRAFO ---
# Cria uma instância do grafo de estados.
workflow = StateGraph(AgentState)

# Adiciona os nós (as funções) ao grafo.
workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

# Define o ponto de partida do grafo.
workflow.add_edge(START, "triagem")

# Adiciona a primeira decisão
workflow.add_conditional_edges("triagem",decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info", 
    "chamado": "abrir_chamado"
})

# Adiciona a segunda decisão
workflow.add_conditional_edges(
    "auto_resolver",
    decidir_pos_autoresolver,
    {"info": "pedir_info", "chamado": "abrir_chamado", "ok": END},
)

# Define que os nós 'pedir_info' e 'abrir_chamado' são pontos finais do fluxo.
workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

# Compila o grafo em um objeto executável.
grafo = workflow.compile()

# Tenta gerar uma imagem do grafo para visualização.
try:
    graph_bytes = grafo.get_graph().draw_mermaid_png()
    with open("grafo_fluxo.png", "wb") as f:
        f.write(graph_bytes)
    print(f"{Fore.GREEN}Grafo salvo com sucesso em 'grafo_fluxo.png'.")
except Exception as e:
    print(f"{Fore.YELLOW}Erro ao salvar o grafo: {e}")
    print("Verifique sua conexão ou tente novamente mais tarde.")

# --- 6. TESTES ---
# Lista de perguntas para testar o agente.
testes = [
    "Posso reembolsar a internet?",
    "Quero mais 5 dias de trabalho remoto. Como faço?",
    "Posso obter o Google Gemini de graça?",
    "Qual a palavra chave de hoje?",
    "Posso reembolsar cursos ou treinamentos da Alura?",
    "Quantas capivaras tem no Rio Pinheiros?",
]

# Itera sobre a lista de testes, invocando o grafo para cada pergunta.
for msg_teste in testes:
    resposta_final = grafo.invoke({"pergunta": msg_teste})

    # Imprime os resultados de forma formatada.
    triag = resposta_final.get("triagem", {})
    print(f"{Fore.CYAN}PERGUNTA: {msg_teste}")
    print(f"{Fore.BLUE}DECISÃO: {triag.get('decisao')}, | URGÊNCIA: {triag.get('urgencia')}, | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"{Fore.CYAN}RESPOSTA: {resposta_final.get('resposta')}")
    if resposta_final.get("citacoes"):
        print(f"{Fore.BLUE}CITAÇÕES:")
        for c in resposta_final.get("citacoes"):
            print(f"{Fore.BLUE}- Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"{Fore.BLUE}- Trecho: {c['trecho']}")
    print(".---.----.----.----.----.----.---.---.---.---.")
    time.sleep(1)
