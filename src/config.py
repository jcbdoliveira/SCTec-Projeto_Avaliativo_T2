import os

def criar_pastas():
    # Lista de pastas a serem criadas
    pastas = [
        "src",
        "data",
        "data/raw",
        "data/processed",
        "data/final",
        "models",
        "outputs",
        "outputs/figures"        
    ]

    # Criação das pastas
    for pasta in pastas:
        if not os.path.exists(pasta):
            os.makedirs(pasta)
            print(f"Pasta '{pasta}' criada com sucesso.")
        else:
            print(f"Pasta '{pasta}' já existe.")

def criar_novo_treinamento():
    # Definindo o caminho da pasta "models"
    pasta_models = "models"

    # Listando as versões existentes na pasta "models"
    versoes = [d for d in os.listdir(pasta_models) if os.path.isdir(os.path.join(pasta_models, d))]

    # Determinando o número da nova versão
    versao = 1
    if versoes:
        versao = max([int(v.replace("v", "")) for v in versoes]) + 1            

    # Criando a nova pasta de versão
    nova_versao = os.path.join(pasta_models, f"v{versao}")
    os.makedirs(nova_versao)
    print(f"Pasta '{nova_versao}' criada com sucesso.")
    return nova_versao

def existe_modelo_treinado():
    # Definindo o caminho da pasta "models"
    pasta_models = "models"

    # Listando as versões existentes na pasta "models"
    versoes = [d for d in os.listdir(pasta_models) if os.path.isdir(os.path.join(pasta_models, d))]

    # Verificando se existe algum modelo treinado
    if versoes:
        return True
    else:
        return False
    
def definir_nomes_arquivos():
    # Definindo os nomes dos arquivos
    nomes_arquivos = {
        "dados": "dados.csv",
        "modelo_regressao": " modelo_regressao.pkl",
        "metricas": " metricas.json",
        "label": "label.json"
    }
    return nomes_arquivos