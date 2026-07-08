import src.config as configurar
import src.install as instalar
import src.pipeline as preditor

configurar.criar_pastas() 
lista_arquivos = configurar.definir_nomes_arquivos()
vTreino= "v1"

if not configurar.existe_modelo_tereinado():
    vTreino = configurar.criar_novo_treinamento() 

instalar.instalar_bibliotecas()

