import sys
import os

import src.config as Config
import src.install as Install
import src.pipeline as Pipeline

#==========================================================================
#Antes de executar o script cria toda a estrutura de pastas e 
#arquivos necessários para o projeto
#==========================================================================
treinar = True
vTreino= ''
Config.criar_pastas() 
lista_arquivos = Config.definir_nomes_arquivos()
Install.instalar_bibliotecas()
#==========================================================================

def main(argumento):
    
    treinar = False

    if argumento[:9] == '--versao:':                
        vTreino = argumento.split(":")[1]         
        print(f"Usando versão de treinamento: {vTreino}")

    if argumento[:9]  == '--treinar':
        treinar = True
        vTreino = Config.criar_novo_treinamento()
        print(f"Criando nova versão de treinamento: {vTreino}")
                
    if argumento == '':
        treinar = True
        vTreino = Config.criar_novo_treinamento()
        print(f"Criando nova versão de treinamento: {vTreino}")

    if treinar:
        Pipeline.treinar(vTreino)   
    
    if not treinar:
        Pipeline.prever(vTreino)

if __name__ == "__main__":  

    argumento = "--versao:v1"  # Valor padrão para o argumento
    argumento = "--treinar" 
    if len(sys.argv) > 1 and sys.argv[1].lower()[:9] in ('--versao:', '--treinar', ''):                 
        argumento = sys.argv[1]  # Atualiza o argumento se fornecido na linha de comando  
    
    main(argumento)    
    



