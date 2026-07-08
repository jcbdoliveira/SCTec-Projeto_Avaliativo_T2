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
vTreino= ""
Config.criar_pastas() 
lista_arquivos = Config.definir_nomes_arquivos()
Install.instalar_bibliotecas()
#==========================================================================

def main():
        
        if len(sys.argv) > 1 and sys.argv[1].lower() in ('--versao:', '--treinar', ''):
            
            if sys.argv[1].lower() == '--versao:':                
                vTreino = sys.argv[2].split(":")[1] 
                treinar = False
                print(f"Usando versão de treinamento: {vTreino}")

            if sys.argv[1].lower() == '--treinar':
                vTreino = Config.criar_novo_treinamento()
                print(f"Criando nova versão de treinamento: {vTreino}")
                        
            if len(sys.argv) > 1 and sys.argv[1].lower() == '':
                vTreino = Config.criar_novo_treinamento()
                print(f"Criando nova versão de treinamento: {vTreino}")
        else:
            print("Argumento fornecido inválido. Use '--versao:v<NUMERO>' ou '--treinar'.")
            exit(1)  

        if treinar:
            Pipeline.treinar(vTreino)   
        
        if not treinar:
            Pipeline.prever(vTreino)

if __name__ == "__main__":
    main()    
    



