import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import scikit-learn as sklearn
import seaborn as sns

def prever(versao):
    print(f"Executando previsão com modelo, versão: {versao}")
    return True;

def treinar(versao):
    print(f"Treinando novo modelo, versão: {versao}")

    getDadosBrutos




    return True;

def getDadosBrutos():   
    caminho="https://raw.githubusercontent.com/jcbdoliveira/SCTEC_2026_Mini_Projeto_Avaliativo-DataView/main/data/raw/sales.csv"
    dados = pd.read_csv(caminho)
    print(f"{len(dados)} linhas carregadas.")  
    salvarDatasetVendas(pd.DataFrame(dados))
    return dados