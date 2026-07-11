import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def prever(versao):
    print(f"Executando previsão com modelo, versão: {versao}")
    return True;

def treinar(versao):
    print(f"Treinando novo modelo, versão: {versao}")

    print(f"Obter dados brutos para treinamento...")    
    dfBruto = pd.read_csv(getDadosBrutos(), sep=";", decimal=",")

    print(f"Tratando dados brutos para treinamento...")    
    dfTratado = dfBruto.copy()
    dfTratado["Valor_Limpo"] = dfBruto["Valor"].apply(tratar_valores)

    print(f"Removendo colunas inuteis...")  
    dfLimpo = remover_colunas_inuteis(dfTratado)

    print(f"Agrupando dados por mes, ano e conta...")  
    dfLimpo = agrupar_por_mes_ano_conta(dfLimpo)

    print(f"Tratando valores nulos...")  
    #dfLimpo = tratando_valores_nulos(dfLimpo)

    print(f"Salvando dados tratados para treinamento...")  
    destino = f"data/processed/Data_{versao}.csv" 
    dfLimpo.to_csv(destino, sep=";", index=False)
    destino = f"data/processed/Data_{versao}.json" 
    dfLimpo.to_json(destino, orient="records")

    print(f"Aplicando engenharia de features...")  
    dfLimpo = feature_engineering(dfLimpo)

    print(f"Treinando modelo com dados tratados e engenharia de features...")  
    X_train_scaled, X_test_scaled, y_train, y_test = treinamento_80_20(dfLimpo)

    print(f"Métricas do modelo treinado...")  
    metricas_modelo(X_train_scaled, X_test_scaled, y_train, y_test, dfLimpo, versao)


    return True;

def tratar_valores(val):
    if pd.isna(val) or str(val).strip() == "" or str(val).strip() == ",00":
        return 0.0 # Retornar 0.0 para valores nulos ou vazios

    val_str = str(val).strip()        
    val_str = val_str.replace(",", ".")

    try:
        #Converte para número e o abs() transforma o negativo contábil em positivo
        return abs(float(val_str))
    except (ValueError, TypeError):
        return 0.0

def remover_colunas_inuteis(df_original):
    # 1. Faz uma cópia de segurança para não afetar o dado original
    df = df_original.copy()    
    # 2. Lista de colunas para remover (ruídos contábeis)
    colunas_para_remover = [
        "Grupo", "Codigo", "Descricao", "Saldo Anterior", 
        "Total de Debitos", "Total de Creditos", "Saldo Atual", "Perc", "Valor"
    ]    
    # Remove as colunas inúteis (ignora caso alguma já não exista)
    return df.drop(columns=colunas_para_remover, errors="ignore")

def agrupar_por_mes_ano_conta(dfWork):
    
    registro_temporal = {}

    for _, row in dfWork.iterrows():
        ano = int(row["Ano"])
        mes_num = int(row["Mes_Num"])
        mes_nome = str(row["Mes"])
        conta = int(row["Conta"])
        valor = row["Valor_Limpo"]
        
        chave = (ano, mes_num)
        if chave not in registro_temporal:
            registro_temporal[chave] = {
                "Ano": ano,
                "Mes_Num": mes_num,
                "Mes_Nome": mes_nome,
                "Receita_Bruta": 0.0,
                "Custos_Variaveis": 0.0,
                "Despesas_Operacionais": 0.0
            }
        
        # Atribui o valor na coluna correta conforme o código contábil
        if conta == 3000000:
            registro_temporal[chave]["Receita_Bruta"] += valor        
        elif conta == 5000000:
            registro_temporal[chave]["Receita_Bruta"] += valor
        elif conta == 6111000:
            registro_temporal[chave]["Receita_Bruta"] += valor              
        elif conta == 3140000:
            registro_temporal[chave]["Custos_Variaveis"] += valor
        elif conta == 4000000:
            registro_temporal[chave]["Despesas_Operacionais"] += valor

    # Converte o dicionário agrupado no DataFrame de treinamento final
    dfResult = pd.DataFrame(registro_temporal.values())
    dfResult = dfResult.sort_values(by=["Ano", "Mes_Num"]).reset_index(drop=True)

    #Formata os valores financeiros para duas casas decimais
    colunas_financeiras = ["Receita_Bruta", "Custos_Variaveis", "Despesas_Operacionais"]
    dfResult[colunas_financeiras] = dfResult[colunas_financeiras].round(2)

    # Adiciona uma coluna de índice temporal para facilitar a análise e o modelo de previsão
    dfResult["Indice_Novo"] = range(1, len(dfResult) + 1)

    return dfResult

def tratando_valores_nulos(dfWork):   
    dfWork["Receita_Bruta"] = dfWork["Receita_Bruta"].replace(0, np.nan).fillna(dfWork["Receita_Bruta"].median())
    dfWork["Custos_Variaveis"] = dfWork["Custos_Variaveis"].replace(0, np.nan).fillna(dfWork["Custos_Variaveis"].median())
    dfWork["Despesas_Operacionais"] = dfWork["Despesas_Operacionais"].replace(0, np.nan).fillna(dfWork["Despesas_Operacionais"].median())
    return dfWork

def feature_engineering(dfWork):   
    # Alvo principal: Margem de contribuição real em reais
    dfWork["Margem_Contribuicao"] = dfWork["Receita_Bruta"] - dfWork["Custos_Variaveis"]

    # Criação de features de média móvel para suavizar os efeitos sazonais da entressafra
    dfWork["Media_Movel_Receita_3M"] = dfWork["Receita_Bruta"].rolling(window=3, min_periods=1).mean()
    dfWork["Media_Movel_Custos_3M"] = dfWork["Custos_Variaveis"].rolling(window=3, min_periods=1).mean()
    dfWork["Media_Movel_Despesas_3M"] = dfWork["Despesas_Operacionais"].rolling(window=3, min_periods=1).mean()

    return dfWork

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def treinamento_80_20(df_tidy):
    # 1. Garante a ordenação cronológica estrita
    df_tidy = df_tidy.sort_values("Indice_Novo").reset_index(drop=True)
    
    # 2. CALCULA A MARGEM NO LUGAR CERTO (Agora ela existe!)
    df_tidy["Margem_Contribuicao"] = (df_tidy["Receita_Bruta"] - df_tidy["Custos_Variaveis"]).round(2)
    
    # 3. ENGENHARIA DE RECURSOS AVANÇADA
    # A. Transformação Cíclica do Mês (Seno e Cosseno)
    df_tidy["Mes_Sin"] = np.sin(2 * np.pi * df_tidy["Mes_Num"] / 12)
    df_tidy["Mes_Cos"] = np.cos(2 * np.pi * df_tidy["Mes_Num"] / 12)
    
    # C. Médias Móveis já validadas
    df_tidy["Media_Movel_Receita_3M"] = df_tidy["Receita_Bruta"].rolling(window=3, min_periods=1).mean()
    df_tidy["Media_Movel_Custos_3M"] = df_tidy["Custos_Variaveis"].rolling(window=3, min_periods=1).mean()
    df_tidy["Media_Movel_Despesas_3M"] = df_tidy["Despesas_Operacionais"].rolling(window=3, min_periods=1).mean()
    
    # 4. DIVISÃO TEMPORAL CRONOLÓGICA (80% treino / 20% teste)
    ponto_corte = int(len(df_tidy) * 0.8)
    train_df = df_tidy.iloc[:ponto_corte]
    test_df = df_tidy.iloc[ponto_corte:]
    
    # 5. LISTA DE FEATURES ROBUSTA
    features = [
        "Ano",
        "Indice_Novo", 
        "Mes_Sin", 
        "Mes_Cos",        
        "Media_Movel_Receita_3M", 
        "Media_Movel_Custos_3M", 
        "Media_Movel_Despesas_3M",
    ]
    
    X_train = train_df[features]
    y_train = train_df["Margem_Contribuicao"]
    
    X_test = test_df[features]
    y_test = test_df["Margem_Contribuicao"]
    
    return X_train, X_test, y_train, y_test

import os
import json
import pickle
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metricas_modelo(X_train_scaled, X_test_scaled, y_train, y_test, dfWork, versao):
    
    # 1. ALGORITMO RIDGE CALIBRADO PARA SÉRIES HARMÔNICAS
    #model = Ridge(alpha=1.0) 
    model = LinearRegression() 
    model.fit(X_train_scaled, y_train)  

    # 2. PREDICÕES NA ESCALA REAL
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # 3. MÉTODOS DE AVALIAÇÃO
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)

    # 4. EXPORTAÇÃO DOS ARTEFATOS DO MANUAL DO PROJETO
    os.makedirs(f"models/{versao}", exist_ok=True)

    with open(f"models/{versao}/modelo_regressao.pkl", "wb") as f:
        pickle.dump(model, f)

    metricas_dict = {
        "Variaveis_X": [
            "Ano",
            "Indice_Novo",
            "Mes_Sin",
            "Mes_Cos",            
            "Media_Movel_Receita_3M",
            "Media_Movel_Custos_3M",
            "Media_Movel_Despesas_3M"           
        ],
        "Metricas_Teste": {
            "MAE": float(mae_test), 
            "MSE": float(mse_test), 
            "RMSE": float(rmse_test), 
            "R2": float(r2_test)
        },
        "Metricas_Treino": {
            "MAE": float(mae_train), 
            "MSE": float(mse_train),
            "RMSE": float(rmse_train),
            "R2": float(r2_train)
        }
    }
    
    with open(f"models/{versao}/metricas.json", "w") as f:
        json.dump(metricas_dict, f, indent=4)
        
    print(f"🎯 Sucesso! Pipeline restaurado com a engenharia de Seno/Cosseno")
    return model


def getDadosBrutos():       
    arquivos_config = {
        "2020": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2020.CSV",
        "2021": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2021.CSV",
        "2022": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2022.CSV",
        "2023": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2023.CSV",
        "2024": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2024.CSV",
        "2025": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2025.CSV",
        "2026": "https://github.com/jcbdoliveira/SCTec-Projeto_Avaliativo_T2/raw/refs/heads/main/DadosBrutos/EXPORTA_2026.CSV"
    }
    
    mesesNomes = ["Janeiro","Fevereiro","Marco","Abril","Maio","Junho",
            "Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
    
    mesesNumeros = {
                "Janeiro":1,"Fevereiro":2,"Marco":3,"Abril":4,"Maio":5,"Junho":6,
                "Julho":7,"Agosto":8,"Setembro":9,"Outubro":10,"Novembro":11,"Dezembro":12
    }

    contasDesejadas = ["3110000","3140000","4000000","5000000","6111000"]

    destino = f"data/raw/Dados.csv" 

    dfs = []
    for ano, url in arquivos_config.items():    
        print(f"Baixando arquivo de dados brutos para o ano {ano}...")
        df = pd.read_csv(url, sep=";", decimal=",")
        df["Ano"] = ano  # adiciona coluna com ano do arquivo
        
        # Filtra pelos códigos desejados
        df = df[df["Conta"].astype(str).isin(contasDesejadas)]
        dfs.append(df)

    # Junta todos os DataFrames
    dfTotal = pd.concat(dfs, ignore_index=True)
            
    dfAuxiliar = dfTotal.melt(
                        id_vars=["Grupo","Codigo","Conta","Descricao",
                        "Saldo Anterior","Total de Debitos","Total de Creditos","Saldo Atual","Perc", "Ano"],
                        value_vars=mesesNomes,
                        var_name="Mes",
                        value_name="Valor"
    )

    dfAuxiliar["Mes_Num"] = dfAuxiliar["Mes"].map(mesesNumeros)
    dfAuxiliar.to_csv(destino, sep=";", index=False)

    return destino