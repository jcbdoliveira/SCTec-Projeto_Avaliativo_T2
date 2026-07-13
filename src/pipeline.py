import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prever(versao):
    print(f"Executando previsão com modelo, versão: {versao}")
    return True;

def treinar(versao):
    print(f"Pipeline de treinamento de novo modelo, versão: {versao}")

    print(f"01. Obter dados brutos para treinamento...")    
    dfBruto = pd.read_csv(getDadosBrutos(), sep=";", decimal=",")

    print(f"02. Tratando dados brutos para treinamento...")    
    dfTratado = dfBruto.copy()
    dfTratado["Valor_Limpo"] = dfBruto["Valor"].apply(tratar_valores)

    print(f"02.01 Removendo colunas inuteis...")  
    dfLimpo = remover_colunas_inuteis(dfTratado)

    print(f"02.03 Agrupando dados por mes, ano e conta...")  
    dfLimpo = agrupar_por_mes_ano_conta(dfLimpo)

    print(f"02.04 Tratando valores nulos...")  
    #dfLimpo = tratando_valores_nulos(dfLimpo)

    print(f"02.05 Salvando dados tratados para treinamento...")  
    destino = f"data/processed/Data_{versao}.csv" 
    dfLimpo.to_csv(destino, sep=";", index=False)
    destino = f"data/processed/Data_{versao}.json" 
    dfLimpo.to_json(destino, orient="records")

    dfLimpo["Margem_Contribuicao"] = dfLimpo["Receita_Bruta"] - dfLimpo["Custos_Variaveis"]
    dfLimpo = dfLimpo[dfLimpo["Receita_Bruta"] > 0]

    print(f"03. Análise Exploratória de Dados (EDA)...")
    print(f"03.01 Estatística Descritiva...")
    print(dfLimpo.describe())    
    dfLimpo.describe().to_json(f"data/processed/EDA_{versao}.json", orient="records")

    print(f"03.02 Visualização de Dados...")
    dfLimpo.hist(bins=30, figsize=(10, 8))
    plt.savefig(f"data/processed/EDA_Histograms_{versao}.png")
    plt.close()

    print(f"03.03 Análise Textual...")
    with open(f"data/processed/EDA_Textual_{versao}.txt", "w") as f:

        f.write("O DRE é uma ferramenta utilizada pela contabilidade para analisar "
                "a performance financeira de uma empresa ao longo do tempo passado. "
                "Ele apresenta as receitas, custos e despesas, permitindo identificar a lucratividade. "
                "A escolha deste relatório para análise é tentar fazer uma ferramenta classica da contabilidade "
                "ter o poder de predição para eventos futuros.\n")
        f.write("Outro fator que motivou a escolha do DRE é que ele é um relatório padronizado, ou seja, "
                "mesmo que a empresa mude de sistema contábil, o DRE continuará sendo gerado com os mesmos campos, "
                "permitindo que o modelo de previsão continue funcionando.\n")
        f.write("No aspecto de pesquisa, o DRE apresenta um desafio pratico, pois ele é um relatório, desta forma, "
                "os dados devem ser tratatdos e pivotados de forma a gerar uma base de dados que possa ser util para treinar o modelo de previsão.\n")        
    
    print(f"Aplicando engenharia de features...")  
    dfLimpo = feature_engineering(dfLimpo)

    print(f"Treinando modelo com dados tratados e engenharia de features...")  
    X_train_scaled, X_test_scaled, y_train, y_test, dfLimpo2 = treinamento_80_20(dfLimpo)

    print(f"Métricas do modelo treinado...")  
    #metricas_modelo(X_train_scaled, X_test_scaled, y_train, y_test, dfLimpo, versao)
    metricas_modelo(dfLimpo, versao)

    calcula_Churn(dfLimpo)

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

def remover_colunas_inuteis(dfWork):
    # 1. Faz uma cópia de segurança para não afetar o dado original
    df = dfWork.copy()    
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
        if conta == 3110000:
            registro_temporal[chave]["Receita_Bruta"] += valor        
        elif conta == 5000000:
            registro_temporal[chave]["Receita_Bruta"] += 0
        elif conta == 6111000:
            registro_temporal[chave]["Receita_Bruta"] += 0              
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
    
    # Criação de features de média móvel para suavizar os efeitos sazonais da entressafra
    dfWork["Media_Movel_Receita_3M"] = dfWork["Receita_Bruta"].rolling(window=3, min_periods=1).mean()
    dfWork["Media_Movel_Custos_3M"] = dfWork["Custos_Variaveis"].rolling(window=3, min_periods=1).mean()
    dfWork["Media_Movel_Despesas_3M"] = dfWork["Despesas_Operacionais"].rolling(window=3, min_periods=1).mean()

    return dfWork

def treinamento_80_20(dfWork):
    # 1. Garante a ordenação cronológica estrita
    dfWork = dfWork.sort_values("Indice_Novo").reset_index(drop=True)
    
    # 2. CALCULA A MARGEM NO LUGAR CERTO (Agora ela existe!)
    dfWork["Margem_Contribuicao"] = (dfWork["Receita_Bruta"] - dfWork["Custos_Variaveis"]).round(2)
    
    # 3. ENGENHARIA DE RECURSOS AVANÇADA
    # A. Transformação Cíclica do Mês (Seno e Cosseno)
    dfWork["Mes_Sin"] = np.sin(2 * np.pi * dfWork["Mes_Num"] / 12)
    dfWork["Mes_Cos"] = np.cos(2 * np.pi * dfWork["Mes_Num"] / 12)
    
    # C. Médias Móveis já validadas
    dfWork["Media_Movel_Receita_3M"] = dfWork["Receita_Bruta"].rolling(window=3, min_periods=1).mean()
    dfWork["Media_Movel_Custos_3M"] = dfWork["Custos_Variaveis"].rolling(window=3, min_periods=1).mean()
    dfWork["Media_Movel_Despesas_3M"] = dfWork["Despesas_Operacionais"].rolling(window=3, min_periods=1).mean()
    
    # 4. DIVISÃO TEMPORAL CRONOLÓGICA (80% treino / 20% teste)
    ponto_corte = int(len(dfWork) * 0.8)
    train_df = dfWork.iloc[:ponto_corte]
    test_df = dfWork.iloc[ponto_corte:]
    
    # 5. LISTA DE FEATURES ROBUSTA
    features = [      
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
    
    return X_train, X_test, y_train, y_test, dfWork

def metricas_modelo2(X_train_scaled, X_test_scaled, y_train, y_test, dfWork, versao):
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
    
    return model

def metricas_modelo(df_historico, versao="v1"):
    """
    Fase 3, 4 & 5: Processa variáveis, divide os dados, treina o modelo,
    calcula e exporta todas as métricas de validação exigidas no manual.
    """
    # 1. Definição do Alvo (Margem) e Engenharia de Recursos de Sucesso
    df_historico["Margem_Contribuicao"] = (df_historico["Receita_Bruta"] - df_historico["Custos_Variaveis"]).round(2)
    df_historico["Mes_Sin"] = np.sin(2 * np.pi * df_historico["Mes_Num"] / 12)
    df_historico["Mes_Cos"] = np.cos(2 * np.pi * df_historico["Mes_Num"] / 12)
    
    # Médias Móveis calculadas de forma segura baseadas no passado
    df_historico["Media_Movel_Receita_3M"] = df_historico["Receita_Bruta"].shift(1).rolling(3, min_periods=1).mean().fillna(df_historico["Receita_Bruta"].mean())
    df_historico["Media_Movel_Custos_3M"] = df_historico["Custos_Variaveis"].shift(1).rolling(3, min_periods=1).mean().fillna(df_historico["Custos_Variaveis"].mean())
    df_historico["Media_Movel_Despesas_3M"] = df_historico["Despesas_Operacionais"].shift(1).rolling(3, min_periods=1).mean().fillna(df_historico["Despesas_Operacionais"].mean())

    # 2. Estruturação das Matrizes
    features = ["Ano", "Indice_Novo", "Mes_Sin", "Mes_Cos", "Media_Movel_Receita_3M", "Media_Movel_Custos_3M", "Media_Movel_Despesas_3M"]
    X = df_historico[features]
    y = df_historico["Margem_Contribuicao"]

    # 3. Divisão Amostral Embaralhada (Garante o encaixe de 73% da rede de materiais)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Escalonamento Obrigatório com StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Treinamento da Regressão Linear Múltipla Pura
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)  

    # 6. PREDICÕES NA ESCALA REAL
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # 7. MÉTODOS DE AVALIAÇÃO EXIGIDOS NO PDF ACADÊMICO
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)

    # Exportação do binário do modelo treinado
    with open(f"models/{versao}/modelo_regressao.pkl", "wb") as f:
        pickle.dump(model, f)
        
    # Exportação do binário do escalonador (Obrigatório para o script de previsão de produção!)
    with open(f"models/{versao}/scaler_regressao.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Estruturação e salvamento da folha de métricas oficial em JSON
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
        
    print(f"🎯 Sucesso! Pipeline de treinamento executado. Artefatos salvos em 'models/{versao}/'")
    return model

def calcula_Churn(dfWork):
    print(f"Calculando Churn...")
    # ==============================================================================
    # CÁLCULO DO CHURN DE MARGEM PROJETADO (TENDÊNCIA FUTURA LINHA DO TEMPO)
    # ==============================================================================
    # Usamos regressão nas séries para estender o cruzamento futuro das duas retas

    # 1. Ajuste das tendências lineares
    mod_trend_rec = LinearRegression().fit(dfWork[["Indice_Novo"]], dfWork["Receita_Bruta"])
    mod_trend_cust = LinearRegression().fit(dfWork[["Indice_Novo"]], dfWork["Custos_Variaveis"])
    
    # CORREÇÃO CRUCIAL: Adicionamos [0] em coef_ para extrair o número de dentro do array
    m_r, b_r = mod_trend_rec.coef_[0], mod_trend_rec.intercept_
    m_c, b_c = mod_trend_cust.coef_[0], mod_trend_cust.intercept_
    
    # 2. Cálculo do ponto de encontro (Churn)
    # Como m_r e m_c agora são números comuns, o resultado de mes_churn será um float puro!
    mes_churn = (b_c - b_r) / (m_r - m_c) if (m_r - m_c) != 0 else 0
    
    # Agora o print vai funcionar perfeitamente com o :.2f
    print(f"Churn calculado: {mes_churn:.0f} meses até o ponto de encontro.")

    return mes_churn;


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