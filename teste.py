from pandas import DataFrame
from leitor_csv import LeitorCSV
import docopt
import random
import numpy as np
from math import pow
import pandas as pd

def tratamento_dados(dados: DataFrame):
    dict_compr_renda = {
        "5%-": 1,
        "5% a 10%": 2,
        "10% a 15%": 3,
        "15% a 20%": 4,
        "20% a 25%": 5,
        "25% a 30%": 6,
        "30%+": 7
    }
    dados['Compr_Renda'] = dados['Compr_Renda'].map(dict_compr_renda)

    dict_nivel_escolaridade = {
        "Med_e_Sup_Inc": 1,
        "Sup_e_Pos": 2
    }
    dados['Nivel_Escolaridade'] = dados['Nivel_Escolaridade'].map(dict_nivel_escolaridade)

    dict_estado_civil = {
        "Casado": 1,
        "Divorciado": 2,
        "Solteiro": 3,
        "Viuvo": 4
    }
    dados['Estado_Civil'] = dados['Estado_Civil'].map(dict_estado_civil)

    dict_regiao = {
        "Centro-Oeste": 1,
        "Nordeste": 2,
        "Norte": 3,
        "Sudeste": 4,
        "Sul": 5
    }
    dados['Regiao'] = dados['Regiao'].map(dict_regiao)

    dados['Nivel_Risco_Novo'] = dados['Nivel_Risco_Novo'].map(lambda v: v+1)

def calcular_alfa(row, columns):
    soma_linha = sum([row[i]*i for i in columns])
    taxas_2 = [i*i for i in columns]
    linhas_sem_taxas = sum(row[i] for i in columns)

    n = 5*soma_linha-sum(columns) * linhas_sem_taxas
    d = 5*sum(taxas_2) - sum(columns)**2

    return (n / d)*100

def calcular_chave(row):
  chave = 0
  for i, valor in enumerate(row):
    chave += valor * (10 ** i)
  return chave

def main():

    qtd_variaveis = 5
    qtd_grupos = 4
    instance = LeitorCSV("C:\\Users\\denis\\Desktop\\brkga_mp_ipr_python\\examples\\tsp\\instances\\Base_Otimização_Final.csv")
    instance.ler_csv()

    colunas_selecionadas = ['Compr_Renda', 'Nivel_Escolaridade', 'Taxa', 'Estado_Civil', 'Regiao', 'Flag_Efet', 'Nivel_Risco_Novo']
    colunas_removidas = [col for col in instance.df.columns if col not in colunas_selecionadas]
    instance.df.drop(colunas_removidas, axis=1, inplace=True)
    
    tratamento_dados(instance.df)

    # #quantidade de variáveis vai como tamanho dos cromossomos
    # tamanho_cromossomo = (qtd_grupos*len(instance.df.index))+qtd_variaveis

    cromossomos = [1 if random.random() > 0.5 else 0 for _ in range(0,(qtd_grupos*len(instance.df.index))+qtd_variaveis)]

    print(len(instance.df.index), len(cromossomos))
    colunas = ['Compr_Renda', 'Nivel_Escolaridade', 'Estado_Civil', 'Regiao', 'Nivel_Risco_Novo']
    colunas_grupos = [f"G{i+1}" for i in range(0, qtd_grupos)]

    instance.df[colunas] = instance.df[colunas]*cromossomos[0:qtd_variaveis]

    cromossomos = np.matrix(cromossomos[qtd_variaveis:])
    cromossomos = cromossomos.reshape(len(instance.df.index),qtd_grupos)
    
    instance.df["CHAVE"] = instance.df[colunas].apply(calcular_chave, axis=1)
    instance.df.reset_index()
    # print(instance.df)

    cromossomos_df = pd.DataFrame(cromossomos, columns=colunas_grupos)
    cromossomos_df.reset_index()
    # print(cromossomos_df)

    instance.df = pd.concat([instance.df, cromossomos_df], axis=1)
    # print(instance.df)

    for i in range(0,qtd_grupos):
        instance.df[f"E{i+1}"] = instance.df[f"G{i+1}"] * instance.df["Flag_Efet"]
    # print(instance.df)

        #Conta a quantidade de grupos por taxa
    ls_contagem = []
    for i in range(0, qtd_grupos):
        result = instance.df.groupby([f"G{i+1}", "Taxa"]).size().drop(0)
        ls_contagem.append(result)
    ls_contagem = np.array(ls_contagem)

    tabela_unificada = pd.DataFrame(ls_contagem, columns=instance.df['Taxa'].unique().tolist())

    #conta a quantidade de efetivados por taxa
    ls_efetivados = []
    for i in range(0, qtd_grupos):
        result = instance.df.groupby([f"E{i+1}", "Taxa"]).size().drop(0)
        ls_efetivados.append(result)
    ls_efetivados = np.array(ls_efetivados)

    tabela_efetivados = pd.DataFrame(ls_efetivados, columns=instance.df['Taxa'].unique().tolist())

    #gera a tabela de percentual grupo por taxa
    divisao = tabela_efetivados.div(tabela_unificada).reset_index()
    divisao = divisao.apply(lambda c: round(c,3))

    #conta a quantidade de grupos cada cliente com a mesma chave participa
    df_melt = instance.df.melt(id_vars='CHAVE', value_vars=colunas_grupos, value_name='presenca')
    contagem_grupos = df_melt.groupby(['CHAVE', 'variable']).agg(sum_presenca=('presenca', 'sum')).unstack()
    contagem_grupos.columns = ['_'.join(col) for col in contagem_grupos.columns]

    listas = []
    for i in range(0, qtd_grupos):
        listas.append(contagem_grupos[f"sum_presenca_G{i+1}"].to_list())

    matriz = np.matrix(listas)
    matriz = matriz.transpose()

    #verifica se clientes com a mesma chave estão em mais de um grupo
    for item in matriz:
        soma_grupo_chave = sum(item.tolist()[0])
        if soma_grupo_chave not in item.tolist()[0]:
            print(np.inf)

    #se os clientes não estão em mais de um grupo, calcula o alfa
    divisao["Alfa"] = divisao.apply(lambda row: calcular_alfa(row, instance.df['Taxa'].unique().tolist()), axis=1)
    #converte o alfa e ordena
    item = divisao["Alfa"].to_list()
    item.sort()

    #calcula a diferença entre os alfas
    soma = 0
    for idx in range(0,len(item)-1):
        soma += round(item[idx+1],2) - round(item[idx],2)

    return soma

if __name__ == "__main__":
    main()