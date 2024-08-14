from pandas import DataFrame
from leitor_csv import LeitorCSV
import random
import numpy as np
from math import pow
import pandas as pd
from tsp_decoder import TSPDecoder

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
    instance = LeitorCSV("Sample.csv")
    instance.ler_csv()

    colunas_selecionadas = ['Compr_Renda', 'Nivel_Escolaridade', 'Taxa', 'Estado_Civil', 'Regiao', 'Flag_Efet', 'Nivel_Risco_Novo']
    colunas_removidas = [col for col in instance.df.columns if col not in colunas_selecionadas]
    instance.df.drop(colunas_removidas, axis=1, inplace=True)
    
    tratamento_dados(instance.df)

    # #quantidade de variáveis vai como tamanho dos cromossomos
    # tamanho_cromossomo = (qtd_grupos*len(instance.df.index))+qtd_variaveis

    cromossomos = [1, 0, 0, 0, 1]
    
    #ordem de parâmetros: instance, qtd_grupos, qtd_variaveis, alfa_medio, qtd_cliente_chave, qtd_min_por_grupo

    decoder = TSPDecoder(instance, 5, 5, 50, 20, 20)
    decoder.decode(cromossomos, False)

    

if __name__ == "__main__":
    main()