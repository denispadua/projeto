###############################################################################
# tsp_decoder.py: simple permutation decoder for the Traveling Salesman Problem.
#
# (c) Copyright 2019, Carlos Eduardo de Andrade. All Rights Reserved.
#
# This code is released under LICENSE.md.
#
# Created on:  Nov 18, 2019 by ceandrade
# Last update: Nov 18, 2019 by ceandrade
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
###############################################################################

from copy import copy
from brkga_mp_ipr.types import BaseChromosome
from tsp_instance import TSPInstance
import numpy as np
import pandas as pd


class TSPDecoder():

    def __init__(self, instance: TSPInstance, qtd_grupos: int, qtd_variaveis: int, qtd_min_por_grupo = 0):
        self.instance = instance
        self.df_original = self.instance.df.copy()
        self.qtd_grupos = qtd_grupos
        self.qtd_variaveis = qtd_variaveis
        self.qtd_min_por_grupo = qtd_min_por_grupo
        self.dict = {}
        self.dict_grupos = {}
    
    def calcular_alfa(self, row, columns):
        soma_linha = sum([row[i]*i for i in columns])
        taxas_2 = [i*i for i in columns]
        linhas_sem_taxas = sum(row[i] for i in columns)

        n = 5*soma_linha-sum(columns) * linhas_sem_taxas
        d = 5*sum(taxas_2) - sum(columns)**2
        return (n / d)*100
    
    def calcular_chave(self, row):
        chave = 0
        for i, valor in enumerate(row):
            chave += valor * (10 ** i)
        return chave

    def calcular_grupo(self, row):
        return self.dict[int(row['CHAVE'])]
    
    def retornar_grupo_final(self, row):
        return int(self.dict_grupos[int(row['CHAVE'])])
    
    def mapear_grupo_final(self, row):
        if self.dict_grupos.get(row["CHAVE"], None) is None:
            self.dict_grupos[row["CHAVE"]] = row["GRUPO_FINAL"]

    def mapear_grupo(self, row):
        if self.dict_grupos.get(row["CHAVE"], None) is None:
            self.dict_grupos[row["CHAVE"]] = row["GRUPOS"]
    
    def definir_grupo(self, row, idx):
        return 1 if idx == row["GRUPO_FINAL"] else 0

    def distribui_valores(self, tamanho_lista, valores):
        quociente, resto = divmod(tamanho_lista, len(valores))

        lista_distribuida = []
        for valor in valores:
            lista_distribuida.extend([valor] * quociente)

        lista_distribuida += [len(valores)] * resto
        return lista_distribuida
    ###########################################################################

    def definir_grupo_provisorio(self, row):
        return self.dict_grupos.get(row['CHAVE'])
    
    def calcular_grupo_final(self, grupos_provisorios, alfa_list):
        alfa_list.sort()
        
        for i in range(0, len(grupos_provisorios)-1):
            if grupos_provisorios[i] != grupos_provisorios[i+1]:
                if abs(alfa_list[i+1] - alfa_list[i]) < abs(alfa_list[i] - alfa_list[i-1]):
                    grupos_provisorios[i] = grupos_provisorios[i+1]
            elif i > 0 and grupos_provisorios[i] != grupos_provisorios[i-1]:
                if abs(alfa_list[i] - alfa_list[i-1]) < abs(alfa_list[i+1] - alfa_list[i]):
                    grupos_provisorios[i] = grupos_provisorios[i-1]
                
        return grupos_provisorios
    
    def setar_grupo_final(self, row):
        return self.dict_grupos[row["CHAVE"]]
    
    def gerar_alfa_cliente(self, taxas):
        chave_taxa = pd.pivot_table(self.instance.df[["CHAVE", "Taxa"]], index='CHAVE', columns='Taxa', aggfunc=len, fill_value=0)
        chave_taxa = pd.DataFrame(chave_taxa, columns=taxas)

        efetivado = pd.pivot_table(self.instance.df[["CHAVE", "Taxa", "Flag_Efet"]], index='CHAVE', columns=['Taxa'], aggfunc=sum, fill_value=0)
        efetivado.columns = taxas
        chave_efetivado = pd.DataFrame(efetivado, columns=taxas)
        df_div = chave_efetivado.div(chave_taxa)

        
        chave_taxa_alfa = df_div.apply(lambda row: self.calcular_alfa(row, taxas), axis=1)
        self.dict = chave_taxa_alfa.to_dict()   
        tamanho = chave_taxa_alfa.count()
        chave_taxa_alfa = pd.DataFrame(chave_taxa_alfa, columns=["ALFA"]).reset_index("CHAVE")
        chave_taxa_alfa.sort_values("ALFA", inplace=True)
        chave_taxa_alfa.reset_index(inplace=True)
        return chave_taxa_alfa, tamanho


    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:

        self.instance.df = self.df_original.copy()
        self.dict = {}
        
        #transforma os cromossomos recebidos para 0 e 1 
        cromossomos_feature = [1 if value > 0.5 else 0 for value in chromosome[0:self.qtd_variaveis]]

        #multiplica as variáveis pelos cromossomos.
        colunas = ['Compr_Renda', 'Nivel_Escolaridade', 'Estado_Civil', 'Regiao', 'Nivel_Risco_Novo']
        self.instance.df[colunas] = self.instance.df[colunas]*cromossomos_feature

        #Gera as chaves
        self.instance.df["CHAVE"] = self.instance.df[colunas].apply(self.calcular_chave, axis=1)
        # self.instance.df.reset_index()

        #lista_de_taxas ordenadas
        taxas = self.instance.df['Taxa'].unique().tolist()
        taxas.sort()
        
        chaves_alfa, tamanho = self.gerar_alfa_cliente(taxas)
        tamanho = self.qtd_grupos

        self.instance.df["ALFA"] = self.instance.df.apply(lambda row: self.calcular_grupo(row), axis=1)

        grupos_provisorios = self.distribui_valores(len(chaves_alfa.index), [i+1 for i in range(tamanho)])
        df_provisorio = pd.DataFrame(grupos_provisorios, columns=["GRUPOS"])
        concatenado = pd.concat([chaves_alfa, df_provisorio], axis=1)
        del concatenado["index"]

        concatenado.apply(self.mapear_grupo, axis=1)
        # self.instance.df.sort_values("CHAVE", inplace=True)
        #Cria um dataframe de cromossomos G1,G2,G3 e G4

        # self.instance.df["GRUPO_PROVISORIO"] = self.instance.df.apply(self.definir_grupo_provisorio, axis=1)

        alfa_list = concatenado["ALFA"].to_list()
        grupos_provisorios = concatenado["GRUPOS"].to_list()

        concatenado["GRUPO_FINAL"] = self.calcular_grupo_final(grupos_provisorios, alfa_list)

        # grupo_final = self.calcular_grupo_final(grupos_provisorios, alfa_list)

        # self.instance.df['GRUPO_FINAL'] = grupo_final
    
        #cria um dataframe com a multiplicação da coluna dos grupos pela coluna "Flag_Efet"

        self.dict_grupos = {}
        concatenado.apply(self.mapear_grupo_final, axis=1)


        self.instance.df["GRUPO_FINAL"] = self.instance.df[["CHAVE"]].apply(self.retornar_grupo_final, axis=1)

        grupos = {}
        for i in range(0, tamanho):
            grupos[f"G{i+1}"] = self.instance.df[["CHAVE","GRUPO_FINAL"]].apply(lambda row: self.definir_grupo(row, i+1), axis=1)

        self.instance.df.reset_index(inplace=True)
        g = pd.DataFrame(grupos)
        self.instance.df.reset_index(drop=True, inplace=True)
        g.reset_index(drop=True, inplace=True)
        del self.instance.df["index"]
        self.instance.df = pd.concat([self.instance.df, g], axis=1)
        efetivados = {}
        for i in range(0, tamanho):
            efetivados[f"E{i+1}"] = self.instance.df[f"G{i+1}"] * self.instance.df["Flag_Efet"]
        
        e = pd.DataFrame(efetivados)
        e.reset_index(drop=True, inplace=True)
        self.instance.df = pd.concat([self.instance.df, e],axis=1)

        #Conta a quantidade de grupos por taxa
        contagem_grupos = (self.instance.df.set_index('Taxa').filter(regex='G[0-9]+').eq(1)
         .groupby(level='Taxa').sum()
        )
        ls_contagem = np.array(contagem_grupos)
        taxas = self.instance.df['Taxa'].unique().tolist()
        taxas.sort()

        tabela_unificada = pd.DataFrame(ls_contagem, columns=[f"G{i+1}" for i in range(0, tamanho)])
        tabela_unificada.index = taxas
        ###########################################

        #conta a quantidade de efetivados por taxa
        contagem_efetivados = (self.instance.df.set_index('Taxa').filter(regex='E[0-9]+').eq(1)
         .groupby(level='Taxa').sum()
        )
        ls_efetivados = np.array(contagem_efetivados)
        tabela_efetivados = pd.DataFrame(ls_efetivados, columns=[f"G{i+1}" for i in range(0, tamanho)])
        tabela_efetivados.index = taxas
        ############################################

        #gera a tabela de percentual grupo por taxa
        divisao = tabela_efetivados.div(tabela_unificada).reset_index()
        divisao = divisao.apply(lambda c: round(c,3)).fillna(0)
        divisao.index = taxas
        del divisao['index']
        divisao = divisao.transpose()
        
        #calcula o alfa
        divisao["Alfa"] = divisao.apply(lambda row: self.calcular_alfa(row, taxas), axis=1)
        #converte o alfa e ordena
        item = divisao["Alfa"].to_list()
        item.sort()

        #calcula a diferença entre os alfas
        soma = 0
        for idx in range(0,len(item)-1):
            soma += round(item[idx+1],2) - round(item[idx],2)

        penalizacao = 0
        for i in range(0, tamanho):
            total = self.instance.df[f"G{i+1}"].sum()
            if total < self.qtd_min_por_grupo:
                penalizacao += 1000

        print(soma - penalizacao)
        return soma - penalizacao

