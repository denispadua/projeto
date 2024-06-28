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

from brkga_mp_ipr.types import BaseChromosome
from tsp_instance import TSPInstance
import numpy as np
import pandas as pd


class TSPDecoder():
    
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

    def __init__(self, instance: TSPInstance, qtd_grupos: int, qtd_variaveis: int, qtd_min_por_grupo = 0):
        self.instance = instance
        self.df_original = self.instance.df.copy()
        self.qtd_grupos = qtd_grupos
        self.qtd_variaveis = qtd_variaveis
        self.qtd_min_por_grupo = qtd_min_por_grupo
        self.dict = {}
    
    def calcular_grupo(self, row):
        if self.dict.get(row['CHAVE'], None) is None:
            self.dict[row['CHAVE']] = row['GRUPO_PROVISORIO']
        return self.dict[row['CHAVE']]
    
    def definir_grupo(self, row, idx):
        return 1 if idx == self.dict[row['CHAVE']] else 0

    ###########################################################################

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
        self.instance.df.reset_index()

        cromossomos = chromosome[self.qtd_variaveis:]
        grupos_provisorios = []
        divisao_partes = [{key:(1/self.qtd_grupos)*key} for key in range(1,self.qtd_grupos+1)]
        for cromossomo in cromossomos:
            for key, value in enumerate(divisao_partes):
                if cromossomo <= value[key+1]:
                    break
            grupos_provisorios.append(key+1)


        #Cria um dataframe de cromossomos G1,G2,G3 e G4
        cromossomos_df = pd.DataFrame(cromossomos, columns=['CROMOSSOMOS'])
        cromossomos_df.reset_index()
        grupo_provisorio = pd.DataFrame(grupos_provisorios, columns=['GRUPO_PROVISORIO'])
        grupo_provisorio.reset_index()

        self.instance.df = pd.concat([self.instance.df, cromossomos_df, grupo_provisorio], axis=1)

        self.instance.df['GRUPO_FINAL'] = self.instance.df[['CHAVE', 'GRUPO_PROVISORIO']].apply(self.calcular_grupo, axis=1)
    
        #cria um dataframe com a multiplicação da coluna dos grupos pela coluna "Flag_Efet"
        for i in range(0, self.qtd_grupos):
            self.instance.df[f"G{i+1}"] = self.instance.df[["CHAVE","GRUPO_FINAL"]].apply(lambda row: self.definir_grupo(row, i+1), axis=1)

        for i in range(0,self.qtd_grupos):
            self.instance.df[f"E{i+1}"] = self.instance.df[f"G{i+1}"] * self.instance.df["Flag_Efet"]

        #Conta a quantidade de grupos por taxa
        contagem_grupos = (self.instance.df.set_index('Taxa').filter(regex='G[0-9]').eq(1)
         .groupby(level='Taxa').sum()
        )
        ls_contagem = np.array(contagem_grupos)
        taxas = self.instance.df['Taxa'].unique().tolist()
        taxas.sort()

        tabela_unificada = pd.DataFrame(ls_contagem, columns=[f"G{i+1}" for i in range(0, self.qtd_grupos)])
        tabela_unificada.index = taxas

        #conta a quantidade de efetivados por taxa
        contagem_efetivados = (self.instance.df.set_index('Taxa').filter(regex='E[0-9]').eq(1)
         .groupby(level='Taxa').sum()
        )
        ls_efetivados = np.array(contagem_efetivados)
        tabela_efetivados = pd.DataFrame(ls_efetivados, columns=[f"G{i+1}" for i in range(0, self.qtd_grupos)])
        tabela_efetivados.index = taxas


        #gera a tabela de percentual grupo por taxa
        divisao = tabela_efetivados.div(tabela_unificada).reset_index()
        divisao = divisao.apply(lambda c: round(c,3)).fillna(0)
        divisao.index = taxas
        del divisao['index']
        divisao = divisao.transpose()
        
        #se os clientes não estão em mais de um grupo, calcula o alfa
        divisao["Alfa"] = divisao.apply(lambda row: self.calcular_alfa(row, taxas), axis=1)
        #converte o alfa e ordena
        item = divisao["Alfa"].to_list()
        item.sort()

        #calcula a diferença entre os alfas
        soma = 0
        for idx in range(0,len(item)-1):
            soma += round(item[idx+1],2) - round(item[idx],2)

        penalizacao = 0
        for i in range(0, self.qtd_grupos):
            total = self.instance.df[f"G{i+1}"].sum()
            if total < self.qtd_min_por_grupo:
                penalizacao += 1000

        return soma - penalizacao

