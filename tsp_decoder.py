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

    def calcular_penalizacao(self, row):
        soma = 0
        soma += sum(row)
        if soma:
            return (sum(row)-1)*100
        return soma + 100

    def __init__(self, instance: TSPInstance, qtd_grupos: int, qtd_variaveis: int):
        self.instance = instance
        self.df_original = self.instance.df.copy()
        self.qtd_grupos = qtd_grupos
        self.qtd_variaveis = qtd_variaveis

    ###########################################################################

    def decode(self, chromosome: BaseChromosome, rewrite: bool) -> float:

        self.instance.df = self.df_original.copy()
        
        #transforma os cromossomos recebidos para 0 e 1 
        cromossomos = [1 if value > 0.5 else 0 for value in chromosome]

        #multiplica as variáveis pelos cromossomos.
        colunas = ['Compr_Renda', 'Nivel_Escolaridade', 'Estado_Civil', 'Regiao', 'Nivel_Risco_Novo']
        self.instance.df[colunas] = self.instance.df[colunas]*cromossomos[0:self.qtd_variaveis]

        
        cromossomos = np.matrix(cromossomos[self.qtd_variaveis:])
        cromossomos = cromossomos.reshape(len(self.instance.df.index),self.qtd_grupos)

        #Gera as chaves
        self.instance.df["CHAVE"] = self.instance.df[colunas].apply(self.calcular_chave, axis=1)
        self.instance.df.reset_index()

        #Cria um dataframe de cromossomos G1,G2,G3 e G4
        colunas_grupos = [f"G{i+1}" for i in range(0, self.qtd_grupos)]
        cromossomos_df = pd.DataFrame(cromossomos, columns=colunas_grupos)
        cromossomos_df.reset_index()

        self.instance.df = pd.concat([self.instance.df, cromossomos_df], axis=1)

        self.instance.df["PENALIDADE"] = self.instance.df[colunas_grupos].apply(self.calcular_penalizacao, axis=1)

        penalidade_cliente = self.instance.df["PENALIDADE"].sum()        

        #cria um dataframe com a multiplicação da coluna dos grupos pela coluna "Flag_Efet"
        for i in range(0,self.qtd_grupos):
            self.instance.df[f"E{i+1}"] = self.instance.df[f"G{i+1}"] * self.instance.df["Flag_Efet"]

        #Conta a quantidade de grupos por taxa
        ls_contagem = []
        for i in range(0, self.qtd_grupos):
            result = self.instance.df.groupby([f"G{i+1}", "Taxa"]).size().drop(0)
            ls_contagem.append(result)
        ls_contagem = np.array(ls_contagem)

        tabela_unificada = pd.DataFrame(ls_contagem, columns=self.instance.df['Taxa'].unique().tolist())

        #conta a quantidade de efetivados por taxa
        ls_efetivados = []
        for i in range(0, self.qtd_grupos):
            result = self.instance.df.groupby([f"E{i+1}", "Taxa"]).size().drop(0)
            ls_efetivados.append(result)
        ls_efetivados = np.array(ls_efetivados)

        tabela_efetivados = pd.DataFrame(ls_efetivados, columns=self.instance.df['Taxa'].unique().tolist())

        #gera a tabela de percentual grupo por taxa
        divisao = tabela_efetivados.div(tabela_unificada).reset_index()
        divisao = divisao.apply(lambda c: round(c,3))

        #conta a quantidade de grupos cada cliente com a mesma chave participa
        df_melt = self.instance.df.melt(id_vars='CHAVE', value_vars=colunas_grupos, value_name='presenca')
        contagem_grupos = df_melt.groupby(['CHAVE', 'variable']).agg(sum_presenca=('presenca', 'sum')).unstack()
        contagem_grupos.columns = ['_'.join(col) for col in contagem_grupos.columns]

        listas = []
        for i in range(0, self.qtd_grupos):
            listas.append(contagem_grupos[f"sum_presenca_G{i+1}"].to_list())
        
        matriz = np.matrix(listas)
        matriz = matriz.transpose()

        #verifica se clientes com a mesma chave estão em mais de um grupo
        penalidade_grupo = 0
        fator_penalidade = 1000
        for item in matriz:
            soma_grupo_chave = sum(item.tolist()[0])
            diferenca_soma_maior = soma_grupo_chave - max(item.tolist()[0])
            if diferenca_soma_maior != 0:
                penalidade_grupo += fator_penalidade * (diferenca_soma_maior / soma_grupo_chave)
        
        #se eu quero minimizar, significa que a penalidade deve ser negativa, pois o resultado será positivo para gerações penalizadas
        #se eu quero maximizar, a penalizade deve ser positiva, pois o valor negativo vai ser multiplicado por um fator positivo, diminuindo o número

        #se os clientes não estão em mais de um grupo, calcula o alfa
        divisao["Alfa"] = divisao.apply(lambda row: self.calcular_alfa(row, self.instance.df['Taxa'].unique().tolist()), axis=1)
        #converte o alfa e ordena
        item = divisao["Alfa"].to_list()
        item.sort()

        #calcula a diferença entre os alfas
        soma = 0
        for idx in range(0,len(item)-1):
            soma += round(item[idx+1],2) - round(item[idx],2)
        
        return soma + penalidade_cliente + penalidade_grupo