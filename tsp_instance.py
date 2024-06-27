###############################################################################
# tsp_instance.py: data structures and support function to deal with instances
# of the Traveling Salesman Problem.
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

from pandas import DataFrame
from brkga_mp_ipr.exceptions import LoadError
import pandas as pd

class TSPInstance():
    def __init__(self, filename: str):
        """
        Initializes the instance loading from a file.
        """
        try:
            self.df = pd.read_csv(filename)
            print(f"Leitura do arquivo {filename} realizada com sucesso!")
        except FileNotFoundError:
            print(f"Erro: Arquivo {filename} não encontrado!")
        except Exception as e:
            print(f"Erro durante leitura do CSV: {e}")

    ###########################################################################
    
    def tratamento_dados(self):
        dict_compr_renda = {
            "5%-": 1,
            "5% a 10%": 2,
            "10% a 15%": 3,
            "15% a 20%": 4,
            "20% a 25%": 5,
            "25% a 30%": 6,
            "30%+": 7
        }
        self.df['Compr_Renda_pre_processada'] = self.df['Compr_Renda'].copy()
        self.df['Compr_Renda'] = self.df['Compr_Renda'].map(dict_compr_renda)

        dict_nivel_escolaridade = {
            "Med_e_Sup_Inc": 1,
            "Sup_e_Pos": 2
        }
        self.df['Nivel_Escolaridade_pre_processada'] = self.df['Nivel_Escolaridade'].copy()
        self.df['Nivel_Escolaridade'] = self.df['Nivel_Escolaridade'].map(dict_nivel_escolaridade)

        dict_estado_civil = {
            "Casado": 1,
            "Divorciado": 2,
            "Solteiro": 3,
            "Viuvo": 4
        }
        self.df['Estado_Civil_pre_processada'] = self.df['Estado_Civil'].copy()
        self.df['Estado_Civil'] = self.df['Estado_Civil'].map(dict_estado_civil)

        dict_regiao = {
            "Centro-Oeste": 1,
            "Nordeste": 2,
            "Norte": 3,
            "Sudeste": 4,
            "Sul": 5
        }
        self.df['Regiao_pre_processada'] = self.df['Regiao'].copy()
        self.df['Regiao'] = self.df['Regiao'].map(dict_regiao)

        self.df['Nivel_Risco_pre_processada'] = self.df['Nivel_Risco_Novo'].copy()
        self.df['Nivel_Risco_Novo'] = self.df['Nivel_Risco_Novo'].map(lambda v: v+1)
