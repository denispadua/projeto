import pandas as pd

class LeitorCSV:
  """
  Classe para leitura de arquivos CSV com Pandas.

  Atributos:
    caminho_arquivo: O caminho para o arquivo CSV.
    df: Um DataFrame Pandas contendo os dados do CSV.

  Métodos:
    __init__(self, caminho_arquivo): Inicializa a classe com o caminho do arquivo.
    ler_csv(self): Lê o arquivo CSV e armazena os dados em um DataFrame.
    mostrar_informacoes(self): Exibe informações sobre o DataFrame.
    selecionar_colunas(self, colunas): Seleciona as colunas especificadas e retorna um novo DataFrame.
    filtrar_dados(self, condicao): Filtra os dados do DataFrame de acordo com a condição especificada e retorna um novo DataFrame.
  """

  def __init__(self, caminho_arquivo):
    """
    Construtor da classe.

    Argumentos:
      caminho_arquivo: O caminho para o arquivo CSV.
    """
    self.caminho_arquivo = caminho_arquivo
    self.df = None

  def ler_csv(self):
    """
    Lê o arquivo CSV e armazena os dados em um DataFrame.
    """
    try:
      self.df = pd.read_csv(self.caminho_arquivo)
      print(f"Leitura do arquivo {self.caminho_arquivo} realizada com sucesso!")
    except FileNotFoundError:
      print(f"Erro: Arquivo {self.caminho_arquivo} não encontrado!")
    except Exception as e:
      print(f"Erro durante leitura do CSV: {e}")

  def mostrar_informacoes(self):
    """
    Exibe informações sobre o DataFrame.

    Inclui:
      - Nome das colunas
      - Número de linhas e colunas
      - Tipo de dados de cada coluna
      - Resumo das estatísticas descritivas dos dados
    """
    if self.df is not None:
      print("---------- Informações do DataFrame ----------")
      print(f"Nome das colunas: {self.df.columns}")
      print(f"Número de linhas: {self.df.shape[0]}")
      print(f"Número de colunas: {self.df.shape[1]}")
      print(f"Tipos de dados das colunas:")
      print(self.df.dtypes)
      print("---------- Descrição das Estatísticas ----------")
      print(self.df.describe())
    else:
      print("Erro: O DataFrame ainda não foi carregado. Leia o CSV primeiro usando o método ler_csv()")

  def selecionar_colunas(self, colunas) -> pd.DataFrame:
    """
    Seleciona as colunas especificadas e retorna um novo DataFrame.

    Argumentos:
      colunas: Uma lista com os nomes das colunas a serem selecionadas.

    Retorno:
      Um novo DataFrame contendo apenas as colunas selecionadas.
    """
    if self.df is not None:
      try:
        df_selecionado = self.df[colunas]
        print(f"Colunas {colunas} selecionadas com sucesso!")
        return df_selecionado.copy()
      except KeyError as e:
        print(f"Erro: Colunas {e} não encontradas no DataFrame.")
    else:
      print("Erro: O DataFrame ainda não foi carregado. Leia o CSV primeiro usando o método ler_csv()")

  def filtrar_dados(self, condicao):
    """
    Filtra os dados do DataFrame de acordo com a condição especificada e retorna um novo DataFrame.

    Argumentos:
      condicao: Uma string contendo a condição de filtragem no formato Pandas (por exemplo, 'Idade > 18').

    Retorno:
      Um novo DataFrame contendo apenas os dados que atendem à condição.
    """
    if self.df is not None:
      try:
        df_filtrado = self.df.query(condicao)
        print(f"Dados filtrados com sucesso pela condição: {condicao}")
        return df_filtrado
      except Exception as e:
        print(f"Erro durante filtragem: {e}")
    else:
      print("Erro: O DataFrame ainda não foi carregado. Leia o CSV primeiro usando o método ler_csv()")