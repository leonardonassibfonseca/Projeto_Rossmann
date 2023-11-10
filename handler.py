import os
import pickle
import pandas as pd
from flask import Flask, request, Response

#Importando classe Rossmann do arquivo Rossmann.py da pasta rossmann
#from nome_pasta ou diretório.nome_arquivo import nome da classe dentro do arquivo
from rossmann.Rossmann import Rossmann 

#Carregando o modelo
modelo = pickle.load(open('modelo/Rossmann.pkl', 'rb'))

#Instaciando objeto da classe Flask que será a API
app = Flask(__name__)

#Método POST envia alguma coisa
#Método GET pede alguma coisa
@app.route('/rossmann/predict', methods = ['POST'])
def rossmann_predict():
    teste_json = request.get_json()
    
    if teste_json:
        if isinstance(teste_json, dict):
            teste_raw = pd.DataFrame(teste_json, index = [0])
        else:
            teste_raw = pd.DataFrame(teste_json, columns = teste_json[0].keys())
        #Instanciando a classe Rossmann
        pipeline = Rossmann()

        #limpeza_dos_dados
        df1 = pipeline.limpeza_dos_dados(teste_raw)
        
        #engenharia_de_atributos
        df2 = pipeline.engenharia_de_atributos(df1)
        
        #preparacao_dos_dados
        df3 = pipeline.preparacao_dos_dados(df2)
        
        #predição
        df_resposta = pipeline.get_prediction(modelo, teste_raw, df3)
        return df_resposta
    else:
        return Response('{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port) #Dizer para endpoint rodar no localhost (rodando na máquina)
#172.25.114.131 -> endereço IPv4 pc local
#app.run('0.0.0.0')