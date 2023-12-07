# Projeto Rossmann

Previsão de vendas para uma empresa de medicamentos da europa

## 1.	Problema de negócio
Rossmann opera mais de 3.000 drogarias em 7 países europeus. Atualmente, os gerentes das lojas Rossmann têm a tarefa de prever suas vendas diárias com até seis semanas de antecedência. As vendas nas lojas são influenciadas por muitos fatores, incluindo promoções, concorrência, feriados escolares e estaduais, sazonalidade e localidade. Com milhares de gestores individuais prevendo vendas com base em circunstâncias específicas, a precisão dos resultados pode ser bastante variada.

## 2.	Objetivo
Construir um modelo robusto para prever 6 semanas de vendas diárias para 1.115 lojas localizadas em toda a Europa.

## 3.	Premissas do negócio
Foi considerada a variável “open” como 1, ou seja, as lojas deverão estar abertas e a variável “sales” deve ser maior que 0, neste caso, as lojas deverão ter no mínimo 1 venda.

## 4.	Estratégia adotada para solução

Passo 1 - Descrição dos dados: O objetivo deste passo é ter um entendimento inicial de como os dados estão relacionados com o problema de negócio proposto, para tal, lançando mão de algumas métricas estatísticas de posição e distribuição.

Passo 2 - Engenharia de atributos: Neste passo foram criadas novas variáveis a partir das variáveis originais a fim de melhorar a qualidade dos dados facilitando seu o entendimento.

Passo 3 - Filtragem das variáveis: No processo de filtragem, busca-se selecionar e reter apenas as variáveis relevantes para a análise ou modelagem, com base nos objetivos do negócio e nas características do problema em questão.
 
Passo 4 - Análise exploratória: A análise exploratória dos dados (EDA) tem como principal objetivo proporcionar uma compreensão mais aprofundada dos dados, tais como: Compreender a distribuição das variáveis numéricas e categóricas, identificar outliers (valores atípicos), se existe desbalanceamento da variável resposta, correlações entre as variáveis entre outras análises.

Passo 5 - Preparação dos dados: Neste tópico, os dados serão transformados, ou seja, as variáveis categóricas serão convertidas em números, pois os algoritmos de machine learning não tem uma boa performance com dados não numéricos. Os dados também precisarão ser reescalados, ou seja, colocar as variáveis em uma escala comum.
 
Passo 6 - Seleção das variáveis: Neste passo o principal objetivo é selecionar as variáveis mais relevantes e descartar as menos importantes, para que estas sejam submetidas aos algoritmos de machine learning. Esta seleção é feita através de algoritmos específicos de seleção, com isso, busca-se reduzir a dimensionalidade dos dados sem perder em performance ao resultado final.

Passo 7 - Algoritmos de machine learning: Serão implementados alguns modelos de algoritmos de machine learning para possamos escolher o que melhor performou com os dados disponíveis.
 
Passo 8 - Ajuste dos hiperparâmetros: Com a definição do melhor modelo de machine learning, busca-se otimizar ainda mais a performance do modelo com alguns ajustes finos em seus parâmetros.

Passo 9 - Performance do negócio: Nesta etapa, o termo "performance do negócio" refere-se ao impacto que os modelos e análises têm nos resultados e metas de uma organização, pois busca garantir que as soluções propostas realmente tragam benefícios tangíveis para a empresa.

Passo 10 - Modelo em produção: Nesta etapa final, o melhor modelo treinado é implantado e usado em um ambiente operacional do mundo real. O objetivo principal deste passo de colocar um modelo em produção é transformar o trabalho de desenvolvimento de modelos em um sistema que pode fornecer valor contínuo e automático.

## 5.	Top 3 insights
Hipótese 1: Lojas com maior sortimento (variedades) deveriam vender mais?
Falsa: Lojas com maior sortimento (variedades) NÃO vendem mais.
Hipótese 2: Lojas deveriam vender mais ao longo dos anos?
Falsa: Lojas NÃO vendem mais ao longo dos anos.
Hipótese 3: Lojas abertas durante o feriado de natal deveriam vender mais?
Falsa: Lojas abertas durante o feriado de natal vendem menos em relação a outros tipos de feriados.

## 6.	Aplicação do modelo de machine learning
Foram feitos testes com vários algoritmos de machine learning, utilizando a técnica de cross-validation e ajuste fino dos parâmetros.

## 7.	Performance do modelo de machine learning
O algoritmo de machine learning escolhido foi o XGBRegressor.
 
![Comparativo](img/comparativo_algoritmos.JPG)

## 8.	Resultado do negócio
Conforme mostrado na figura abaixo, foram adicionados mais 2 cenários além do predito para as próximas 6 semanas em todas as lojas Rossmann, ou seja, um cenário pessimista e outro otimista, com isso o gestou conseguirá tomar a melhor decisão frente aos valores encontrados.
 
 ![Cenarios](img/cenarios.JPG)

## 9.	Conclusão
O resultado prático deste projeto de previsão de vendas para uma empresa de medicamentos da Europa, reflete-se na capacidade de adicionar cenários otimistas e pessimistas além das previsões tradicionais. Essa abordagem oferece aos gestores uma visão mais abrangente, permitindo a tomada de decisões mais informadas diante de diferentes contextos.
Dentre os principais insights, destacam-se a falsificação de hipóteses relacionadas à variedade de produtos, tendência de vendas ao longo dos anos e desempenho durante o feriado de Natal. Esses insights desafiam preconcepções e contribuem para uma compreensão mais precisa dos padrões de vendas.
Em última análise, o projeto não apenas fornece um modelo de previsão confiável, mas também agrega valor ao negócio ao oferecer insights valiosos e cenários alternativos. A implementação prática do modelo permite que a Rossmann tome decisões estratégicas mais fundamentadas em um ambiente de negócios dinâmico e complexo.

## 10.	Próximos passos
•	Fazer o balanceamento dos dados;
•	Testar outros algoritmos de machine learning;
•	Buscar novas variáveis para definir melhor o comportamento dos clientes.
