#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:09:37 2019

@author: eucassio
"""


from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import numpy as np
import pickle
from datetime import datetime

def date_diff_in_Seconds(dt2, dt1):
  timedelta = dt2 - dt1
  return timedelta.microseconds/1000


new_model = keras.models.load_model('documentos_mpl_model.h5')
selector = pickle.load(open("selector.pickle", "rb"))
vectorizer = pickle.load(open("vectorizer.pickle", "rb"))

new_model.summary()


labels = ['stic', 'judicial','admin']

val_texts = ["""
             
Concurso de cartórios: TJ-PI divulga resultado provisório na avaliação de títulos

O presidente do Tribunal de Justiça do Estado do Piauí (TJ-PI), desembargador Sebastião Ribeiro Martins, assinou, nesta quarta-feira (29), o Edital nº 39/2019, que divulga resultado provisório na avaliação de títulos dos candidatos do concurso para outorga de delegações de serventias extrajudiciais de notas e de registro do Estado do Piauí.

Esta é a sexta etapa do certame, em andamento desde julho de 2013. O Edital traz “resultado provisório na avaliação de títulos, na seguinte ordem: modalidade de outorga, número de inscrição, nome do candidato em ordem alfabética e nota provisória na avaliação de títulos”.

De acordo com o cronograma estabelecido pelo Centro de Seleção e de Promoção de Eventos da Universidade de Brasília (Cespe/UnB) e pela Comissão Organizadora do Concurso (COC), os candidatos têm entre os dias 3 e 4 de junho para apresentarem recurso administrativo acerca do resultado provisório divulgado nesta quarta-feira.

Na última sexta-feira (24) o TJ-PI divulgou a Relação de Vacâncias das serventias notariais e/ou de registro vagas no Estado do Piauí com vistas ao provimento de tais vagas por concurso público. A lista foi elaborada pela Vice-Corregedoria Geral da Justiça do Estado do Piauí, com a inclusão dos cartórios vagos até 24 de maio de 2019. Ao todo, há 239 serventias extrajudiciais nessa situação no Piauí.

Confira o edital .

 """,
 """
 Em momento histórico, TJ-PI implanta Plenário Virtual

Em momento histórico, o Tribunal de Justiça do Estado do Piauí (TJ-PI) implantou, nesta sexta-feira (7), o seu Plenário Virtual. Ao todo, 69 processos foram pautados para a primeira sessão de julgamento eletrônico de recursos e processos originários de segundo grau no âmbito da Justiça estadual piauiense. A implementação do Plenário Virtual colabora com a racionalização e a celeridade dos julgamentos sob responsabilidade do Pleno e das Câmaras Cíveis e Criminais, auxiliando o aumento da produtividade do Tribunal.

“Hoje é um dia histórico para o TJ-PI. O Plenário Virtual é mais uma ferramenta que utilizamos para nos transformar de fato em um Judiciário digital, mais célere e acessível. Não falamos mais de futuro, e sim de presente”, comentou o presidente do TJ-PI, desembargador Sebastião Ribeiro Martins.

A instalação do Plenário Virtual no âmbito do TJ-PI foi aprovada por meio da Resolução nº 133/2019 e regulamentada pelo Provimento nº 13/2019, da Presidência. Segundo a Resolução 133, os agravos internos e os embargos de declaração serão obrigatoriamente submetidos ao julgamento em ambiente eletrônico, por exemplo. Já o Provimento nº 13/2019 especifica que “os processos de competência originária e os recursos interpostos no segundo grau de jurisdição, distribuídos no Sistema de Processo Judicial Eletrônico – PJe, poderão ser julgados por meio eletrônico, utilizando a ferramenta do Plenário Virtual”.

Pelo normativo, após a inserção do relatório no Sistema PJe, o relator deve indicar que o julgamento do processo se dará em ambiente virtual, observando-se os processos com envio obrigatório (agravo interno e embargos de declaração) e os que serão encaminhados a critério do relator. Para que o processo seja incluído em sessão em ambiente virtual, o relatório e o voto precisam estar necessariamente inseridos no Sistema PJe até a data da abertura da sessão virtual.

As sessões em ambiente virtual são semanais, com início às 10h das sextas-feiras, e têm duração de sete dias corridos, encerrando-se o prazo para votação dos demais desembargadores integrantes da Câmara na sexta-feira subsequente, às 09h. Os integrantes do órgão julgador têm acesso ao relatório e ao voto inseridos pelo relator.

O desembargador Olímpio Galvão, um dos idealizadores da implantação do Plenário Virtual no TJ-PI, declara: “Foram três meses de organização e implantação do sistema, bem como para realização do treinamento dos gabinetes dos desembargadores, secretários de sessão e assistentes de procuradores, para utilização da nova ferramenta. Hoje, o Tribunal de Justiça do Piauí dá um passo importante para a melhoria da prestação jurisdicional.”

Acompanhamento
De caráter público, as sessões podem ser acompanhadas pela internet, em endereço eletrônico disponível no sítio do TJ-PI. Porém, os votos somente serão tornados públicos depois de concluído seu julgamento. Os processos não concluídos estarão automaticamente incluídos na pauta da sessão seguinte nos termos do Art. 935 (CPC).
As sessões podem ser acompanhadas pela página http://www.tjpi.jus.br/sessoes-virtuais/.

Virtual
Atualmente cerca de 75% dos tribunais de Justiça do Brasil já implantaram o Plenário Virtual no 2° grau. O Supremo Tribunal Federal (STF) e o Conselho Nacional de Justiça (CNJ), por exemplo, já utilizam essa ferramenta.
 """
 ]
t1 = datetime.now()

x_val = vectorizer.transform(val_texts)
x_val = selector.transform(x_val)
x_val = x_val.astype('float32')

result = new_model.predict(x_val)

t2 = datetime.now()

print("\n%d Milisegundos" %(date_diff_in_Seconds(t2, t1)))
print(labels[np.argmax(result[0])])
print(labels[np.argmax(result[1])])


print(result.shape)
print (result[0])
print (result[1])

