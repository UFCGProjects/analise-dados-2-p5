#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier


def handlerData(filepath):

    # Carrega o CSV e seta algumas colunas como object (factor).
    df = pd.read_csv(filepath,
                     sep = ';',
                     dtype = {'CURSO': np.object,
                              'CODIGO': np.object,
                              'PERIODO': np.object,
                              'MATRICULA': np.object,
                              'PERIODO_INGRESSO': np.object},
                     header=0)

    # Se alguma valor de MEDIA for nulo, seta como 0.
    df.loc[df.MEDIA.isnull(), 'MEDIA'] = 0

    # Crio um novo dataframe com as matriculas unicas, também copio as CURSO, PERIODO e se existir COD_EVASAO.
    if 'COD_EVASAO' in df:
        adf = df.drop_duplicates('MATRICULA')[['MATRICULA', 'COD_EVASAO', 'CURSO', 'PERIODO']]
    else:
        adf = df.drop_duplicates('MATRICULA')[['MATRICULA', 'CURSO', 'PERIODO']]

    # Adiciono a coluna TOTAL_CADEIRAS com a quantidade de cadeiras que o aluno se matriculou.
    adf['TOTAL_CADEIRAS'] = adf['MATRICULA'].map(lambda x: len(df[df.MATRICULA == x]))

    # Adiciono a coluna REPROVADO_NOTA com a quantidade de cadeiras que o aluno foi reprovado pela nota.
    adf['REPROVADO_NOTA'] = adf['MATRICULA'].map(lambda x: len(df[(df.MATRICULA == x) & (df.SITUACAO == 'Reprovado')]))
    # Adiciono a coluna REPROVADO_NOTA_P com a proporção de cadeiras reprovadas por nota.
    adf['REPROVADO_NOTA_P'] = adf.REPROVADO_NOTA / adf.TOTAL_CADEIRAS

    # Adiciono a coluna REPROVADO_FALTA com a quantidade de cadeiras que o aluno foi reprovado por falta.
    adf['REPROVADO_FALTA'] = adf['MATRICULA'].map(lambda x: len(df[(df.MATRICULA == x) & (df.SITUACAO == 'Reprovado por Falta')]))
    # Adiciono a coluna REPROVADO_FALTA_P com a proporção de cadeiras reprovadas por falta.
    adf['REPROVADO_FALTA_P'] = adf.REPROVADO_FALTA / adf.TOTAL_CADEIRAS

    # Adiciono a coluna TRANCADO com a quantidade de cadeiras que o aluno trancou.
    adf['TRANCADO'] = adf['MATRICULA'].map(lambda x: len(df[(df.MATRICULA == x) & (df.SITUACAO == 'Trancado')]))
    # Adiciono a coluna TRANCADO_P com a proporção de cadeiras que o aluno trancou.
    adf['TRANCADO_P'] = adf.TRANCADO / adf.TOTAL_CADEIRAS

    # Adiciono a coluna MEDIA com média do aluno.
    adf['MEDIA'] = adf['MATRICULA'].map(lambda x: np.mean(df[df.MATRICULA == x][['MEDIA']]).values[0])

    # Retorno o array de matriculas e o dataframe sem a coluna matricula
    return adf['MATRICULA'].values, adf.drop(['MATRICULA'], axis=1)

def predict(train_ids, train_data, test_ids, test_data, filename):
    print 'Training...'
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

    print 'Predicting...'
    output = forest.predict(test_data).astype(int)

    predictions_file = open(filename, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["MATRICULA","COD_EVASAO"])
    open_file_object.writerows(zip(test_ids, output))
    predictions_file.close()
    print 'Done.'

# Metodo utilitario para analisar o resultado da predição.
def analyse():
    pdf = pd.read_csv('./dados/matheus_sampaio_CC_EM.csv',
                     dtype = {'MATRICULA': np.object,
                              'COD_EVASAO': np.object},
                     header=0)

    tdf = pd.read_csv('./dados/dadosAlunos-teste-com-evasao.csv',
                     sep = ';',
                     dtype = {'CURSO': np.object,
                              'CODIGO': np.object,
                              'COD_EVASAO': np.object,
                              'PERIODO': np.object,
                              'MATRICULA': np.object,
                              'PERIODO_INGRESSO': np.object},
                     header=0)

    pdf['REAL_EVASAO'] = pdf.MATRICULA.map(lambda x: tdf[tdf.MATRICULA == x].COD_EVASAO.iloc[0])

    pdf['DIFF'] = (pdf.COD_EVASAO != pdf.REAL_EVASAO).astype(object)

    print pdf[pdf.DIFF == 1]

# Prepara os dados de treino
train_mats, train_df = handlerData('./dados/dadostreino.csv')

# PREDICT
teste_mats, teste_df = handlerData('./dados/dadosteste_CC_EM.csv')
predict(train_mats, train_df.values, teste_mats, teste_df.values, "matheus_sampaio_CC_EM.csv")

teste_mats, teste_df = handlerData('./dados/dadosteste_CC.csv')
predict(train_mats, train_df.values, teste_mats, teste_df.values, "matheus_sampaio_CC.csv")

teste_mats, teste_df = handlerData('./dados/dadosteste_EM.csv')
predict(train_mats, train_df.values, teste_mats, teste_df.values, "matheus_sampaio_EM.csv")
# analyse()
