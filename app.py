from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import tensorflow
import sqlite3
import torch
import pandas as pd
import numpy as np
import faiss
import re
import scipy
import pickle



from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model/')
index = faiss.read_index('faiss/faiss_index_all.index')
sql_data = 'db/serapeum.db'
conn = sqlite3.connect(sql_data, check_same_thread=False)
cur = conn.cursor()


def get_title(ID):
    cur.execute("SELECT ID_Doc ,  Title from paper where ID_Doc =" + str(ID))
    title = cur.fetchone()[0]
    return title


def vector_search(query, model, indexx, num_results=10):
    vector = model.encode(list(query))
    D, I = indexx.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    return [list(df[df.id == idx][column]) for idx in I[0]]


def get_text(ID):
    cur.execute('SELECT   body_text  FROM paper WHERE ID_Doc=' + str(ID))
    try:
        text = str(cur.fetchone())
        text = text.strip()
        text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r"\d", "", text)
        text = text.strip()
    except:
        return jsonify({'msg': 'No data found!'}), 401
    return text

if __name__ == '__main__':
    app.run()
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})





@app.route('/vecsrch1', methods=['GET'])
def get_vecsrch1():
    question = request.args.get('quest', None)
    D, I = vector_search([question], model, index, num_results=10)
    df1 = pd.DataFrame({'Distance': D.flatten().tolist(), 'ID_Doc': I.flatten().tolist()})
    xx = ','.join([str(x) for x in I.flatten().tolist()])
    sql1 = "SELECT ID_Doc ,  Title from paper where ID_Doc in (" + xx + ")"
    df = pd.read_sql_query(sql1, conn)
    dfinal = df.merge(df1, how='inner', left_on='ID_Doc', right_on='ID_Doc')
    return Response(response=(dfinal.sort_values(by=['Distance']).to_json(orient="records")), status=200,
                    mimetype="application/json")


@app.route('/vecsrcha', methods=['GET'])
def get_vecsrcha():
    question = request.args.get('quest', None)
    D, I = vector_search([question], model, index, num_results=1)
    df1 = pd.DataFrame({'Distance': D.flatten().tolist(), 'ID_Doc': I.flatten().tolist()})
    xx = ','.join([str(x) for x in I.flatten().tolist()])
    sql1 = "SELECT ID_Doc ,  Title, abstract from paper where ID_Doc in (" + xx + ")"
    df = pd.read_sql_query(sql1, conn)
    dfinal = df.merge(df1, how='inner', left_on='ID_Doc', right_on='ID_Doc')
    return Response(response=(dfinal.sort_values(by=['Distance']).to_json(orient="records")), status=200,
                    mimetype="application/json")





@app.route('/vecsrch2', methods=['GET'])
def vecsrch2():
    question1 = request.args.get('quest', None)
    df1 = pd.read_sql_query('SELECT ID, ID_Doc,  sentence  FROM sections WHERE part !="TITLE" ', conn)
    sentences = df1['sentence'].values.tolist()
    with open('emb/sections_all.pkl', 'rb') as file:
        sentence_embeddings = pickle.load(file)
    query_embeddings = model.encode([question1])
    number_top_matches = 10
    for query, query_embedding in zip([question1], query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        zz = []
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        for idx, distance in results[0:number_top_matches]:
            z = '.'.join(sentences[idx - 2:idx + 2])
            ida = df1.iloc[idx]['ID']
            zz.append({'Context': z, 'Answer': sentences[idx].strip(), 'Distance': (1 - distance), 'ID': ida})
    res = pd.DataFrame.from_dict(zz)
    cc = ','.join([str(x) for x in res['ID'].to_list()])
    df2 = pd.read_sql_query(
        "Select sections.ID,sections.ID_Doc,paper.Title  from sections INNER JOIN paper  on sections.ID_Doc = paper.ID_Doc  where  ID in " + " (" + cc + ")",
        conn)
    res = res.merge(df2, how='inner', left_on='ID', right_on='ID')
    return Response(response=(res.to_json(orient="records")), status=200, mimetype="application/json")





@app.route('/vecsrch3', methods=['GET'])
def vecsrch3():
    question1 = request.args.get('quest', None)
    df1 = pd.read_sql_query('SELECT ID, ID_Doc,  sentence  FROM transition  ', conn)
    sentences = df1['sentence'].values.tolist()
    with open('emb/transition.pkl', 'rb') as file:
        sentence_embeddings = pickle.load(file)
    query_embeddings = model.encode([question1])
    number_top_matches = 10
    for query, query_embedding in zip([question1], query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        zz = []
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        for idx, distance in results[0:number_top_matches]:
            z = '.'.join(sentences[idx - 2:idx + 2])
            ida = df1.iloc[idx]['ID']
            zz.append({'Context': z, 'Answer': sentences[idx].strip(), 'Distance': (1 - distance), 'ID': ida})
    res = pd.DataFrame.from_dict(zz)
    cc = ','.join([str(x) for x in res['ID'].to_list()])
    df2 = pd.read_sql_query(
        "Select sections.ID,sections.ID_Doc,paper.Title  from sections INNER JOIN paper  on sections.ID_Doc = paper.ID_Doc  where  ID in " + " (" + cc + ")",
        conn)
    res = res.merge(df2, how='inner', left_on='ID', right_on='ID')
    return Response(response=(res.to_json(orient="records")), status=200, mimetype="application/json")







@app.route('/questId', methods=['GET'])
def questId():
    ID = request.args.get('id', None)
    question1 = request.args.get('quest', None)
    df1 = pd.read_sql_query ('SELECT ID, ID_Doc,  sentence  FROM sections WHERE part !="TITLE" and  ID_Doc='+ID,conn)
    sentences = df1['sentence'].values.tolist()
    sentence_embeddings = model.encode(sentences)
    query_embeddings = model.encode([question1])
    number_top_matches = 10
    for query, query_embedding in zip([question1], query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]
        zz = []
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        for idx, distance in results[0:number_top_matches]:
            z = '.'.join(sentences[idx - 2:idx + 2])
            zz.append({'Context': z, 'Answer': sentences[idx].strip(), 'Distance': (1 - distance)})
    res = pd.DataFrame.from_dict(zz)
    res=res.drop_duplicates(subset=['Answer'])
    return Response(response=(res.to_json(orient="records")), status=200, mimetype="application/json")





