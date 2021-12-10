import sys
from datetime import datetime, timedelta
from flask.helpers import url_for
sys.path.append('..')
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
from bson import json_util
app = Flask(__name__)
CORS(app, support_credentials=True)
from flask_pymongo import PyMongo
app.config['MONGO_URI'] = "mongodb://localhost:27017/DB"
mongo = PyMongo(app)

@app.route('/', methods=['GET'])
@cross_origin(supports_credentials=True)
def routeread():
    try:
        result =  mongo.db.ObjDet.find()
        return Response(json_util.dumps(result) , mimetype='text/json')
    except Exception as e:
        print(e)
        return 'error'
@app.route('/file/<id>')
@cross_origin(supports_credentials=True)
def file(id):
   
    return mongo.send_file(id)
    # return f'''
    # <img src="{mongo.send_file(id)}">
    # '''
@app.route('/compare')
@cross_origin(supports_credentials=True)
def profile():
    details = mongo.db.ObjDet.find_one();
    return {"img":url_for('file',id=details['image']),"imgogm":url_for('file',id=details['ODAC'])}

@app.route('/alldata')
@cross_origin(supports_credentials=True)
def alldata():
    return mongo.db.ObjDet.find();
 
@app.route('/towat')
@cross_origin(supports_credentials=True)
def towat():
    details = mongo.db.ObjDet.find_one();
    con=f'''
    <img src="{url_for('file',id=details['image'])}">
    <img src="{url_for('file',id=details['ODAC'])}">
    '''

    for i in details["subImagedata"]:
        con+=f'''<img src="{url_for('file',id=i['data'])}">''';
    return con
@app.route('/dateData',methods=['POST'])
@cross_origin(supports_credentials=True)
def dateData():
    data = request.get_json()
    fromt=datetime.strptime(data['from'],"%Y-%m-%dT%H:%M:%S.000Z")
    
    tot = datetime.strptime(data['to'],"%Y-%m-%dT%H:%M:%S.000Z")
    details = mongo.db.ObjDet.find({"timestamp":{"$gt":fromt,"$lt":tot}})
    a = []
    jindex=0
    for i in details:
        a.append(i)
    
    return Response(json_util.dumps(a) , mimetype='text/json')
# #   |host="127.0.0.3",
app.run(port=8010)

# OLD one 
# import sys

# from flask.helpers import url_for
# sys.path.append('..')
# from flask import Flask, Response
# from flask_cors import CORS, cross_origin
# from bson import json_util
# app = Flask(__name__)
# CORS(app, support_credentials=True)
# from flask_pymongo import PyMongo
# app.config['MONGO_URI'] = "mongodb://localhost:27017/DB"
# mongo = PyMongo(app)

# @app.route('/', methods=['GET'])
# @cross_origin(supports_credentials=True)
# def routeread():
#     try:
#         result =  mongo.db.ObjDet.find_one()
#         return Response(json_util.dumps(result) , mimetype='text/json')
#     except Exception as e:
#         print(e)
#         return 'error'
# @app.route('/file/<id>')
# @cross_origin(supports_credentials=True)
# def file(id):
#     return mongo.send_file(id)
# @app.route('/compare')
# @cross_origin(supports_credentials=True)
# def profile():
#     details = mongo.db.ObjDet.find_one();
#     return f'''
#     <img src="{url_for('file',id=details['image'])}">
#     <img src="{url_for('file',id=details['ODAC'])}">
#     '''
# #   |host="127.0.0.3",

# app.run(port=8010)


