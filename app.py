from bottle import Bottle, route, run
from paintnet.inference import *

app = Bottle()

@app.route('/hello')
def hello():
    return "Hello World!"




run(app, host='localhost', port=8080, debug=True)