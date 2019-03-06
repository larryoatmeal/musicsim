#!/usr/bin/env python

import pytest
import math
import re

'''
Simple and functional REST server for Python (2.7) using no dependencies beyond the Python standard library.
Features:
* Map URI patterns using regular expressions
* Map any/all the HTTP VERBS (GET, PUT, DELETE, POST)
* All responses and payloads are converted to/from JSON for you
* Easily serve static files: a URI can be mapped to a file, in which case just GET is supported
* You decide the media type (text/html, application/json, etc.)
* Correct HTTP response codes and basic error messages
* Simple REST client included! use the rest_call_json() method
As an example, let's support a simple key/value store. To test from the command line using curl:
curl "http://localhost:8080/records"
curl -X PUT -d '{"name": "Tal"}' "http://localhost:8080/record/1"
curl -X PUT -d '{"name": "Shiri"}' "http://localhost:8080/record/2"
curl "http://localhost:8080/records"
curl -X DELETE "http://localhost:8080/record/2"
curl "http://localhost:8080/records"
Create the file web/index.html if you'd like to test serving static files. It will be served from the root URI.
@author: Tal Liron (tliron @ github.com)
'''

import sys, os, re, shutil, json, urllib, urllib2, BaseHTTPServer

# Fix issues with decoding HTTP responses
reload(sys)
sys.setdefaultencoding('utf8')

here = os.path.dirname(os.path.realpath(__file__))



# sim3d = pytest.sim3d(32, 32, 32)
# sim3d.init()

 
sim3d = [None]

pattern = re.compile('([0-9]+)/([0-9]+)')

def get_args(handler):
  content_len = int(handler.headers.getheader('content-length', 0))
  body = handler.rfile.read(content_len)
  s = urllib.unquote(handler.path)
  return map(int, s.split('/')[2:-1])

def get_args_post(handler):
  content_len = int(handler.headers.getheader('content-length', 0))
  body = handler.rfile.read(content_len)
  # s = urllib.unquote(handler.path)
  print body
  return map(int, body.split('/')[2:-1])

# def parse(s):
#     m = pattern.search(s)
#     x = int(m.group(1))
#     y = int(m.group(2))
#     return (x, y)

# def write_sim(handler):
#     x,y = parse(urllib.unquote(handler.path))
#     sim.setWall(x,y)
#     return "DONE"

# def delete_sim(handler):
#     x,y = parse(urllib.unquote(handler.path))
#     sim.clearWall(x,y)
#     return "DONE"

def init(handler):
  args = get_args(handler)
  if len(args) != 3:
    return None

  if sim3d[0] != None:
    sim3d[0].clean()
  sim = pytest.sim3d(args[0], args[1], args[2])
  sim.init()

  sim3d[0] = sim  
  return "SUCCESS"
def clean(handler):
  sim3d[0].clean()

def reset(handler):
  sim3d[0].reset()
def readBackAudio(handler):
  return sim3d[0].readBackAudio()

def readBackData(handler):
  data = sim3d[0].readBackData()
  for i in range(len(data)):
    for j in range(4):
      if math.isnan(data[i][j]):
        data[i][j] = 1000000000
  return data
      

def readBackAux(handler):
  return sim3d[0].readBackAux()

def step(handler):
  args = get_args(handler)
  if len(args) == 1:
    sim3d[0].step(args[0])
  else:
    sim3d[0].step(1)
  return "SUCCESS"

def setSigma(handler):
  return get_args(handler)

def setWall(handler):
  args = get_args(handler)
  if len(args) != 4:
    return None
  sim3d[0].setWall(args[0], args[1], args[2], args[3])
  return "SUCCESS"

def scheduleWalls(handler):
  args = get_args_post(handler)
  if len(args) % 4 != 0:
    print "BAD ARGS"
    return None
  for i in range(0, len(args), 4):
    sim3d[0].scheduleWall(args[i + 0], args[i + 1], args[i + 2], args[i + 3])
  return "SUCCESS"

def writeWalls(handler):
  sim3d[0].writeWalls()
  return "SUCCESS"

def setListener(handler):
  args = get_args(handler)
  if len(args) != 3:
    return None
  sim3d[0].setListener(args[0], args[1], args[2])
  return "SUCCESS"

def setExcitor(handler):
  args = get_args(handler)
  if len(args) != 4:
      return None
  sim3d[0].setExcitor(args[0], args[1], args[2], args[3])
  return "SUCCESS"
    
def setPBore(handler):
  args = get_args(handler)
  if len(args) != 3:
    return None
  sim3d[0].setPBore(args[0], args[1], args[2])
  return "SUCCESS"
    
def setPressureMouth(handler):
  args = get_args(handler)
  if len(args) != 1:
     return None
  sim3d[0].setPressureMouth(args[0])
  return "SUCCESS"

def rest_call_json(url, payload=None, with_payload_method='PUT'):
    'REST call with JSON decoding of the response and JSON payloads'
    if payload:
        if not isinstance(payload, basestring):
            payload = json.dumps(payload)
        # PUT or POST
        response = urllib2.urlopen(MethodRequest(url, payload, {'Content-Type': 'application/json'}, method=with_payload_method))
    else:
        # GET
        response = urllib2.urlopen(url)
    response = response.read().decode()
    return json.loads(response)

class MethodRequest(urllib2.Request):
    'See: https://gist.github.com/logic/2715756'
    def __init__(self, *args, **kwargs):
        if 'method' in kwargs:
            self._method = kwargs['method']
            del kwargs['method']
        else:
            self._method = None
        return urllib2.Request.__init__(self, *args, **kwargs)

    def get_method(self, *args, **kwargs):
        return self._method if self._method is not None else urllib2.Request.get_method(self, *args, **kwargs)




class RESTRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.routes = {
            #r'^/$': {'file': 'web/index.html', 'media_type': 'text/html'},
            #r'^/records$': {'GET': get_records, 'media_type': 'application/json'},
            #r'^/record/': {'GET': get_record, 'PUT': set_record, 'DELETE': delete_record, 'media_type': 'application/json'}},
          r'^/init/': {'GET': init, 'media_type': 'application/json'},
          r'^/clean/': {'GET': clean, 'media_type': 'application/json'},
          r'^/reset/': {'GET': reset, 'media_type': 'application/json'},
          r'^/setSigma/': {'GET': setSigma, 'media_type': 'application/json'},
          r'^/step/': {'GET': step, 'media_type': 'application/json'},
          r'^/setWall/': {'GET': setWall, 'media_type': 'application/json'},
          r'^/readBackAux/': {'GET': readBackAux, 'media_type': 'application/json'},
          r'^/readBackAudio/': {'GET': readBackAudio, 'media_type': 'application/json'},
          r'^/readBackData/': {'GET': readBackData, 'media_type': 'application/json'},
          r'^/scheduleWalls/': {'POST': scheduleWalls, 'media_type': 'application/json'},
          r'^/writeWalls/': {'GET': writeWalls, 'media_type': 'application/json'},
          r'^/setListener/': {'GET': setListener, 'media_type': 'application/json'},     
          r'^/setExcitor/': {'GET': setExcitor, 'media_type': 'application/json'},            
          r'^/setPBore/': {'GET': setPBore, 'media_type': 'application/json'},
          r'^/setPressureMouth/': {'GET': setPressureMouth, 'media_type': 'application/json'},
        }

        return BaseHTTPServer.BaseHTTPRequestHandler.__init__(self, *args, **kwargs)
    def end_headers (self):
        self.send_header('Access-Control-Allow-Origin', '*')
        BaseHTTPServer.BaseHTTPRequestHandler.end_headers(self)

    def do_HEAD(self):
        self.handle_method('HEAD')

    def do_GET(self):
        self.handle_method('GET')

    def do_POST(self):
        self.handle_method('POST')

    def do_PUT(self):
        self.handle_method('PUT')

    def do_DELETE(self):
        self.handle_method('DELETE')

    def get_payload(self):
        payload_len = int(self.headers.getheader('content-length', 0))
        payload = self.rfile.read(payload_len)
        payload = json.loads(payload)
        return payload

    def handle_method(self, method):
        route = self.get_route()
        if route is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write('Route not found\n')
        else:
            if method == 'HEAD':
                self.send_response(200)
                if 'media_type' in route:
                    self.send_header('Content-type', route['media_type'])
                self.end_headers()
            else:
                if 'file' in route:
                    if method == 'GET':
                        try:
                            f = open(os.path.join(here, route['file']))
                            try:
                                self.send_response(200)
                                if 'media_type' in route:
                                    self.send_header('Content-type', route['media_type'])
                                self.end_headers()
                                shutil.copyfileobj(f, self.wfile)
                            finally:
                                f.close()
                        except:
                            self.send_response(404)
                            self.end_headers()
                            self.wfile.write('File not found\n')
                    else:
                        self.send_response(405)
                        self.end_headers()
                        self.wfile.write('Only GET is supported\n')
                else:
                    if method in route:
                        content = route[method](self)
                        if content is not None:
                            self.send_response(200)
                            if 'media_type' in route:
                                self.send_header('Content-type', route['media_type'])
                            self.end_headers()
                            if method != 'DELETE':
                                self.wfile.write(json.dumps(content))
                        else:
                            self.send_response(404)
                            self.end_headers()
                            self.wfile.write('Not found\n')
                    else:
                        self.send_response(405)
                        self.end_headers()
                        self.wfile.write(method + ' is not supported\n')


    def get_route(self):
        for path, route in self.routes.iteritems():
            if re.match(path, self.path):
                return route
        return None

def rest_server(port):
    'Starts the REST server'
    http_server = BaseHTTPServer.HTTPServer(('', port), RESTRequestHandler)
    print 'Starting HTTP server at port %d' % port
    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        pass
    print 'Stopping HTTP server'
    http_server.server_close()

def main(argv):
    rest_server(8080)

if __name__ == '__main__':
    main(sys.argv[1:])
