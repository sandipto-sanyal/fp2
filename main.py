# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging

from flask import Flask, request
from train import Train
import constants as c
import traceback

app = Flask(__name__)



try:
  import googleclouddebugger
  googleclouddebugger.enable(
    breakpoint_enable_canary=True
  )
except ImportError:
  pass

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Welcome to Food Recommender'

@app.route('/prediction',methods=['POST'])
def prediction():
    '''
    Predicts on the ML algo

    Returns
    -------
    None.

    '''
    try:
        return {'response':'Success'}, 200
    except:
        exception_stack_trace = str(traceback.format_exc())
        return {'response':exception_stack_trace}, 500
    


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=False)
# [END gae_flex_quickstart]
