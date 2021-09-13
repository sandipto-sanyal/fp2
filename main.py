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

@app.route('/train',methods=['POST'])
def train():
    '''
    Trains the ML algo

    Returns
    -------
    None.

    '''
    try:
        training_file_path = request.args['training_file_path']
        epochs = int(request.args['epochs'])
        tr = Train(df_path=training_file_path, epochs=epochs)
        tr.train()
        tr.export_binaries()
        tr.evaluate()
        tr.upload_binaries()
        tr.deploy_model()
        return {'response':'training complete. Visit {} to monitor deployment.'.format(c.website_path),
                'model_version': tr.version,
                'r2_score': tr.r2_score
                }, 200
        # return {'response':'Success'}, 200
    except:
        exception_stack_trace = str(traceback.format_exc())
        return {'response':exception_stack_trace}, 500
    


# @app.errorhandler(500)
# def server_error(e):
#     logging.exception('An error occurred during a request.')
#     return """
#     An internal error occurred: <pre>{}</pre>
#     See logs for full stacktrace.
#     """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=False)
# [END gae_flex_quickstart]
