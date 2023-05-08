# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:14:24 2023

@author: Jeff
"""

from __future__ import print_function
from config import *

import openai
import tiktoken
import pinecone
import uuid
import sys
import logging

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask import request

from handle_file import handle_file
from answer_question import get_answer_from_files


openai.api_key = "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"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_pinecone_index() -> str:
    """
    Load index name from Pinecone, raise error if the index can't be found.
    """
    pinecone.init(
        pinecone_api_key="b8f5f7b4-d5fb-4ae9-a220-d1f4e371b709",
        environment="us-west4-gcp",
    )
    index_name = "ee104"
    if not index_name in pinecone.list_indexes():
        print(pinecone.list_indexes())
        raise KeyError(f"Index '{index_name}' does not exist.")

    return index_name

def create_app():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = "ee104"
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")
    pinecone_index = pinecone.Index(index_name)
    tokenizer = tiktoken.get_encoding("gpt2")
    session_id = str(uuid.uuid4().hex)
    app = Flask(__name__)
    app.pinecone_index = pinecone_index
    app.tokenizer = tokenizer
    app.session_id = session_id
    # log session id
    logging.info(f"session_id: {session_id}")
    app.config["file_text_dict"] = {}
    CORS(app, supports_credentials=True)

    return app



app = create_app()

@app.route(f"/process_file", methods=["POST"])
@cross_origin(supports_credentials=True)
def process_file():
    try:
        file = request.files['file']
        logging.info(str(file))
        handle_file(file, app.session_id, app.pinecone_index, app.tokenizer)

        return jsonify({"success": True})
    except Exception as e:
        logging.error(str(e))
        return jsonify({"success": False})

@app.route(f"/answer_question", methods=["POST"])
@cross_origin(supports_credentials=True)
def answer_question():
    try:
        params = request.get_json()
        question = params["question"]

        answer_question_response = get_answer_from_files(question, app.session_id, app.pinecone_index)

        return answer_question_response
    except Exception as e:
        return str(e)

@app.route("/healthcheck", methods=["GET"])
@cross_origin(supports_credentials=True)
def healthcheck():
    return "OK"
if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT, threaded=True)

def handle_file_string(file_str: str, session_id: str, tokenizer: EncodedTokenizer):
    """
    Encode text in file_str and add it to the Pinecone index
    """
    # Tokenize text
    encoding = tokenizer.encode(file_str, return_tensors="pt")

    # Get embeddings
    with torch.no_grad():
        model_output = model(encoding, output_hidden_states=True)
        embeddings = model_output.hidden_states[-2].squeeze(0)

    # Convert embeddings to numpy array
    embeddings_np = embeddings.numpy()

    # Upsert embeddings to Pinecone index
    app.pinecone_index.upsert(items=[(f"{session_id}_{uuid.uuid4().hex}", embeddings_np[i]) for i in range(embeddings_np.shape[0])])
