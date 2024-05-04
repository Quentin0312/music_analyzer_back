#!/bin/bash

source ~/.pyenv/versions/3.9.19/envs/.env_bs11/bin/activate && uvicorn main:app --reload
