#!/bin/bash

source ~/.pyenv/versions/3.9.19/envs/.env_bs11_deploy/bin/activate && uvicorn main:app --reload
