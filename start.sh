#!/bin/bash

~/.pyenv/versions/3.9.19/envs/.env_bs11_onnx/bin/python -m uvicorn --ws-max-size 100000000 main:app --env-file ./.env.dev
