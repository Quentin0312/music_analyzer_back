#!/bin/bash

~/.pyenv/versions/3.9.19/envs/.env_music_analyzer_back_onnx/bin/python -m uvicorn --ws-max-size 100000000 main:app --reload --env-file ./.env.dev
