web: uvicorn --host=0.0.0.0 --port=${PORT:-5000} --ws-max-size 100000000 main:app --env-file ./.env.prod
# --ws-max-size 100000000 file size limit is set to 100Mb
# Old => web: fastapi run --host=0.0.0.0 --port=${PORT:-5000}
