docker stop llama.cpp
docker rm llama.cpp
docker run --name llama.cpp \
--restart unless-stopped \
-d \
-v /home/vv/projects/takb-ml-poc/models_cache:/models \
-p 8080:8080 ghcr.io/ggerganov/llama.cpp:server \
-m /models/llama3.1_8b_instruct.gguf \
--port 8080 \
--host 0.0.0.0 \
-n 512 \
-ub 131072
