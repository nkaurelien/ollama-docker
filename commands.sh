

# RUN OLLAMA VERSION
docker compose -f docker-compose-gpu.yml run ollama  --version


# INSTALL OLLAMA MODEL
docker compose -f docker-compose-gpu.yml run ollama - pull nomic