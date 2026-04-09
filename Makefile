.PHONY: help up down logs clean pull-model test-single test-concurrent test-stress test-unit health metrics grafana

help:
	@echo "  make up               Start all services"
	@echo "  make down             Stop all services"
	@echo "  make logs             Tail all logs"
	@echo "  make pull-model       Pull Llama 3 8B into Ollama"
	@echo "  make test-single      Phase 2 gate: single request"
	@echo "  make test-concurrent  Phase 3 gate: 4 concurrent"
	@echo "  make test-stress      Phase 4/5 gate: 60s stress"
	@echo "  make test-unit        Run pytest unit tests"
	@echo "  make health           Check all service health"
	@echo "  make metrics          Print Prometheus metrics"
	@echo "  make grafana          Open Grafana dashboard"
	@echo "  make clean            Remove containers + volumes"

up:
	docker compose up -d
	@sleep 5 && docker compose ps
	@echo "Gateway:    http://localhost:8000"
	@echo "Ray:        http://localhost:8265"
	@echo "Grafana:    http://localhost:3000  (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

down:
	docker compose down

logs:
	docker compose logs -f --tail=50

logs-gateway:
	docker compose logs -f gateway

logs-ray:
	docker compose logs -f ray-head ray-worker-1 ray-worker-2

pull-model:
	docker compose exec ollama ollama pull llama3:8b-instruct-q4_K_M

test-single:
	python tests/load_test.py --mode single

test-concurrent:
	python tests/load_test.py --mode concurrent --users 4

test-stress:
	python tests/load_test.py --mode stress --users 4 --duration 60

test-unit:
	pytest tests/ -v --timeout=30 -x

test-openai:
	python3 -c "\
from openai import OpenAI; \
c = OpenAI(base_url='http://localhost:8000/v1', api_key='local'); \
r = c.chat.completions.create(model='llama3', messages=[{'role':'user','content':'Say hi.'}]); \
print('Response:', r.choices[0].message.content); print('PASS')"

health:
	curl -s http://localhost:8000/health | python3 -m json.tool

metrics:
	curl -s http://localhost:8000/metrics | grep "^llm_"

grafana:
	xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null || echo "Open: http://localhost:3000"

clean:
	docker compose down -v --remove-orphans
	docker builder prune -f
