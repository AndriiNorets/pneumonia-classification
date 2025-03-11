install:
	poetry install --no-root

download-data:
	poetry run python data/download.py

test:
	poetry run pytest ./tests -vv

update:
	poetry update

format:
	poetry run ruff format .
	poetry run ruff check . --fix

check:
	poetry run ruff format --check .

