install:
	poetry install --no-root

test:
	poetry run pytest ./tests -vv

update:
	poetry update

format:
	poetry run ruff format .

check:
	poetry run ruff format --check .

train:
	poetry run python -m train