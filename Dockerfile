# SDR virtual da estação — substituto do grs-iq-rx.
#
# Sem gcc e sem librtlsdr: não há hardware para falar. numpy e pyzmq vêm de
# wheel manylinux no CPython 3.11, então a imagem é slim de verdade.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

EXPOSE 5556

# Sem argumentos, transmite o FS-2 sintético em sintonia fixa. O cenário real
# vem do `command:` do compose.
CMD ["python", "-m", "sdr_sim.main"]
