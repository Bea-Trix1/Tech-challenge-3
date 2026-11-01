"""Tráfego de demonstração: predições válidas e erros de validação esperados."""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.requests < 1 or args.interval < 0:
        parser.error("--requests deve ser positivo e --interval não pode ser negativo")
    counts: Counter[int] = Counter()
    for i in range(args.requests):
        # 10% de payloads inválidos para demonstrar a taxa de erro (HTTP 422).
        text = "" if i % 10 == 9 else "Metastatic carcinoma treated with chemotherapy"
        req = urllib.request.Request(
            f"{args.url.rstrip('/')}/predict",
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
        except urllib.error.URLError as exc:
            parser.exit(1, f"Não foi possível acessar a API: {exc.reason}\n")
        counts[status] += 1
        if status != (422 if not text else 200):
            parser.exit(1, f"Resposta inesperada da API: HTTP {status}\n")
        if i < args.requests - 1:
            time.sleep(args.interval)
    print(f"Chamadas concluídas por status HTTP: {dict(counts)}")


if __name__ == "__main__":
    main()
