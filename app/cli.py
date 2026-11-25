import sys
from pathlib import Path

from . import config
from .answer import answer_question
from .embeddings import get_embedder
from .ingest import IngestError, parse_document
from .store import Store


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"ingest", "ask"}:
        print(__doc__)
        return 2
    store = Store(config.DB_PATH, get_embedder())
    if argv[0] == "ingest":
        for arg in argv[1:]:
            p = Path(arg)
            for f in sorted(p.iterdir()) if p.is_dir() else [p]:
                try:
                    r = store.upsert(parse_document(f.name, f.read_bytes()))
                    print(f"{r.status:9} {r.doc_id} ({r.chunks} chunks, v{r.version})")
                except IngestError as e:
                    print(f"skipped   {f.name}: {e}")
        return 0
    print(answer_question(store, " ".join(argv[1:])).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
