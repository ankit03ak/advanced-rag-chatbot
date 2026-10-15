# Run the chatbot with Docker

This image runs the requested Streamlit frontend and LangGraph backend in one
non-root container. PostgreSQL is external (for example, Supabase); FAISS
indexes and uploaded PDFs are stored in the named `rag_stores` volume.

## Configure secrets

Copy `.env.example` to `.env` and set:

```env
GOOGLE_API_KEY=...
DATABASE_URL=postgresql://...?...sslmode=require
ALPHA_VANTAGE_API_KEY=...
```

Do not commit `.env`. The database URL and API keys are passed at runtime and
are not copied into the image.

## Start

From this directory:

```powershell
docker compose up --build
```

Open <http://localhost:8501>.

The application initializes its PostgreSQL authentication and LangGraph
checkpoint tables on startup. It does not migrate or delete the existing local
SQLite database automatically.

## Stop and restart

```powershell
docker compose down
docker compose up
```

`docker compose down` keeps the named volume by default. To verify the same
FAISS data is available after restart, upload a PDF, stop the container, start
it again, log in, and reopen the original conversation.

Do not use `docker compose down -v` unless you intentionally want to delete
the persisted FAISS indexes and uploaded PDFs.

## Hosted storage limitation

The named volume is suitable for one Docker host. For multi-host deployment,
move `/app/data/rag_stores` to durable shared object storage or a managed file
volume before scaling beyond one container.
