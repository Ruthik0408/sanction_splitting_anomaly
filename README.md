# sanction_splitting_anomaly

## Run duplicate checker UI

Run frontend and backend together:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting
./run_app.sh
```

Open:

```text
http://localhost:5000/ui
```

Manual frontend build:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting/frontend
npm install
npm run build
```

Start the Python API:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting
source venv/bin/activate
uvicorn scripts.api_server:app --host 0.0.0.0 --port 5000
```

Open:

```text
http://localhost:5000/ui
```

If port 5000 is already in use, stop the old server with `Ctrl+C` and run the command again.

For frontend development with hot reload:

```bash
cd /home/ruthikreddy/Desktop/sanction_spliting/frontend
npm run dev
```

Open:

```text
http://localhost:5173/ui
```
