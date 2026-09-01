# data/raw/

La aplicación descarga los precios automáticamente (acción desde Yahoo
Finance, índice desde Stooq). Esta carpeta es el respaldo manual para cuando
la fuente automática del índice falla — que es lo que pasa con Stooq en la
práctica (ver README principal, sección "Fuente de datos": Cloudflare
bloquea las descargas automatizadas de forma persistente, no solo
ocasional).

**Detección automática**: si existe un archivo `<ticker>_manual.csv` acá
(por ejemplo `ipsa_manual.csv` para el ticker `^IPSA`), `pipeline.run_beta_case`
lo usa directamente, sin llamar a Stooq ni a Yahoo para el índice — no hace
falta pasar nada a mano ni tocar código. Ver
`beta_analyzer.pipeline._default_manual_csv_path`.

**Ya incluido**: `ipsa_manual.csv` — histórico diario del S&P/CLX IPSA,
2021-01-04 a 2025-12-30 (1.243 observaciones), columnas
`Date,Open,High,Low,Close,Volume`. Se obtuvo de Investing.com (histórico
público del índice) porque el endpoint de descarga de Stooq estaba
bloqueando las solicitudes automatizadas de forma persistente. `Open` queda
igual a `Close` de la misma fila (Investing.com no expone un valor de
apertura intradía distinto para este índice) y `Volume` en 0 — ninguno de
los dos se usa en el cálculo de Beta, que solo consume `Close` (ver
`returns.py`: para índices siempre se usa el campo `close`, sin importar el
`price_field` elegido para la acción). `High`/`Low` sí son el rango
intradía real reportado por el proveedor.

Para actualizar este archivo con fechas más recientes, o para armar el
manual CSV de otro índice, cualquier fuente que entregue una serie diaria
del S&P/CLX IPSA (u otro índice) sirve, siempre que el CSV final tenga esas
columnas y el nombre `<ticker_sin_circunflejo_en_minúscula>_manual.csv`.

También puede usarse como respaldo manual de la **acción** (no solo del
índice): descargue el CSV histórico del proveedor que prefiera y cárguelo
con `beta_analyzer.data_sources.load_manual_csv`, o páselo directamente a
`pipeline.run_beta_case(..., manual_stock_csv=...)` — para la acción no hay
detección automática por nombre de archivo, hay que indicarlo explícitamente.

Los CSV que se dejen acá no se versionan en git (ver `.gitignore`), salvo
que se edite esa regla a propósito — es la razón por la que `ipsa_manual.csv`
se entrega como archivo aparte en vez de quedar solo documentado.
