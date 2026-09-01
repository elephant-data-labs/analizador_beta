"""Descarga de precios históricos, verificada manualmente antes de programar.

Verificación hecha en Yahoo Finance (navegador) y Stooq antes de escribir este
módulo — ver la sección "Fuente de datos" del README para el detalle caso por
caso. Resumen de lo que este módulo asume porque ya se comprobó:

- Acción (`fetch_stock_prices`): Yahoo Finance vía `yfinance`. Para
  AGUAS-A.SN hay historial diario continuo desde 2021, en CLP, con "Adj
  Close" ya corregido por dividendos (se confirmó al menos un dividendo
  dentro de la ventana 2021-2025 donde Close y Adj Close difieren). No se
  detectaron splits.
- Índice (`fetch_index_prices`): Stooq. El ticker "^IPSA" en Yahoo Finance no
  tiene serie histórica utilizable (solo entrega un punto de dato suelto), así
  que se usa Stooq como fuente consistente para el índice — su serie diaria
  de ^IPSA cubre 2021-2025 sin huecos. Stooq no entrega un "Adj Close"
  separado para índices: se usa el cierre directamente.

Ambas funciones devuelven un DataFrame con índice de fechas (`DatetimeIndex`,
sin huso horario) y columnas `open, high, low, close, adj_close, volume`, para
que el resto del pipeline (retornos, regresión) no necesite saber de dónde
vino cada serie.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# Stooq exige un User-Agent de navegador real: sin esto responde
# "Access denied" incluso para descargas legítimas del CSV histórico.
_STOOQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


class DataSourceError(RuntimeError):
    """La fuente de datos no entregó lo que el pipeline necesita."""


def _empty_range_error(ticker: str, start: date, end: date, source: str) -> DataSourceError:
    return DataSourceError(
        f"{source} no devolvió datos para '{ticker}' entre {start} y {end}. "
        "Verifique el ticker y la ventana de fechas antes de reintentar."
    )


def _exclusive_end_iso(end: date) -> str:
    """yfinance trata `end` como exclusivo: se pide un día extra para que la
    fecha de término elegida por el usuario quede incluida.

    `end` es siempre un `datetime.date` plano (así lo entrega tanto
    `CaseParameters` como `st.date_input` de Streamlit) — se suma un
    `timedelta` normal, no un `pd.Timedelta`, porque `date + pd.Timedelta`
    devuelve otro `date` (sin método `.date()`), no un `Timestamp`.
    """
    return (end + timedelta(days=1)).isoformat()


def fetch_stock_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Precios diarios de una acción desde Yahoo Finance (`yfinance`).

    `end` se interpreta inclusivo (yfinance trata `end` como exclusivo, así
    que internamente se pide un día extra).
    """
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=_exclusive_end_iso(end),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise _empty_range_error(ticker, start, end, "Yahoo Finance")

    # yfinance >=0.2 puede devolver columnas MultiIndex (Precio, Ticker)
    # incluso para un solo ticker; se aplanan antes de normalizar nombres.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )[PRICE_COLUMNS].copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    return frame.sort_index()


def fetch_index_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Precios diarios de un índice desde Stooq.

    Stooq no entrega "Adj Close" para índices, así que `adj_close` queda
    igual a `close` (un índice de precio no paga dividendos por sí mismo).
    """
    symbol = ticker.lstrip("^").lower()
    url = (
        f"https://stooq.com/q/d/l/?s=^{symbol}"
        f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d"
    )
    response = requests.get(url, headers=_STOOQ_HEADERS, timeout=30)
    response.raise_for_status()
    text = response.text.strip()

    if not text or text.lower().startswith("access denied") or "Date,Open" not in text:
        reason = (
            "una página de verificación anti-bot (Cloudflare) en vez del CSV — un "
            "`requests.get` normal no puede resolverla, aunque los headers estén bien"
            if "<html" in text.lower() or "noscript" in text.lower()
            else f"una respuesta inesperada: {text[:120]!r}"
        )
        raise DataSourceError(
            f"Stooq devolvió {reason} para el índice '{ticker}'. "
            "Puede ser un bloqueo temporal — reintente en unos minutos. Si persiste, "
            "el pipeline puede usar en su lugar un índice replicado a partir de los "
            "componentes del IPSA en Yahoo Finance (ver `build_replicated_index_prices` "
            "y el README), o puede descargar el CSV manualmente desde stooq.com y "
            f"guardarlo en data/raw/{ticker.lstrip('^').lower()}_manual.csv "
            "con columnas Date,Open,High,Low,Close,Volume."
        )

    frame = pd.read_csv(io.StringIO(text))
    if frame.empty:
        raise _empty_range_error(ticker, start, end, "Stooq")

    frame = frame.rename(columns={c: c.lower() for c in frame.columns})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    frame["adj_close"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = pd.NA
    return frame[PRICE_COLUMNS]


def build_replicated_index_prices(
    constituent_tickers: list[str], start: date, end: date, base_level: float = 100.0
) -> pd.DataFrame:
    """Índice de reemplazo para cuando Stooq no está disponible: una cartera
    equiponderada de `constituent_tickers`, construida enteramente con datos
    de Yahoo Finance (la misma fuente que la acción).

    Esto NO es el S&P/CLX IPSA oficial — es una aproximación. El retorno
    diario del índice replicado es el promedio simple de los retornos
    diarios de los componentes que tengan dato ese día (no se descarta el
    día completo si a un componente le falta un dato puntual). El nivel del
    índice se arma acumulando esos retornos desde `base_level` (por defecto
    100), ya que lo único que importa para el Beta es el retorno, no el
    nivel absoluto.

    Quien llama es responsable de excluir de `constituent_tickers` la acción
    que se está analizando (para no comparar la acción contra un mercado que
    la incluye a ella misma) — `pipeline.run_beta_case` ya lo hace.
    """
    import yfinance as yf

    if not constituent_tickers:
        raise DataSourceError("No hay componentes para construir el índice replicado.")

    raw = yf.download(
        constituent_tickers,
        start=start.isoformat(),
        end=_exclusive_end_iso(end),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise DataSourceError(
            "No se pudo construir el índice replicado: Yahoo Finance no devolvió "
            "precios para los componentes del IPSA."
        )

    adj_close = raw["Adj Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Adj Close"]]
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    constituent_returns = adj_close.pct_change()
    # Equiponderado: promedio simple entre los componentes disponibles ese
    # día. fillna(0.0) solo afecta el primer día (sin retorno previo), que
    # queda en el nivel base sin mover el índice.
    average_return = constituent_returns.mean(axis=1, skipna=True).fillna(0.0)
    level = base_level * (1.0 + average_return).cumprod()

    frame = pd.DataFrame({"close": level})
    frame["open"] = frame["high"] = frame["low"] = frame["adj_close"] = frame["close"]
    frame["volume"] = pd.NA
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    return frame[PRICE_COLUMNS].sort_index()


@dataclass(frozen=True)
class TickerMatch:
    """Un resultado de `search_tickers`: un ticker candidato en Yahoo Finance."""

    symbol: str
    name: str
    exchange: str
    quote_type: str

    @property
    def label(self) -> str:
        return f"{self.symbol} — {self.name} ({self.exchange})"

    @property
    def is_santiago(self) -> bool:
        return "Santiago" in self.exchange


def search_tickers(query: str, max_results: int = 8) -> list[TickerMatch]:
    """Busca un ticker en Yahoo Finance a partir de un nemotécnico o nombre
    de empresa — para cuando la empresa que se necesita no está en
    `catalog.KNOWN_COMPANIES`.

    Usa `yfinance.Search`, que reutiliza la misma sesión/cookies que
    `fetch_stock_prices` (evita reimplementar a mano el manejo de
    autenticación de la API de Yahoo). Verificado manualmente contra el
    endpoint de búsqueda de Yahoo antes de programar: buscar por nemotécnico
    ("ENTEL", "AGUAS-A") o por nombre ("banco de chile") ambos funcionan y
    devuelven el ticker de Bolsa de Santiago junto con eventuales homónimos
    de otras bolsas.

    Devuelve una lista vacía si Yahoo no encontró nada para la búsqueda (no
    es un error: el llamador decide cómo mostrarlo). Si la búsqueda en sí
    falla (Yahoo no responde, problema de red), lanza `DataSourceError`.
    """
    query = query.strip()
    if not query:
        return []

    from yfinance import Search

    try:
        quotes = Search(query, max_results=max_results, news_count=0, lists_count=0).quotes
    except Exception as exc:  # yfinance puede lanzar varias excepciones propias
        raise DataSourceError(
            f"No se pudo buscar '{query}' en Yahoo Finance ({exc}). "
            "Intente de nuevo en unos segundos, o ingrese el ticker "
            "directamente si ya lo conoce."
        ) from exc

    matches = [
        TickerMatch(
            symbol=q["symbol"],
            name=q.get("shortname") or q.get("longname") or q["symbol"],
            exchange=q.get("exchDisp") or q.get("exchange") or "?",
            quote_type=q.get("quoteType", "?"),
        )
        for q in quotes
        if q.get("quoteType") == "EQUITY" and q.get("symbol")
    ]
    # No se descartan los resultados de otras bolsas (puede ser justo lo que
    # se busca), pero se prioriza Bolsa de Santiago por ser el caso de uso
    # principal de este proyecto.
    matches.sort(key=lambda m: 0 if m.is_santiago else 1)
    return matches


def load_manual_csv(path: Path) -> pd.DataFrame:
    """Respaldo manual: carga un CSV con columnas Date,Open,High,Low,Close[,Volume]
    (formato estándar de exportación de Stooq/Yahoo) cuando la descarga
    automática falla. No inventa `adj_close`: si no viene en el archivo, se
    iguala a `close`.
    """
    frame = pd.read_csv(path)
    frame = frame.rename(columns={c: c.lower() for c in frame.columns})
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = pd.NA
    return frame[PRICE_COLUMNS]
