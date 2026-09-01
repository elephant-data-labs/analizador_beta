"""Catálogo de empresas chilenas para el combo box de la interfaz.

Son las componentes actuales del índice S&P/CLX IPSA según la propia página
de Yahoo Finance (`/quote/%5EIPSA/components/`), que a la fecha de esta
verificación (31-08-2026) lista 27 — el índice no tiene un número fijo de 30,
varía con los rebalanceos periódicos. Cada ticker se comprobó además de
forma individual en Yahoo Finance: existe, cotiza en CLP en la Bolsa de
Santiago y tiene historial diario continuo desde enero de 2021 (mismo
procedimiento que con Aguas Andinas — ver README, sección "Fuente de
datos"). Nota: `LTM.SN` (LATAM Airlines) muestra una caída fuerte de precio
entre 2021 y ahora consistente con la reestructuración societaria de la
empresa en ese período (Chapter 11, 2020-2022) — es un dato real, no un
error de la fuente.

No agregar una empresa sin repetir esa verificación — es el mismo principio
de "no asumir, verificar antes de programar" del encargo original. Si se
necesita una empresa que no está acá, la interfaz permite buscarla en Yahoo
Finance por nombre o nemotécnico (`data_sources.search_tickers`), o
ingresar el ticker directamente si ya se conoce.
"""

from __future__ import annotations

from dataclasses import dataclass

OTHER_LABEL = "Otra empresa (buscar en Yahoo Finance)"


@dataclass(frozen=True)
class CompanyEntry:
    name: str
    ticker: str
    sector: str


# Verificadas en Yahoo Finance el 31-08-2026: ticker válido, CLP, Santiago,
# historial diario desde ene-2021. Orden: el mismo de la página de
# componentes del IPSA (de mayor a menor capitalización de mercado).
KNOWN_COMPANIES: list[CompanyEntry] = [
    CompanyEntry("Sociedad Química y Minera de Chile (SQM-B)", "SQM-B.SN", "Minería / litio"),
    CompanyEntry("Banco de Chile", "CHILE.SN", "Banca"),
    CompanyEntry("Falabella S.A.", "FALABELLA.SN", "Retail"),
    CompanyEntry("Banco Santander-Chile", "BSANTANDER.SN", "Banca"),
    CompanyEntry("Banco de Crédito e Inversiones (BCI)", "BCI.SN", "Banca"),
    CompanyEntry("LATAM Airlines Group S.A.", "LTM.SN", "Aerolíneas"),
    CompanyEntry("Enel Américas S.A.", "ENELAM.SN", "Energía eléctrica"),
    CompanyEntry("Plaza S.A. (Mallplaza)", "MALLPLAZA.SN", "Retail — centros comerciales"),
    CompanyEntry("Empresas Copec S.A.", "COPEC.SN", "Energía / forestal"),
    CompanyEntry("Cencosud S.A.", "CENCOSUD.SN", "Retail"),
    CompanyEntry("Enel Chile S.A.", "ENELCHILE.SN", "Energía eléctrica"),
    CompanyEntry("Embotelladora Andina S.A.", "ANDINA-B.SN", "Bebidas"),
    CompanyEntry("Parque Arauco S.A.", "PARAUCO.SN", "Retail — centros comerciales"),
    CompanyEntry("Empresas CMPC S.A.", "CMPC.SN", "Forestal / celulosa"),
    CompanyEntry("Colbún S.A.", "COLBUN.SN", "Energía eléctrica"),
    CompanyEntry("Inversiones La Construcción S.A. (ILC)", "ILC.SN", "Holding — seguros/salud"),
    CompanyEntry("Compañía Sud Americana de Vapores S.A.", "VAPORES.SN", "Naviera"),
    CompanyEntry("Compañía Cervecerías Unidas S.A. (CCU)", "CCU.SN", "Bebidas"),
    CompanyEntry("Engie Energía Chile S.A.", "ECL.SN", "Energía eléctrica"),
    CompanyEntry("Aguas Andinas S.A.", "AGUAS-A.SN", "Utilities — agua"),
    CompanyEntry("Empresa Nacional de Telecomunicaciones S.A. (Entel)", "ENTEL.SN", "Telecomunicaciones"),
    CompanyEntry("Inversiones Aguas Metropolitanas S.A. (IAM)", "IAM.SN", "Utilities — agua"),
    CompanyEntry("CAP S.A.", "CAP.SN", "Minería / acero"),
    CompanyEntry("Ripley Corp S.A.", "RIPLEY.SN", "Retail"),
    CompanyEntry("SalfaCorp S.A.", "SALFACORP.SN", "Construcción / inmobiliaria"),
    CompanyEntry("Viña Concha y Toro S.A.", "CONCHATORO.SN", "Bebidas — vinos"),
    CompanyEntry("Sonda S.A.", "SONDA.SN", "Tecnología / servicios TI"),
]

_BY_NAME = {c.name: c for c in KNOWN_COMPANIES}


def company_labels() -> list[str]:
    """Opciones para el selectbox: catálogo verificado + búsqueda manual."""
    return [c.name for c in KNOWN_COMPANIES] + [OTHER_LABEL]


def find_by_name(name: str) -> CompanyEntry | None:
    return _BY_NAME.get(name)


# --- Catálogo de índices de mercado -----------------------------------------
#
# Dos presets verificados, para que la interfaz no obligue a escribir a mano
# nombre + ticker + fuente cada vez (ver README, sección "Fuente de datos").
# `source` decide cómo los descarga `pipeline._fetch_market_prices`:
#   "stooq" — Yahoo Finance no tiene serie histórica utilizable para ^IPSA
#             (solo un dato suelto), así que se usa Stooq, con caída a un
#             CSV manual en data/raw/ (ya incluido para IPSA) y, si tampoco
#             hay CSV, a un índice replicado con las componentes del IPSA.
#   "yahoo" — ^GSPC (S&P 500) sí tiene historial diario completo y estable
#             en Yahoo Finance, la misma fuente que ya se usa para la
#             acción — no hace falta pasar por Stooq para este caso.
OTHER_INDEX_LABEL = "Otro índice (ingresar manualmente)"


@dataclass(frozen=True)
class IndexEntry:
    name: str
    ticker: str
    source: str  # "stooq" | "yahoo" — ver config.IndexSource
    note: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.ticker})"


KNOWN_INDICES: list[IndexEntry] = [
    IndexEntry(
        name="S&P/CLX IPSA",
        ticker="^IPSA",
        source="stooq",
        note="Mercado accionario chileno (Bolsa de Santiago). Vía Stooq / CSV manual — ver README.",
    ),
    IndexEntry(
        name="S&P 500",
        ticker="^GSPC",
        source="yahoo",
        note="Mercado accionario de EE.UU. Historial completo directo en Yahoo Finance.",
    ),
]

_INDEX_BY_LABEL = {e.label: e for e in KNOWN_INDICES}


def index_labels() -> list[str]:
    """Opciones para el selectbox de índice: catálogo verificado + manual."""
    return [e.label for e in KNOWN_INDICES] + [OTHER_INDEX_LABEL]


def find_index_by_label(label: str) -> IndexEntry | None:
    return _INDEX_BY_LABEL.get(label)
