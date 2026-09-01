from __future__ import annotations

import base64
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from beta_analyzer.catalog import (
    KNOWN_COMPANIES,
    KNOWN_INDICES,
    OTHER_INDEX_LABEL,
    OTHER_LABEL,
    company_labels,
    find_by_name,
    find_index_by_label,
    index_labels,
)
from beta_analyzer.config import DEFAULT_CASE, FREQUENCY_LABELS, CaseParameters
from beta_analyzer.data_sources import DataSourceError, search_tickers
from beta_analyzer.outliers import DEFAULT_Z_THRESHOLD, detect_outliers
from beta_analyzer.pipeline import BetaCaseResult, run_beta_case
from beta_analyzer.plotting import price_with_outliers, scatter_with_regression
from beta_analyzer.regression import (
    beta_interpretation,
    ci_interpretation,
    p_value_interpretation,
    r_squared_interpretation,
    significance_conclusion,
)
from beta_analyzer.returns import resample_price_field

st.set_page_config(page_title="Analizador de Beta", page_icon="📈", layout="wide")


def _apply_search_result(name: str, ticker: str) -> None:
    """Callback de "Usar este resultado": corre antes del próximo rerun, así
    que puede escribir el session_state de los widgets manual_company_name /
    manual_company_ticker sin chocar con la restricción de Streamlit de no
    modificar el estado de un widget ya instanciado en el rerun actual."""
    st.session_state["manual_company_name"] = name
    st.session_state["manual_company_ticker"] = ticker


def table_height(frame: pd.DataFrame) -> int:
    """Alto necesario para mostrar la tabla completa, sin scroll interno.

    Misma utilidad que en el proyecto EEFF hermano: Streamlit muestra unas 10
    filas por defecto y corta el resto; varias tablas de este tablero superan
    ese largo y quedarían truncadas sin avisar.
    """
    return (len(frame) + 1) * 35 + 3


_logo_path = ROOT / "Elephant.png"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else None

# Mismas clases y estilos que el proyecto EEFF hermano (Elephant Data Labs):
# encabezado compacto + portada de Inicio, para que ambas apps se vean como
# un mismo laboratorio y no como dos proyectos sueltos.
st.markdown(
    """
    <style>
    /* Encabezado compacto: acompaña a todas las hojas sin robarles espacio.
       Los datos del autor viven en la portada (hoja Inicio), no acá. */
    .analizador-header {
        background: #ffffff;
        border: 1px solid #d7e6ee;
        border-left: 6px solid #157a8a;
        border-radius: 12px;
        padding: 0.8rem 1.4rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        box-shadow: 0 1px 6px rgba(11, 61, 98, 0.05);
    }
    .analizador-header img {
        height: 62px;
        width: auto;
    }
    .analizador-header h1 {
        color: #0b3d62;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0;
    }
    /* Portada: acá sí va la identidad completa, en grande. */
    .analizador-portada {
        background: linear-gradient(135deg, #f4fafc 0%, #ffffff 60%);
        border: 1px solid #d7e6ee;
        border-left: 6px solid #157a8a;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 2rem;
        box-shadow: 0 2px 10px rgba(11, 61, 98, 0.06);
    }
    .analizador-portada img {
        height: 150px;
        width: auto;
    }
    .analizador-portada .autor {
        color: #0b3d62;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }
    .analizador-portada .credencial {
        color: #157a8a;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .analizador-portada .lab {
        color: #3c6b82;
        font-size: 0.95rem;
        margin-top: 0.6rem;
    }
    /* Los botones de hoja ocupan el ancho completo, sin amontonarse. */
    div[role="radiogroup"] {
        gap: 1.6rem;
    }
    [data-testid="stMetricValue"] {
        white-space: normal;
        overflow: visible;
        text-overflow: unset;
        line-height: 1.25;
    }
    .beta-card {
        background: linear-gradient(135deg, #f4fafc 0%, #ffffff 60%);
        border: 1px solid #d7e6ee;
        border-left: 6px solid #157a8a;
        border-radius: 14px;
        padding: 1.4rem 2rem;
        margin-bottom: 1rem;
    }
    .beta-card .label { color: #3c6b82; font-size: 1rem; font-weight: 600; margin: 0; }
    .beta-card .value { color: #0b3d62; font-size: 3rem; font-weight: 800; margin: 0; line-height: 1.1; }
    .beta-card .meta { color: #3c6b82; font-size: 0.95rem; margin-top: 0.4rem; }
    .conclusion-ok {
        background: #eef8f0; border-left: 6px solid #2f9e44; border-radius: 10px;
        padding: 0.9rem 1.2rem; color: #205a29; white-space: pre-line; font-family: monospace;
    }
    .conclusion-no {
        background: #fdf3ec; border-left: 6px solid #d9822b; border-radius: 10px;
        padding: 0.9rem 1.2rem; color: #7a4a12; white-space: pre-line; font-family: monospace;
    }
    .alert-box {
        background: #fdecec; border-left: 6px solid #d64545; border-radius: 10px;
        padding: 0.8rem 1.2rem; color: #7a1f1f;
    }
    .info-box {
        background: #f4fafc; border-left: 6px solid #157a8a; border-radius: 10px;
        padding: 0.9rem 1.2rem; color: #0b3d62;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_logo_html = f'<img src="data:image/png;base64,{_logo_b64}" alt="Elephant Data Labs" />' if _logo_b64 else ""
st.markdown(
    f"""
    <div class="analizador-header">
        {_logo_html}
        <h1>Analizador de Beta</h1>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Descargando y calculando…")
def run_case_cached(
    company_name: str,
    stock_ticker: str,
    index_name: str,
    index_ticker: str,
    index_source: str,
    start_date: date,
    end_date: date,
    frequency: str,
    price_field: str,
) -> BetaCaseResult:
    case = CaseParameters(
        company_name=company_name,
        stock_ticker=stock_ticker,
        index_name=index_name,
        index_ticker=index_ticker,
        index_source=index_source,  # type: ignore[arg-type]
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,  # type: ignore[arg-type]
        price_field=price_field,
    )
    return run_beta_case(case)


# Misma navegación que el proyecto EEFF hermano: botones tipo radio en vez de
# pestañas, con "Inicio" como portada que se muestra primero y bloquea el
# resto (st.stop()) hasta que se elige otra hoja.
page = st.radio(
    "Hoja",
    ["Inicio", "Analizador de Beta", "Días atípicos"],
    horizontal=True,
    label_visibility="collapsed",
)

# =============================================================================
# Hoja Inicio: portada, cómo usar la app, notas de datos y estado del catálogo.
# Mismo rol y misma estructura que el Inicio del proyecto EEFF hermano — es
# la puerta de entrada del laboratorio, no un reemplazo del README (que tiene
# el detalle técnico completo).
# =============================================================================
if page == "Inicio":
    st.markdown(
        f"""
        <div class="analizador-portada">
            {_logo_html}
            <div>
                <div class="autor">Carlos Alaniz Salinas</div>
                <div class="credencial">Ingeniero Civil Industrial · Magíster Data Science · Candidato a Magíster en Gestión de Inversiones Financieras</div>
                <div class="lab">Elephant Data Labs · Estimación determinística de Beta por regresión OLS contra un índice de mercado</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Qué hace esta aplicación")
    st.markdown(
        "Estima el **Beta** de una acción por regresión OLS contra un índice de mercado "
        "(modelo de mercado: `R_i = alpha + beta · R_m + epsilon`), lo valida con un segundo "
        "método independiente (`Cov(Ri,Rm) / Var(Rm)`) y detecta días con retornos atípicos "
        "frente a lo que predice el modelo."
    )

    st.markdown("### Cómo usarla")
    guide_cols = st.columns(2)
    with guide_cols[0]:
        st.markdown("**1. Elegir empresa e índice**")
        st.markdown(
            "En **Analizador de Beta**, elija la empresa del catálogo (27 componentes IPSA "
            "verificadas) o ingrese directamente el nombre y ticker de cualquier otra — "
            "incluidas empresas de EE.UU. como Google (`GOOGL`) o Apple (`AAPL`), no solo "
            "chilenas — con un buscador de Yahoo Finance como ayuda opcional. El índice de "
            "mercado funciona igual: S&P/CLX IPSA o S&P 500 del catálogo, u otro índice "
            "ingresado a mano (ver la nota de fuentes más abajo)."
        )
        st.markdown("**2. Definir ventana y frecuencia**")
        st.markdown(
            "Fechas de inicio y término, frecuencia (diaria, semanal o mensual) y si el precio "
            "de la acción va ajustado por dividendos (Adj Close) o sin ajustar (Close)."
        )
    with guide_cols[1]:
        st.markdown("**3. Calcular y revisar**")
        st.markdown(
            "Presione **CALCULAR BETA**: aparece el Beta, la tabla de regresión completa, el "
            "diagnóstico de significancia —con explicación en lenguaje simple— y el gráfico de "
            "dispersión interactivo."
        )
        st.markdown("**Qué se valida automáticamente**")
        st.markdown(
            "- La fuente real del índice se muestra siempre (Stooq, CSV manual, respaldo "
            "replicado o Yahoo Finance), con advertencia visible si es el respaldo replicado.\n"
            "- El Beta de la regresión OLS se contrasta contra un segundo cálculo independiente "
            "(Cov/Var); si no coinciden, se avisa.\n"
            "- **Días atípicos** reutiliza exactamente la misma regresión ya calculada — no "
            "vuelve a descargar ni a estimar nada por su cuenta."
        )

    st.markdown('### Sobre las fuentes del índice — por qué a veces no es la "oficial"')
    st.markdown(
        "- **S&P/CLX IPSA**: Yahoo Finance no tiene historial usable para este ticker, así que "
        "se usa **Stooq**. Stooq bloquea las descargas automatizadas con bastante frecuencia "
        "(verificación anti-bot de Cloudflare) — cuando eso pasa, la app cae automáticamente a "
        "un **CSV manual** (`data/raw/ipsa_manual.csv`, ya incluido, con datos reales del IPSA "
        "2021–2025 bajados a mano de Investing.com) y, si tampoco existe ese archivo, a un "
        "**índice replicado** con las componentes del IPSA en Yahoo Finance — una aproximación, "
        "no el índice oficial.\n"
        "- **¿Por qué no se descarga de Investing.com automáticamente, si de ahí salió el CSV "
        "manual?** Su API solo responde dentro de una sesión de navegador real (con las cookies "
        "y el contexto que deja cargar la página primero), igual que Stooq exige resolver su "
        "verificación anti-bot — un `requests.get` normal, que es como corre la app en cada "
        "cálculo, no puede replicar eso de forma confiable ni automática. Por eso ese dato se "
        "bajó **una vez, a mano**, y quedó guardado como CSV en el proyecto (ver "
        "`data/raw/README.md` para refrescarlo más adelante).\n"
        "- **S&P 500**: sin este problema — Yahoo Finance tiene el historial completo de "
        "`^GSPC`, se descarga directo de ahí, sin pasar por Stooq ni por descargas manuales."
    )

    st.markdown("### Estado actual del catálogo")
    catalog_inventory = pd.DataFrame(
        [
            {
                "Catálogo": "Empresas (IPSA)",
                "Entradas": len(KNOWN_COMPANIES),
                "Detalle": "Verificadas a mano en Yahoo Finance; también se puede buscar cualquier otra empresa.",
            },
            {
                "Catálogo": "Índices",
                "Entradas": len(KNOWN_INDICES),
                "Detalle": "S&P/CLX IPSA (Stooq + respaldo automático) y S&P 500 (Yahoo Finance directo).",
            },
        ]
    )
    st.dataframe(
        catalog_inventory, use_container_width=True, hide_index=True, height=table_height(catalog_inventory)
    )
    manual_csv_path = ROOT / "data" / "raw" / "ipsa_manual.csv"
    if manual_csv_path.exists():
        st.caption(
            "`data/raw/ipsa_manual.csv` está presente: el IPSA usa datos reales aunque Stooq "
            "esté bloqueado ese día."
        )
    else:
        st.caption(
            "`data/raw/ipsa_manual.csv` no está presente todavía: si Stooq falla, el IPSA caerá "
            "al índice replicado (aproximado)."
        )

    st.markdown("### Qué hay en cada hoja")
    st.markdown(
        "- **Analizador de Beta** — parámetros, botón de cálculo, Beta y estadísticos completos, "
        "diagnóstico explicado en lenguaje simple, gráfico interactivo y validación cruzada.\n"
        "- **Días atípicos** — reutiliza el último cálculo (mismo alpha, beta y ventana): tabla y "
        "gráfico interactivo de los días que más se alejaron de lo que predice el modelo.\n"
        "- **Comparables** — módulo en preparación (ajuste de Beta por apalancamiento, método "
        "Hamada; ver `src/beta_analyzer/comparables.py`)."
    )
    st.markdown(
        "**Por qué regresión OLS determinística** — es el método estándar para estimar Beta en "
        "finanzas corporativas, y acá queda totalmente trazable: cada número del tablero "
        "(retornos, regresión, p-values, R², intervalo de confianza, outliers) se puede volver a "
        "calcular a mano con la misma fórmula y la misma versión de librería fijada en "
        "`requirements.txt`. Nada de esto está hardcodeado a una empresa o índice — cambiar los "
        "parámetros alcanza, no hace falta tocar código (ver `README.md`, sección 'Reutilizar el "
        "proyecto para otra empresa')."
    )
    st.stop()

# =============================================================================
# Hoja Analizador de Beta: parámetros, botón, resultados y validación.
# =============================================================================
if page == "Analizador de Beta":
    st.markdown("### Parámetros")

    # Fuera de un st.form a propósito: así los combo box de empresa e índice
    # reaccionan al tiro (cambian el ticker, muestran/ocultan campos
    # manuales) sin tener que apretar el botón. CALCULAR BETA sigue siendo
    # el único gatillo que efectivamente descarga datos y corre la regresión.
    col1, col2 = st.columns(2)
    with col1:
        company_label = st.selectbox(
            "Empresa",
            options=company_labels(),
            index=0,
            help="Catálogo verificado a mano en Yahoo Finance (ver hoja Inicio). "
            "Para otra empresa, elija la última opción e ingrese el ticker.",
        )
        known_company = find_by_name(company_label)
        if known_company is not None:
            company_name = known_company.name
            stock_ticker = known_company.ticker
            st.caption(f"Ticker verificado: `{known_company.ticker}` · Sector: {known_company.sector}")
        else:
            # Mismo patrón que "Otro índice" en la columna de la derecha:
            # los campos de nombre/ticker aparecen al tiro, sin tener que
            # buscar primero — así se puede escribir directamente GOOGL,
            # AAPL, MSFT o cualquier otra empresa que exista en Yahoo
            # Finance, no solo las del catálogo IPSA. El buscador queda
            # como ayuda opcional más abajo, para cuando no se conoce el
            # ticker exacto.
            st.session_state.setdefault("manual_company_name", "")
            st.session_state.setdefault("manual_company_ticker", "")

            manual_name_col, manual_ticker_col = st.columns(2)
            with manual_name_col:
                manual_name = st.text_input(
                    "Nombre de la empresa",
                    key="manual_company_name",
                    placeholder="p. ej. Alphabet Inc. (Google)",
                )
            with manual_ticker_col:
                manual_ticker = st.text_input(
                    "Ticker (Yahoo Finance)",
                    key="manual_company_ticker",
                    placeholder="p. ej. GOOGL, AAPL, SQM-B.SN",
                    help="Tal como lo reconoce Yahoo Finance. Empresas de EE.UU. normalmente "
                    "sin sufijo (GOOGL, AAPL, MSFT); empresas chilenas suelen llevar el "
                    "sufijo .SN.",
                )
            company_name = manual_name or manual_ticker
            stock_ticker = manual_ticker

            with st.expander("¿No conoce el ticker exacto? Buscar por nombre en Yahoo Finance"):
                st.session_state.setdefault("search_results", [])
                st.session_state.setdefault("search_query", "")

                search_query = st.text_input(
                    "Nombre o nemotécnico a buscar",
                    value=st.session_state["search_query"],
                    placeholder="p. ej. Google, ENTEL, banco de chile, SONDA",
                    help="Se busca directamente en Yahoo Finance. Si lo encuentra, elija el "
                    "resultado correcto de la lista y presione «Usar este resultado» para "
                    "completar los campos de arriba.",
                )
                search_clicked = st.button("Buscar en Yahoo Finance")

                if search_clicked:
                    st.session_state["search_query"] = search_query
                    try:
                        st.session_state["search_results"] = search_tickers(search_query)
                    except DataSourceError as exc:
                        st.session_state["search_results"] = []
                        st.error(str(exc))

                results = st.session_state["search_results"]
                if search_clicked and not results:
                    st.warning(
                        f"No se encontró ningún ticker para «{search_query}» en Yahoo Finance. "
                        "Pruebe con el nombre completo de la empresa, con el nemotécnico exacto, "
                        "o ingrese el ticker directamente arriba si ya lo conoce."
                    )

                if results:
                    chosen_label = st.selectbox("Resultados encontrados", options=[m.label for m in results])
                    chosen = next(m for m in results if m.label == chosen_label)
                    if chosen.is_santiago:
                        st.caption(f"`{chosen.symbol}` · Bolsa de Santiago, CLP.")
                    else:
                        st.caption(f"`{chosen.symbol}` · {chosen.exchange}.")
                    # Los campos de nombre/ticker de arriba ya se instanciaron con
                    # key="manual_company_name"/"manual_company_ticker" en este mismo
                    # rerun, así que Streamlit no deja tocar ese session_state acá
                    # directamente (StreamlitAPIException). La forma correcta es un
                    # callback on_click: se ejecuta ANTES de que el próximo rerun
                    # vuelva a crear esos widgets, así que ahí sí se puede escribir.
                    st.button(
                        "Usar este resultado",
                        key="use_search_result",
                        on_click=_apply_search_result,
                        args=(chosen.name, chosen.symbol),
                    )
        start_date = st.date_input("Fecha inicio", value=DEFAULT_CASE.start_date)
        end_date = st.date_input("Fecha término", value=DEFAULT_CASE.end_date)

    with col2:
        index_label = st.selectbox(
            "Índice de mercado",
            options=index_labels(),
            index=0,
            help="S&P/CLX IPSA (Chile) o S&P 500 (EE.UU.), verificados — ver hoja "
            "'Inicio'. Para otro índice, elija la última opción.",
        )
        known_index = find_index_by_label(index_label)
        if known_index is not None:
            index_name = known_index.name
            index_ticker = known_index.ticker
            index_source = known_index.source
            fuente_txt = "Stooq (+ respaldo automático)" if known_index.source == "stooq" else "Yahoo Finance"
            st.caption(f"Ticker: `{known_index.ticker}` · Fuente: {fuente_txt}")
            st.caption(known_index.note)
        else:
            index_name = st.text_input("Nombre del índice", value="")
            index_ticker = st.text_input(
                "Ticker del índice",
                value="",
                help="Ticker tal como lo reconoce la fuente elegida abajo.",
            )
            index_source_choice = st.selectbox(
                "Fuente de descarga del índice",
                options=["Stooq (con respaldo automático)", "Yahoo Finance"],
                index=0,
                help="Elija Yahoo Finance si el índice tiene historial completo ahí "
                "(como ^GSPC). Elija Stooq si es como ^IPSA, sin historial usable en "
                "Yahoo — Stooq trae su propio respaldo automático si falla.",
            )
            index_source = "stooq" if index_source_choice.startswith("Stooq") else "yahoo"

        frequency_label = st.selectbox(
            "Frecuencia", options=list(FREQUENCY_LABELS.values()), index=0
        )
        price_field_label = st.selectbox(
            "Precio de la acción",
            options=["Ajustado por dividendos (Adj Close)", "Cierre sin ajustar (Close)"],
            index=0,
        )

    frequency = {v: k for k, v in FREQUENCY_LABELS.items()}[frequency_label]
    price_field = "adj_close" if price_field_label.startswith("Ajustado") else "close"

    submitted = st.button("CALCULAR BETA", type="primary", use_container_width=True)
    if submitted and (not company_name or not stock_ticker):
        st.error("Ingrese nombre de empresa y ticker antes de calcular.")
        submitted = False
    if submitted and (not index_name or not index_ticker):
        st.error("Ingrese nombre y ticker del índice antes de calcular.")
        submitted = False

    if submitted:
        try:
            result = run_case_cached(
                company_name,
                stock_ticker,
                index_name,
                index_ticker,
                index_source,
                start_date,
                end_date,
                frequency,
                price_field,
            )
        except DataSourceError as exc:
            st.error(f"No se pudo obtener la información de mercado.\n\n{exc}")
            st.stop()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        # Se guarda en session_state para que la hoja de Días atípicos pueda
        # leer el último cálculo aunque el usuario cambie de hoja, sin
        # recalcular nada.
        st.session_state["last_result"] = result

    result: BetaCaseResult | None = st.session_state.get("last_result")

    if result is None:
        st.markdown(
            f"""
            <div class="info-box">
                Complete los parámetros y presione <b>CALCULAR BETA</b>. Por defecto se
                carga el caso de prueba: {DEFAULT_CASE.company_name} ({DEFAULT_CASE.stock_ticker})
                contra {DEFAULT_CASE.index_name}, {DEFAULT_CASE.start_date} a
                {DEFAULT_CASE.end_date}, frecuencia {DEFAULT_CASE.frequency_label.lower()}.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        reg = result.regression
        val = result.validation

        st.markdown(
            f"""
            <div class="beta-card">
                <p class="label">BETA</p>
                <p class="value">β = {reg.beta:.2f}</p>
                <p class="meta">
                    {result.case.company_name} vs. {result.case.index_name} &nbsp;·&nbsp;
                    Frecuencia: {result.case.frequency_label.lower()} &nbsp;·&nbsp;
                    Período: {reg.start_date} – {reg.end_date} &nbsp;·&nbsp;
                    {reg.n_obs} observaciones
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "replicado" in result.index_source_label.lower():
            st.warning(
                "⚠️ El índice de mercado usado NO es el oficial: la fuente configurada no "
                f"respondió, así que se usó un respaldo automático. Detalle: "
                f"{result.index_source_label}."
            )
        else:
            st.caption(f"Fuente del índice: {result.index_source_label}")

        metric_cols = st.columns(6)
        metric_cols[0].metric("Alpha", f"{reg.alpha:.5f}")
        metric_cols[1].metric("R²", f"{reg.r_squared:.4f}")
        metric_cols[2].metric("p-value (β)", f"{reg.beta_p_value:.4f}")
        metric_cols[3].metric("Error estándar (β)", f"{reg.beta_std_error:.4f}")
        metric_cols[4].metric("Observaciones", f"{reg.n_obs}")
        metric_cols[5].metric("IC 95% (β)", f"[{reg.beta_ci_low:.3f}, {reg.beta_ci_high:.3f}]")

        st.markdown("### Regresión")
        regression_table = pd.DataFrame(
            {
                "Estadístico": [
                    "Alpha (intercepto)",
                    "Beta (pendiente)",
                    "Error estándar de Beta",
                    "t-stat de Beta",
                    "p-value de Beta",
                    "p-value de Alpha",
                    "R²",
                    "N° observaciones",
                    "IC 95% Beta — límite inferior",
                    "IC 95% Beta — límite superior",
                    "Frecuencia",
                    "Período",
                    "Fuente del índice",
                ],
                "Valor": [
                    f"{reg.alpha:.6f}",
                    f"{reg.beta:.6f}",
                    f"{reg.beta_std_error:.6f}",
                    f"{reg.beta_t_stat:.4f}",
                    f"{reg.beta_p_value:.6f}",
                    f"{reg.alpha_p_value:.6f}",
                    f"{reg.r_squared:.6f}",
                    f"{reg.n_obs}",
                    f"{reg.beta_ci_low:.6f}",
                    f"{reg.beta_ci_high:.6f}",
                    result.case.frequency_label,
                    f"{reg.start_date} – {reg.end_date}",
                    result.index_source_label,
                ],
            }
        )
        st.dataframe(regression_table, hide_index=True, use_container_width=True)

        conclusion = significance_conclusion(reg)
        css_class = "conclusion-ok" if reg.is_significant_5pct else "conclusion-no"
        st.markdown("### Diagnóstico estadístico")
        st.markdown(f'<div class="{css_class}">{conclusion}</div>', unsafe_allow_html=True)

        with st.expander("¿Qué significa cada resultado? (explicación a prueba de estudiantes)", expanded=True):
            st.markdown(f"**Beta.** {beta_interpretation(reg.beta)}")
            st.markdown(f"**R² (qué tanto explica el mercado).** {r_squared_interpretation(reg.r_squared)}")
            st.markdown(f"**p-value (¿es confiable el Beta?).** {p_value_interpretation(reg)}")
            st.markdown(f"**Intervalo de confianza.** {ci_interpretation(reg)}")
            st.caption(
                "Estas cuatro explicaciones se arman con reglas fijas a partir de los "
                "números de la regresión (no hay ningún modelo de lenguaje generando "
                "este texto) — son la misma lógica de umbrales que ya usa la "
                "conclusión de significancia de arriba."
            )

        st.markdown("### Gráfico")
        st.caption("Pase el mouse sobre un punto para ver la fecha y los retornos exactos.")
        fig = scatter_with_regression(result.returns, reg, result.case)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Validación (OLS vs. Cov/Var)")
        validation_table = pd.DataFrame(
            {
                "Método": ["Regresión OLS", "Covarianza / Varianza", "Diferencia"],
                "Beta": [f"{val.beta_ols:.4f}", f"{val.beta_cov_var:.4f}", f"{val.difference:.8f}"],
            }
        )
        st.dataframe(validation_table, hide_index=True, use_container_width=True)

        if not val.is_consistent:
            st.markdown(
                '<div class="alert-box">⚠️ La diferencia entre el Beta de la regresión OLS y el '
                "Beta Cov/Var no es prácticamente cero. Revise la alineación de fechas y la fuente "
                "de datos antes de reportar este resultado.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.success("Beta OLS y Beta Cov/Var coinciden: la estimación es consistente.")

# =============================================================================
# Hoja Días atípicos: usa el último cálculo de Analizador de Beta.
# =============================================================================
if page == "Días atípicos":
    result = st.session_state.get("last_result")

    if result is None:
        st.markdown(
            """
            <div class="info-box">
                Primero calcule un Beta en la hoja <b>Analizador de Beta</b> — esta hoja
                reutiliza esa regresión (mismo alpha, beta y ventana), no vuelve a
                descargar ni a estimar nada por su cuenta.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        reg = result.regression
        st.caption(
            f"Usando el último cálculo: {result.case.company_name} vs. {result.case.index_name} "
            f"({result.case.frequency_label.lower()}, {reg.start_date} – {reg.end_date})."
        )
        st.caption(
            "Metodología (adaptada del proyecto Bootcamp de detección de movimientos "
            "anómalos, generalizada aquí al índice de mercado configurado en este caso "
            "en vez de un mercado fijo): para cada observación se calcula el residuo "
            "frente a la recta de regresión — lo que el índice NO explica del retorno "
            "de la acción ese día — y su z-score. Un día se marca atípico si el valor "
            "absoluto de su z-score supera el umbral."
        )
        umbral = st.number_input(
            "Umbral (desviaciones estándar de z-score)",
            min_value=0.1,
            value=DEFAULT_Z_THRESHOLD,
            step=0.1,
            help="Un día se marca como atípico si |z-score| > este umbral.",
        )

        try:
            outlier_result = detect_outliers(result.returns, reg, threshold=umbral)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            summary_cols = st.columns(3)
            summary_cols[0].metric("Días atípicos", f"{outlier_result.n_outliers}")
            summary_cols[1].metric("Positivos (sobre lo esperado)", f"{outlier_result.n_positive}")
            summary_cols[2].metric("Negativos (bajo lo esperado)", f"{outlier_result.n_negative}")

            if outlier_result.n_outliers == 0:
                st.info(
                    f"Ningún día superó ±{umbral:.1f} desviaciones estándar de z-score "
                    "en esta ventana."
                )
            else:
                st.markdown("##### Listado de días atípicos")
                st.caption(
                    "Cada fila es un día donde la acción se movió distinto a lo que el "
                    "modelo esperaba según el mercado ese día. **Residuo** es esa "
                    "diferencia (retorno real − retorno predicho); **Z-score** mide qué "
                    "tan extremo fue ese residuo comparado con un día normal (más lejos "
                    "de 0 = más atípico)."
                )
                index_col = f"Retorno {result.case.index_name}"
                outliers_table = outlier_result.outliers.rename(
                    columns={
                        "retorno_activo": "Retorno acción",
                        "retorno_mercado": index_col,
                        "residuo": "Residuo",
                        "z_score": "Z-score",
                        "z_score_abs": "|Z-score|",
                    }
                ).reset_index(names="Fecha")
                outliers_table["Dirección"] = outliers_table["Residuo"].apply(
                    lambda r: "🔼 Sobre lo esperado" if r > 0 else "🔽 Bajo lo esperado"
                )
                outliers_table = outliers_table[
                    ["Fecha", "Dirección", "Retorno acción", index_col, "Residuo", "Z-score", "|Z-score|"]
                ]
                st.dataframe(
                    outliers_table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Fecha": st.column_config.DateColumn(
                            "Fecha", help="Día de la observación atípica.", format="YYYY-MM-DD"
                        ),
                        "Dirección": st.column_config.TextColumn(
                            "Dirección",
                            help="Si la acción rindió más (🔼) o menos (🔽) de lo que el "
                            "modelo predecía ese día, según el mercado.",
                        ),
                        "Retorno acción": st.column_config.NumberColumn(
                            "Retorno acción",
                            help=f"Retorno real de {result.case.company_name} ese día.",
                            format="percent",
                        ),
                        index_col: st.column_config.NumberColumn(
                            index_col,
                            help=f"Retorno de {result.case.index_name} ese mismo día (lo que el modelo usa para predecir).",
                            format="percent",
                        ),
                        "Residuo": st.column_config.NumberColumn(
                            "Residuo",
                            help="Retorno real menos el retorno predicho por la regresión: "
                            "la parte del movimiento que el mercado NO explica.",
                            format="percent",
                        ),
                        "Z-score": st.column_config.NumberColumn(
                            "Z-score",
                            help="Cuántas desviaciones estándar se alejó el residuo de su "
                            "promedio. Positivo = sobre lo esperado, negativo = bajo lo esperado.",
                            format="%.2f",
                        ),
                        "|Z-score|": st.column_config.NumberColumn(
                            "|Z-score|",
                            help="Valor absoluto del Z-score — entre más alto, más atípico "
                            "fue el día (este es el que se compara contra el umbral).",
                            format="%.2f",
                        ),
                    },
                )

                st.markdown("##### Precio con días atípicos marcados")
                st.caption("Pase el mouse sobre un punto rojo para ver la fecha, el precio y su z-score.")
                price_series = resample_price_field(
                    result.stock_prices, result.case.price_field, result.case
                )
                outlier_fig = price_with_outliers(price_series, outlier_result.outliers, result.case)
                st.plotly_chart(outlier_fig, use_container_width=True)
