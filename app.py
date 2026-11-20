# -*- coding: utf-8 -*-
"""
MPD · Grupo 2 — Estudio de modelos para una pizzería local
==========================================================
Aplicación única en Streamlit que integra tres modelos conectados entre sí:

    💲 Elasticidad de precios  →  📈 Demanda (series de tiempo)  →  📦 Inventario (revisión continua)

Cadena de dependencias (todo se recalcula al mover cualquier parámetro):
  1. El escenario de precios (Δ%) y la elasticidad ε generan un multiplicador de demanda.
  2. El pronóstico de series de tiempo (Holt-Winters / ARIMA / SARIMA) fija el nivel
     de demanda esperado respecto al histórico (factor g).
  3. Ambos factores escalan el consumo diario de insumos, y con él SS, ROP, EOQ y costos.

Ejecución local:  streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import warnings

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf as sm_acf, adfuller, pacf

# ----------------------------------------------------------------------------
# 0. CONFIGURACIÓN GENERAL Y ESTILO
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MPD · Pizzería: Elasticidad + Demanda + Inventario",
                   page_icon="🍕", layout="wide")

PALETA = ["#C1272D", "#F4A300", "#6B8E23", "#8B4513", "#F5D061"]
COLOR_BASE, COLOR_ESC = "#7A2E0E", "#C1272D"

Z_TABLA = {"90 %": 1.282, "95 %": 1.645, "97.5 %": 1.960, "99 %": 2.326}
OPC_TOP5 = "Top 5 agregado"
OPC_TODAS = "Todas las pizzas (pizzería completa)"
M_HW, M_ARIMA, M_SARIMA = "Holt-Winters", "ARIMA", "SARIMA"
M_NAIVE, M_MA4 = "Ingenuo (última semana)", "Media móvil 4 semanas"
MODELOS_TS = [M_HW, M_ARIMA, M_SARIMA]
MODELOS_TODOS = [M_NAIVE, M_MA4] + MODELOS_TS
M_AUTO = "Automático (menor MAPE en backtesting)"
IDX_DIA = "Diario (order_date) · estacionalidad semanal m = 7"
IDX_SEM = "Semanal (bloques de 7 días) · ciclo m = 4"
VARS_Y = {"Unidades (pizzas)": "unidades", "Ventas (US$)": "ventas", "Pedidos": "pedidos"}
COLUMNAS_REQ = ["order_id", "quantity", "order_date", "unit_price", "total_price",
                "pizza_size", "pizza_category", "pizza_ingredients", "pizza_name"]

st.markdown(
    """
    <style>
    .stApp { background-color: #FFF8EE; }
    h1, h2, h3 { color: #7A2E0E; font-family: 'Trebuchet MS', sans-serif; }
    h1 { border-bottom: 4px solid #C1272D; padding-bottom: 8px; }
    [data-testid="stSidebar"] { background-color: #FBE7C8; }
    [data-testid="stMetricValue"] { color: #C1272D; }
    hr { border-top: 2px dashed #F4A300; }
    </style>
    """,
    unsafe_allow_html=True,
)


def estilo(ax):
    ax.set_facecolor("#FFF8EE")
    ax.figure.set_facecolor("#FFF8EE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#7A2E0E")
    return ax


def mostrar(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8-sig")


# ----------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# ----------------------------------------------------------------------------
def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    faltan = [c for c in COLUMNAS_REQ if c not in df.columns]
    if faltan:
        raise ValueError("Al CSV le faltan las columnas: " + ", ".join(faltan))
    txt = df["order_date"].astype(str).str.strip().str.replace("-", "/", regex=False)
    fechas = pd.to_datetime(txt, format="%d/%m/%Y", errors="coerce")
    if fechas.isna().mean() > 0.5:  # por si el archivo viniera como mes/día/año
        fechas = pd.to_datetime(txt, format="%m/%d/%Y", errors="coerce")
    df["order_date"] = fechas
    for c in ["quantity", "unit_price", "total_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["pizza_size", "pizza_category", "pizza_ingredients", "pizza_name"]:
        df[c] = df[c].astype(str).str.strip()
    df = df.dropna(subset=["order_date", "quantity", "unit_price", "total_price"])
    return df.reset_index(drop=True)


def _leer_csv(texto: str) -> pd.DataFrame:
    primera = texto.split("\n", 1)[0]
    sep = ";" if primera.count(";") > primera.count(",") else ","
    return _limpiar(pd.read_csv(io.StringIO(texto), sep=sep))


@st.cache_data(show_spinner="Cargando datos…")
def cargar_desde_bytes(contenido: bytes) -> pd.DataFrame:
    return _leer_csv(contenido.decode("utf-8-sig", errors="replace"))


def obtener_datos(subido) -> pd.DataFrame:
    """Solo se trabaja con el archivo que carga el usuario (no hay datos precargados)."""
    try:
        return cargar_desde_bytes(subido.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()


# ----------------------------------------------------------------------------
# 2. MODELO DE ELASTICIDAD DE PRECIOS (OLS log-log + benchmarks)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def agg_variantes(_df, clave: str) -> pd.DataFrame:
    agg = (_df.groupby(["pizza_name", "pizza_category", "pizza_size"])
           .agg(precio_promedio=("unit_price", "mean"), cantidad_total=("quantity", "sum"),
                ingreso_total=("total_price", "sum")).reset_index())
    agg = agg[(agg["precio_promedio"] > 0) & (agg["cantidad_total"] > 0)].reset_index(drop=True)
    agg["ln_precio"] = np.log(agg["precio_promedio"])
    agg["ln_cantidad"] = np.log(agg["cantidad_total"])
    return agg


def disenio(sub: pd.DataFrame, tamano=True, categoria=True) -> pd.DataFrame:
    X = pd.DataFrame({"Intercepto": 1.0, "ln_precio": sub["ln_precio"].values}, index=sub.index)
    partes = [X]
    if tamano and sub["pizza_size"].nunique() > 1:
        partes.append(pd.get_dummies(sub["pizza_size"], prefix="tamaño", drop_first=True))
    if categoria and sub["pizza_category"].nunique() > 1:
        partes.append(pd.get_dummies(sub["pizza_category"], prefix="categoría", drop_first=True))
    return pd.concat(partes, axis=1).astype(float)


def ajustar_ols(X: pd.DataFrame, y: np.ndarray):
    """OLS con inferencia clásica (equivale a statsmodels.OLS). None si no es identificable."""
    Xv, y = X.values.astype(float), np.asarray(y, dtype=float)
    n, k = Xv.shape
    gl = n - k
    if gl <= 0 or np.linalg.matrix_rank(Xv) < k:
        return None
    beta = np.linalg.lstsq(Xv, y, rcond=None)[0]
    res = y - Xv @ beta
    rss, tss = float(res @ res), float(((y - y.mean()) ** 2).sum())
    cov = (rss / gl) * np.linalg.inv(Xv.T @ Xv)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), gl)
    tc = stats.t.ppf(0.975, gl)
    tabla = pd.DataFrame({"coef": beta, "error_std": se, "t": t, "p_valor": p,
                          "IC95_inf": beta - tc * se, "IC95_sup": beta + tc * se}, index=X.columns)
    r2 = 1 - rss / tss if tss > 0 else np.nan
    f = ((tss - rss) / (k - 1)) / (rss / gl) if k > 1 and rss > 0 else np.nan
    return {"tabla": tabla, "eps": float(beta[1]), "p": float(p[1]), "r2": r2,
            "r2_aj": 1 - (1 - r2) * (n - 1) / gl, "n": n, "k": k,
            "F": f, "p_F": float(stats.f.sf(f, k - 1, gl)) if np.isfinite(f) else np.nan}


def elasticidad_subconjunto(sub: pd.DataFrame):
    if len(sub) < 4 or sub["precio_promedio"].nunique() < 2:
        return None
    return ajustar_ols(disenio(sub), sub["ln_cantidad"].values)


@st.cache_data(show_spinner=False)
def modelo_global_y_segmentos(_agg, clave: str):
    g = ajustar_ols(disenio(_agg), _agg["ln_cantidad"].values)
    filas_c, filas_t = [], []
    for c, sub in _agg.groupby("pizza_category"):
        r = elasticidad_subconjunto(sub)
        filas_c.append({"Categoría": c, "N variantes": len(sub),
                        "Elasticidad": r["eps"] if r else np.nan,
                        "p-valor": r["p"] if r else np.nan, "R²": r["r2"] if r else np.nan,
                        "Estado": "estimada" if r else "no identificable"})
    for s, sub in _agg.groupby("pizza_size"):
        r = elasticidad_subconjunto(sub)
        filas_t.append({"Tamaño": s, "N variantes": len(sub),
                        "Elasticidad": r["eps"] if r else np.nan,
                        "p-valor": r["p"] if r else np.nan, "R²": r["r2"] if r else np.nan,
                        "Estado": "estimada" if r else "no identificable"})
    return g, pd.DataFrame(filas_c).set_index("Categoría"), pd.DataFrame(filas_t).set_index("Tamaño")


@st.cache_data(show_spinner="Entrenando benchmark (OLS vs Ridge vs Random Forest)…")
def benchmark_elasticidad(_df, clave: str):
    d = _df.assign(dow=_df["order_date"].dt.dayofweek, mes=_df["order_date"].dt.month)
    cel = (d.groupby(["pizza_name", "pizza_category", "pizza_size", "dow", "mes"])
           .agg(q=("quantity", "sum"), p=("unit_price", "mean")).reset_index())
    cel = cel[cel["q"] > 0].reset_index(drop=True)
    cat = pd.get_dummies(cel[["pizza_category", "pizza_size"]], drop_first=True)
    cal = pd.get_dummies(cel[["dow", "mes"]].astype(str), drop_first=True)
    X_lin = pd.concat([pd.DataFrame({"ln_p": np.log(cel["p"])}), cat, cal], axis=1).astype(float)
    y_log, y = np.log(cel["q"].values), cel["q"].values.astype(float)
    X_rf = pd.concat([cel[["p"]], pd.get_dummies(cel[["pizza_category", "pizza_size"]]),
                      cel[["dow", "mes"]]], axis=1).astype(float)
    kf = KFold(5, shuffle=True, random_state=42)

    def met(real, pred):
        ss_res, ss_tot = ((real - pred) ** 2).sum(), ((real - real.mean()) ** 2).sum()
        return (float(np.mean(np.abs(real - pred))), float(np.sqrt(np.mean((real - pred) ** 2))),
                float(1 - ss_res / ss_tot))

    filas = []
    for nombre, mod in [("OLS log-log", LinearRegression()), ("Ridge log-log (α=1)", Ridge(alpha=1.0))]:
        pred = np.exp(cross_val_predict(mod, X_lin, y_log, cv=kf))
        mae, rmse, r2 = met(y, pred)
        coef = mod.fit(X_lin, y_log).coef_[0]
        filas.append({"Modelo": nombre, "Elasticidad (ε)": coef, "MAE (uds)": mae, "RMSE (uds)": rmse, "R² (uds)": r2})
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    pred = cross_val_predict(rf, X_rf, y, cv=kf)
    mae, rmse, r2 = met(y, pred)
    rf.fit(X_rf, y)
    arr, aba = X_rf.copy(), X_rf.copy()
    arr["p"], aba["p"] = X_rf["p"] * 1.1, X_rf["p"] * 0.9
    eps_rf = np.log(max(rf.predict(arr).sum(), 1e-9) / max(rf.predict(aba).sum(), 1e-9)) / np.log(1.1 / 0.9)
    filas.append({"Modelo": "Random Forest (benchmark)", "Elasticidad (ε)": eps_rf,
                  "MAE (uds)": mae, "RMSE (uds)": rmse, "R² (uds)": r2})
    return pd.DataFrame(filas).set_index("Modelo"), len(cel)


def construir_elasticidades(modo, eps_manual, forzar, eps_global, tabla_cat, cats):
    """Devuelve (dict categoría→ε usada, lista de categorías que cayeron al valor global)."""
    eps, respaldo = {}, []
    for c in cats:
        if modo.startswith("Global"):
            eps[c] = eps_global
        elif modo.startswith("Manual"):
            eps[c] = eps_manual
        else:
            e = tabla_cat["Elasticidad"].get(c, np.nan)
            if not np.isfinite(e) or (forzar and e >= 0):
                eps[c] = eps_global
                respaldo.append(c)
            else:
                eps[c] = float(e)
    return eps, respaldo


def mult_demanda(pesos: pd.Series, deltas: dict, eps: dict, extra=0.0) -> float:
    """Σ w_c · (1+Δ_c)^(ε_c+extra). extra=0 → unidades; extra=1 → ingresos."""
    return float(sum(w * (1 + deltas[c] / 100.0) ** (eps[c] + extra) for c, w in pesos.items()))


def mult_insumos(shares: pd.DataFrame, deltas: dict, eps: dict) -> pd.Series:
    f = pd.Series({c: (1 + deltas[c] / 100.0) ** eps[c] for c in shares.columns})
    return (shares * f).sum(axis=1)


def simular_precio(df_seg: pd.DataFrame, eps: dict, grilla) -> pd.DataFrame:
    base = df_seg.groupby("pizza_category").agg(q=("quantity", "sum"), r=("total_price", "sum"))
    q0, r0 = base["q"].sum(), base["r"].sum()
    filas = []
    for d in grilla:
        k = 1 + d / 100.0
        q = sum(base.loc[c, "q"] * k ** eps[c] for c in base.index)
        r = sum(base.loc[c, "r"] * k ** (1 + eps[c]) for c in base.index)
        filas.append({"Δ precio (%)": d, "Precio medio ($)": r / q, "Unidades": q,
                      "Ingreso ($)": r, "Δ ingreso (%)": (r / r0 - 1) * 100,
                      "Δ unidades (%)": (q / q0 - 1) * 100})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# 3. MODELO DE DEMANDA CON SERIES DE TIEMPO (descomposición, Holt-Winters, ARIMA, SARIMA)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calcular_top5(_df, clave: str) -> pd.Series:
    return _df.groupby("pizza_name")["quantity"].sum().sort_values(ascending=False).head(5)


def filtrar_alcance(df: pd.DataFrame, productos: tuple, seleccion: str) -> pd.DataFrame:
    """Mismo subconjunto de pizzas para pronóstico, ingreso e inventario."""
    if seleccion == OPC_TODAS:
        return df
    if seleccion == OPC_TOP5:
        return df[df["pizza_name"].isin(productos)]
    return df[df["pizza_name"] == seleccion]


@st.cache_data(show_spinner=False)
def pesos_alcance(_d, clave_s: str):
    """Participación de cada categoría en unidades e ingresos, y precio medio por pizza (US$)."""
    pu = _d.groupby("pizza_category")["quantity"].sum()
    pr = _d.groupby("pizza_category")["total_price"].sum()
    precio_unit = float(_d["total_price"].sum() / _d["quantity"].sum())
    return pu / pu.sum(), pr / pr.sum(), precio_unit


@st.cache_data(show_spinner=False)
def construir_series(_d, clave_s: str):
    """Serie DIARIA continua (índice order_date) y serie SEMANAL en bloques de 7 días.

    · Los días sin ventas (local cerrado) se completan por interpolación para que la serie
      no tenga huecos (los modelos de series de tiempo exigen periodos consecutivos).
    · Las semanas son bloques de 7 días contados hacia atrás desde la última fecha, así
      ninguna semana queda incompleta (se descartan solo los días sobrantes del inicio).
    """
    dia = _d["order_date"].dt.normalize()
    g = _d.groupby(dia).agg(unidades=("quantity", "sum"), ventas=("total_price", "sum"),
                            pedidos=("order_id", "nunique"))
    idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
    n_faltantes = len(idx) - len(g)
    diaria = g.reindex(idx).interpolate(limit_direction="both")
    sobrantes = len(diaria) % 7
    diaria = diaria.iloc[sobrantes:].asfreq("D")
    n_sem = len(diaria) // 7
    semanal = diaria.groupby(np.repeat(np.arange(n_sem), 7)).sum()
    semanal.index = pd.DatetimeIndex(diaria.index[6::7], freq="7D")
    diaria.index.name = semanal.index.name = "fecha"
    return diaria, semanal, n_faltantes, sobrantes


def _mape(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.mean(np.abs((real - pred) / np.maximum(real, 1))) * 100)


def _metricas(real, pred):
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return (float(np.mean(np.abs(real - pred))), float(np.sqrt(np.mean((real - pred) ** 2))), _mape(real, pred))


def _sarimax(y, orden, estacional=(0, 0, 0, 0)):
    tendencia = "c" if (orden[1] == 0 and estacional[1] == 0) else "n"
    return SARIMAX(y, order=orden, seasonal_order=estacional, trend=tendencia,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def seleccionar_ordenes(y: pd.Series, m: int) -> dict:
    """Búsqueda por AIC (menor es mejor) de los órdenes ARIMA(p,d,q) y SARIMA(P,D,Q)m."""
    mejor_a, aic_a = (1, 0, 0), np.inf
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    aic = _sarimax(y, (p, d, q)).aic
                except Exception:  # noqa: BLE001
                    continue
                if aic < aic_a:
                    mejor_a, aic_a = (p, d, q), aic
    mejor_s, aic_s = (0, 0, 0, m), np.inf
    for P, D, Q in [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]:
        try:
            aic = _sarimax(y, mejor_a, (P, D, Q, m)).aic
        except Exception:  # noqa: BLE001
            continue
        if aic < aic_s:
            mejor_s, aic_s = (P, D, Q, m), aic
    return {"arima": mejor_a, "aic_arima": aic_a, "sarima": (mejor_a, mejor_s), "aic_sarima": aic_s}


def ajustar_ts(y: pd.Series, nombre: str, m: int, ordenes: dict):
    if nombre == M_HW:
        return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m,
                                    initialization_method="estimated").fit()
    if nombre == M_ARIMA:
        return _sarimax(y, ordenes["arima"])
    return _sarimax(y, *ordenes["sarima"])


def predecir_bloques(y, nombre, m, ordenes, pasos, agrupar):
    """Pronostica 'pasos' periodos y, si la serie es diaria, los suma en bloques de 7 días."""
    if nombre == M_NAIVE:
        base = y.iloc[-agrupar:].sum()
        return np.repeat(base, pasos // agrupar)
    if nombre == M_MA4:
        base = y.iloc[-4 * agrupar:].sum() / 4
        return np.repeat(base, pasos // agrupar)
    f = np.asarray(ajustar_ts(y, nombre, m, ordenes).forecast(pasos), float)
    return f.reshape(-1, agrupar).sum(axis=1)


@st.cache_data(show_spinner="Ajustando modelos de series de tiempo…")
def modelos_series(_d, clave_s: str, indice: str, variable: str, horizonte: int):
    """Ajusta y valida Holt-Winters, ARIMA y SARIMA (más dos referencias simples) sobre la serie elegida.
    Todo error se mide en SEMANAS (bloques de 7 días) para que los índices diario y semanal sean comparables."""
    warnings.filterwarnings("ignore")
    diaria, semanal, n_falt, sobr = construir_series(_d, clave_s)
    diario = indice == IDX_DIA
    y = (diaria if diario else semanal)[variable].astype(float)
    m = 7 if diario else 4                 # estacionalidad: día de la semana (diaria) o ciclo de 4 semanas
    agrupar = 7 if diario else 1           # periodos por semana
    h_bt = 4 * agrupar                      # backtesting: 4 semanas por corte
    min_train = 28 * agrupar                # al menos 28 semanas de entrenamiento
    ordenes = seleccionar_ordenes(y.iloc[:min_train], m)   # órdenes elegidos SOLO con el primer tramo

    reales_sem = semanal[variable].astype(float)
    filas, tablas = [], {}
    for nombre in MODELOS_TODOS:
        fechas, reales, preds = [], [], []
        for corte in range(min_train, len(y) - h_bt + 1, h_bt):
            try:
                p = predecir_bloques(y.iloc[:corte], nombre, m, ordenes, h_bt, agrupar)
            except Exception:  # noqa: BLE001
                p = np.full(4, np.nan)
            k = corte // agrupar
            reales.extend(reales_sem.iloc[k:k + 4].tolist()); preds.extend(p)
            fechas.extend(reales_sem.index[k:k + 4])
        tb = pd.DataFrame({"fecha": fechas, "real": reales, "prediccion": np.round(preds, 1)}).dropna()
        tablas[nombre] = tb
        mae, rmse, mape = _metricas(tb["real"], tb["prediccion"]) if len(tb) else (np.nan,) * 3
        filas.append({"Modelo": nombre, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "Folds": len(tb) // 4})
    comp = pd.DataFrame(filas).set_index("Modelo")

    # Pronóstico final: cada modelo se reajusta con TODA la serie y proyecta 'horizonte' semanas
    futuros = pd.DatetimeIndex([semanal.index[-1] + pd.Timedelta(days=7 * k) for k in range(1, horizonte + 1)])
    pron, ajustados = {}, {}
    for nombre in MODELOS_TODOS:
        try:
            pron[nombre] = predecir_bloques(y, nombre, m, ordenes, horizonte * agrupar, agrupar)
            if nombre in MODELOS_TS:
                ajustados[nombre] = ajustar_ts(y, nombre, m, ordenes).fittedvalues
        except Exception:  # noqa: BLE001
            pron[nombre] = np.full(horizonte, np.nan)
    pron = pd.DataFrame(pron, index=futuros)

    deco = seasonal_decompose(y, model="additive", period=m)
    adf_p = float(adfuller(y.dropna())[1])
    return {"y": y, "semanal": semanal, "m": m, "ordenes": ordenes, "comp": comp, "tablas": tablas,
            "pron": pron, "ajustados": ajustados, "deco": deco, "adf_p": adf_p,
            "acf": sm_acf(y, nlags=2 * m), "pacf": pacf(y, nlags=2 * m), "n_faltantes": n_falt, "sobrantes": sobr}


def descriptivos(df, columnas, moda_en=None) -> pd.DataFrame:
    """moda_en: columnas donde la moda tiene sentido (en variables continuas casi nunca se repite un valor)."""
    filas = []
    for c in columnas:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        moda = s.mode() if (moda_en is None or c in moda_en) else pd.Series(dtype=float)
        filas.append({"Variable": c, "Media": s.mean(), "Moda": moda.iloc[0] if len(moda) else np.nan,
                      "Desv. estándar": s.std(), "Mínimo": s.min(), "Máximo": s.max()})
    return pd.DataFrame(filas).set_index("Variable").round(3)
# ----------------------------------------------------------------------------
# 4. MODELO DE INVENTARIO (REVISIÓN CONTINUA, s-Q) — EN LA MISMA BASE SEMANAL QUE LA DEMANDA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Procesando consumo de insumos…")
def preparar_insumos(_df, clave: str, top_n: int):
    """Consumo de insumos en PORCIONES POR SEMANA, con los mismos bloques de 7 días que la demanda.

    Una porción = un ingrediente usado en una pizza vendida. Los días sin ninguna venta (local cerrado)
    se interpolan igual que en la serie de demanda; los días con ventas pero sin ese insumo valen 0.
    """
    d = _df[["order_date", "quantity", "pizza_category", "pizza_ingredients"]].copy()
    d["ingredient"] = d["pizza_ingredients"].str.split(",")
    d = d.explode("ingredient")
    d["ingredient"] = d["ingredient"].str.strip()
    d = d[d["ingredient"] != ""]
    top = d.groupby("ingredient")["quantity"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    dt = d[d["ingredient"].isin(top)]
    dias_venta = _df["order_date"].dt.normalize()
    consumo = dt.groupby([dt["order_date"].dt.normalize(), "ingredient"])["quantity"].sum().unstack(fill_value=0)
    idx = pd.date_range(dias_venta.min(), dias_venta.max(), freq="D")
    consumo = consumo.reindex(idx)[top]
    abiertos = idx.isin(dias_venta.unique())
    consumo.loc[abiertos] = consumo.loc[abiertos].fillna(0)
    consumo = consumo.interpolate(limit_direction="both")
    consumo = consumo.iloc[len(consumo) % 7:]                       # mismos bloques que la demanda
    n_sem = len(consumo) // 7
    semanal = consumo.groupby(np.repeat(np.arange(n_sem), 7)).sum()
    semanal.index = consumo.index[6::7]
    shares = dt.groupby(["ingredient", "pizza_category"])["quantity"].sum().unstack(fill_value=0).reindex(top)
    shares = shares.div(shares.sum(axis=1), axis=0)
    return semanal, shares


def estadisticos_base(consumo_sem: pd.DataFrame, pizzas_sem: float) -> pd.DataFrame:
    """Descriptivos del consumo semanal y conversión pizzas → porciones de cada insumo."""
    return pd.DataFrame({"Porciones_por_pizza": (consumo_sem.mean() / pizzas_sem).round(3),
                         "Media": consumo_sem.mean().round(2), "Moda": consumo_sem.round(0).mode().iloc[0],
                         "Desv_Std": consumo_sem.std().round(2), "Minimo": consumo_sem.min().round(2),
                         "Maximo": consumo_sem.max().round(2)})


def metricas_inventario(est: pd.DataFrame, factor, Z, L_dias, S, H_anual, sem_anio) -> pd.DataFrame:
    """SS, ROP, EOQ y costos, TODO POR SEMANA.

    μ y σ en porciones/semana · L en semanas (= días ÷ 7) · H semanal = H anual ÷ semanas del año.
    'factor' escala media y desviación del consumo histórico (coeficiente de variación constante).
    """
    mu = (est["Media"] * factor).round(2)
    sd = (est["Desv_Std"] * factor).round(2)
    L = L_dias / 7.0
    H = H_anual / sem_anio
    SS = np.ceil(Z * sd * np.sqrt(L)).astype(int)
    ROP = np.ceil(mu * L + SS).astype(int)
    EOQ = np.round(np.sqrt(2 * mu * S / H)).astype(int)
    sem_ped = pd.Series(np.where(mu > 0, EOQ / mu.replace(0, np.nan), np.nan), index=est.index).round(2)
    costo = np.sqrt(2 * mu * S * H)                                   # pedir + mantener el lote, S/ por semana
    return pd.DataFrame({"Demanda_semanal": mu, "Desv_semanal": sd, "SS": SS, "ROP": ROP, "EOQ": EOQ,
                         "Semanas_entre_pedidos": sem_ped, "Costo_EOQ_sem": costo.round(2),
                         "Costo_total_sem": (costo + H * SS).round(2)})


def pca_insumos(consumo_sem: pd.DataFrame):
    """Reducción de dimensiones: PCA sobre el consumo semanal estandarizado de los insumos."""
    p = PCA().fit(StandardScaler().fit_transform(consumo_sem.values))
    cargas = pd.DataFrame(p.components_[:2].T, index=consumo_sem.columns, columns=["CP1", "CP2"]).round(3)
    return p.explained_variance_ratio_, cargas


def simular_ciclo(mu_sem, ROP, EOQ, SS, L_dias, semanas=16):
    """Simulación día a día (consumo = μ/7), presentada en semanas."""
    stock, camino, serie = EOQ + SS, None, []
    for t in range(semanas * 7):
        if camino and t == camino[0]:
            stock += camino[1]; camino = None
        stock -= mu_sem / 7.0
        if stock <= ROP and camino is None:
            camino = (t + int(L_dias), EOQ)
        serie.append(stock)
    return np.arange(1, semanas * 7 + 1) / 7.0, np.array(serie)

# ----------------------------------------------------------------------------
# 5. CARGA DE DATOS (la app NO trae datos: todo aparece al cargar el CSV)
# ----------------------------------------------------------------------------
st.title("🍕 Pizzería · Elasticidad + Demanda + Inventario")
st.caption("MPD · Grupo 2 — Un solo sistema: el precio mueve la demanda, la demanda mueve el inventario, "
           "y el inventario (servicio y costos) vuelve sobre las ventas, el margen y el precio óptimo.")

st.sidebar.title("🎛️ Panel de control")
subido = st.sidebar.file_uploader("📂 Cargar CSV de ventas (separador ; o ,)", type=["csv"])
if subido is None:
    st.info("👈 **Cargue el archivo de ventas** (pizza_sales) en el panel lateral para iniciar el análisis. "
            "La aplicación no trae datos precargados: todos los modelos se calculan a partir del archivo cargado.")
    st.markdown("**Columnas requeridas:** " + ", ".join(f"`{c}`" for c in COLUMNAS_REQ) +
                ". Fechas en formato día/mes/año.")
    st.stop()
df = obtener_datos(subido)
clave = f"{len(df)}-{df['total_price'].sum():.2f}"
CATS = sorted(df["pizza_category"].unique())

agg = agg_variantes(df, clave)
ols_g, tabla_cat, tabla_tam = modelo_global_y_segmentos(agg, clave)
top5 = calcular_top5(df, clave)
productos_top5 = tuple(top5.index.tolist())

# ----------------------------------------------------------------------------
# 6. PANEL LATERAL: PARÁMETROS (cada uno alimenta a todo el sistema)
# ----------------------------------------------------------------------------
st.sidebar.subheader("💲 1 · Escenario de precios")
delta_global = st.sidebar.slider("Cambio de precio global (%)", -30, 30, 0, 1)
extra = {}
with st.sidebar.expander("Ajuste adicional por categoría (puntos %)"):
    for c in CATS:
        extra[c] = st.slider(f"{c}", -30, 30, 0, 1, key=f"extra_{c}")
deltas = {c: float(np.clip(delta_global + extra[c], -90, 200)) for c in CATS}
modo = st.sidebar.selectbox("Elasticidad a utilizar",
                            ["Global (OLS log-log)", "Por categoría (OLS)", "Manual"])
eps_manual = st.sidebar.slider("ε manual", -3.0, -0.05, -1.0, 0.05, disabled=(modo != "Manual"))
forzar = st.sidebar.checkbox("Si la categoría no es identificable o ε ≥ 0, usar la global", value=True)
aplicar_precio = st.sidebar.checkbox("Propagar el precio a demanda e inventario", value=True)
if all(v == 0 for v in deltas.values()):
    st.sidebar.caption("Con Δ precio = 0 % la elasticidad no cambia el escenario actual, "
                       "pero sí la curva de sensibilidad y el precio óptimo.")

st.sidebar.subheader("📈 2 · Demanda (series de tiempo)")
serie_sel = st.sidebar.selectbox("Alcance / serie a pronosticar", [OPC_TOP5, OPC_TODAS] + list(productos_top5),
                                 help="Define qué pizzas entran en el pronóstico Y en el inventario (mismo alcance).")
indice_sel = st.sidebar.selectbox("Índice temporal", [IDX_DIA, IDX_SEM],
                                  help="Diario: un dato por día (order_date), capta el patrón por día de la semana. "
                                       "Semanal: suma de bloques de 7 días.")
var_label = st.sidebar.selectbox("Variable a proyectar (pestaña Demanda)", list(VARS_Y.keys()),
                                 help="El sistema integrado (inventario, ingreso, margen) usa siempre unidades.")
var_sel = VARS_Y[var_label]
modelo_sel = st.sidebar.selectbox("Modelo de pronóstico", [M_AUTO] + MODELOS_TS)
horizonte = st.sidebar.slider("Horizonte (semanas)", 4, 12, 8)
aplicar_pron = st.sidebar.checkbox("Propagar el pronóstico al inventario", value=True)

st.sidebar.subheader("📦 3 · Inventario (revisión continua)")
top_n = st.sidebar.slider("N.º de insumos principales", 5, 15, 10)
L = st.sidebar.number_input("Lead time L (días)", 1, 30, 2)
nivel = st.sidebar.selectbox("Nivel de servicio (ciclo)", list(Z_TABLA.keys()), index=1)
Z = Z_TABLA[nivel]
S_ped = st.sidebar.number_input("Costo por pedido S (S/)", 1.0, 1000.0, 45.0, 1.0)
H_men = st.sidebar.number_input("Costo de mantener H (S/ por porción-año)", 0.1, 50.0, 1.0, 0.1)
sem_anio = st.sidebar.number_input("Semanas operativas por año", 40, 53, 52,
                                   help="Convierte H anual a H semanal (H ÷ semanas del año).")

st.sidebar.subheader("💵 4 · Economía del escenario")
cv = st.sidebar.slider("Costo de insumos por pizza (% del precio actual)", 0, 80, 30, 1,
                       help="Supuesto (el dataset no trae costos). Se fija en US$ por pizza: no cambia al mover el precio.")
tc = st.sidebar.number_input("Tipo de cambio (S/ por US$, supuesto)", 1.0, 10.0, 3.70, 0.01,
                             help="Ventas en US$ (moneda del dataset); S y H en soles. Convierte todo a una sola moneda.")

# ----------------------------------------------------------------------------
# 7. MOTOR INTEGRADO: Precio ⇄ Demanda ⇄ Inventario ⇄ Margen
# ----------------------------------------------------------------------------
eps, respaldo = construir_elasticidades(modo, eps_manual, forzar, ols_g["eps"], tabla_cat, CATS)
cero = {c: 0.0 for c in CATS}
d_efectivos = deltas if aplicar_precio else cero

# --- Alcance común: las MISMAS pizzas para demanda, ingreso e inventario
d_scope = filtrar_alcance(df, productos_top5, serie_sel)
clave_s = f"{clave}|{serie_sel}"
cats_scope = sorted(d_scope["pizza_category"].unique())

# --- Demanda (series de tiempo): el sistema integrado siempre proyecta UNIDADES
pes_u, pes_r, precio_unit = pesos_alcance(d_scope, clave_s)
ts_u = modelos_series(d_scope, clave_s, indice_sel, "unidades", horizonte)
ts_v = ts_u if var_sel == "unidades" else modelos_series(d_scope, clave_s, indice_sel, var_sel, horizonte)
bt_res = {n: (r["MAE"], r["RMSE"], r["MAPE (%)"], ts_u["tablas"][n]) for n, r in ts_u["comp"].iterrows()}
validos = [n for n in MODELOS_TS if np.isfinite(bt_res[n][2])]
modelo_usado = (min(validos, key=lambda n: bt_res[n][2]) if modelo_sel == M_AUTO else modelo_sel)
mejor_global = min(bt_res, key=lambda n: bt_res[n][2] if np.isfinite(bt_res[n][2]) else np.inf)
serie_sem = ts_u["semanal"]
fc = pd.DataFrame({"fecha": ts_u["pron"].index, "prediccion": ts_u["pron"][modelo_usado].round(1).values})
media_hist = float(serie_sem["unidades"].mean())
g_pron = float(fc["prediccion"].mean() / media_hist) if aplicar_pron else 1.0
pron_total = float(fc["prediccion"].sum())

# --- Inventario (insumos del mismo alcance)
consumo, shares = preparar_insumos(d_scope, clave_s, top_n)
est = estadisticos_base(consumo, media_hist)        # porciones/semana y porciones por pizza
met_hist = metricas_inventario(est, 1.0, Z, L, S_ped, H_men, sem_anio)

G_Z = float(stats.norm.pdf(Z) - Z * stats.norm.sf(Z))  # función de pérdida normal estándar G(z)
costo_pizza = cv / 100.0 * precio_unit                 # US$ por pizza (fijo)


def evaluar(dd: dict, Zv: float | None = None) -> dict:
    """Un escenario completo, con todas las unidades llevadas al MISMO horizonte y a US$.
    Zv permite evaluar otro nivel de servicio (escenarios prescriptivos)."""
    Zv = Z if Zv is None else Zv
    Gv = float(stats.norm.pdf(Zv) - Zv * stats.norm.sf(Zv))
    m_u = mult_demanda(pes_u, dd, eps, 0.0)            # precio → unidades
    m_r = mult_demanda(pes_r, dd, eps, 1.0)            # precio → ingreso
    m_i = mult_insumos(shares, dd, eps)                # precio → consumo de cada insumo
    fac = g_pron * m_i                                 # pronóstico × precio → insumos
    met = metricas_inventario(est, fac, Zv, L, S_ped, H_men, sem_anio)
    # inventario → ventas: faltante esperado por ciclo = σ_L·G(z); fill rate = 1 − faltante / Q
    faltante = met["Desv_semanal"] * np.sqrt(L / 7.0) * Gv
    fill_i = (1 - faltante / met["EOQ"].where(met["EOQ"] > 0)).clip(0, 1).fillna(1.0)
    w = met["Demanda_semanal"] / met["Demanda_semanal"].sum()
    fill = float((w * fill_i).sum())
    dem = pron_total * m_u
    vend = dem * fill
    ingreso = pron_total * precio_unit * m_r * fill
    c_ins = costo_pizza * vend
    c_inv_s = float(met["Costo_total_sem"].sum()) * horizonte          # S/ por semana × semanas
    c_inv = c_inv_s / tc
    return {"m_u": m_u, "m_r": m_r, "m_i": m_i, "fac": fac, "met": met, "fill_i": fill_i, "fill": fill,
            "dem": dem, "vend": vend, "perdidas": dem - vend, "ingreso": ingreso, "c_ins": c_ins,
            "c_inv": c_inv, "c_inv_s": c_inv_s, "margen": ingreso - c_ins - c_inv}


R = evaluar(d_efectivos)   # escenario actual
B = evaluar(cero)          # base: mismo pronóstico e inventario, precio sin cambio
# Precio óptimo por MARGEN: aquí el inventario (S, H, L, servicio) y los costos vuelven sobre el precio
sens = pd.DataFrame([{"d": d, **{k: v for k, v in evaluar(
    {c: float(np.clip(d + extra[c], -90, 200)) for c in CATS}).items() if np.isscalar(v)}}
    for d in range(-30, 31, 5)])
opt = sens.loc[sens["margen"].idxmax()]

# alias usados por las pestañas de detalle
m_unid_serie, m_insumo, factor_esc = R["m_u"], R["m_i"], R["fac"]
met_base, met_esc = B["met"], R["met"]
fc["escenario"] = (fc["prediccion"] * R["m_u"]).round(1)
fc["vendidas"] = (fc["escenario"] * R["fill"]).round(1)
fc["ingreso_base"] = fc["prediccion"] * precio_unit * B["fill"]
fc["ingreso_escenario"] = fc["prediccion"] * precio_unit * R["m_r"] * R["fill"]


def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f} % vs base" if b else "—"


# ----------------------------------------------------------------------------
# 8. INTERFAZ
# ----------------------------------------------------------------------------
tab_panel, tab_ind, tab_esc, tab_dem, tab_el, tab_inv, tab_met = st.tabs(
    ["🧭 Panel integrado", "📊 Indicadores clave", "🎯 Escenarios prescriptivos",
     "📈 Demanda (series de tiempo)", "💲 Elasticidad de precios", "📦 Inventario", "📚 Metodología y supuestos"])

# ============================ PANEL INTEGRADO ================================
with tab_panel:
    st.header("Panel integrado: un solo escenario, tres modelos conectados")
    st.markdown(
        "**Ciclo del sistema:** precio × elasticidad → demanda · pronóstico (g) → consumo de insumos → "
        "SS, ROP, EOQ → **nivel de servicio real y ventas perdidas** → ingreso, costos y **margen** → "
        "**precio óptimo**. Todo se mide en el mismo horizonte "
        f"(**{horizonte} semanas**), en pizzas, porciones y US$."
    )
    if set(CATS) - set(cats_scope):
        st.caption(f"Alcance actual: {serie_sel} (categorías {', '.join(cats_scope)}). Los ajustes de precio de "
                   f"{', '.join(sorted(set(CATS) - set(cats_scope)))} no afectan este alcance; elija «{OPC_TODAS}» "
                   "para incluir todas las categorías.")

    st.markdown("##### 1 · Precio y demanda")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Δ precio global", f"{delta_global:+d} %")
    c2.metric("Elasticidad usada (media ponderada)", f"{sum(pes_u[c] * eps[c] for c in pes_u.index):.2f}")
    c3.metric("Multiplicador de demanda (precio)", f"×{R['m_u']:.3f}", f"{(R['m_u'] - 1) * 100:+.1f} %")
    c4.metric("Factor de pronóstico g", f"×{g_pron:.3f}", f"{(g_pron - 1) * 100:+.1f} % vs histórico",
              help="Pronóstico medio semanal ÷ media semanal histórica del alcance")

    st.markdown("##### 2 · Inventario y servicio")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(f"Demanda proyectada ({horizonte} sem.)", f"{R['dem']:,.0f} pizzas", pct(R["dem"], B["dem"]))
    c6.metric("Fill rate (demanda atendida)", f"{R['fill'] * 100:.2f} %",
              f"{R['perdidas']:,.1f} pizzas perdidas", delta_color="off",
              help="1 − faltante esperado por ciclo (σ·√L·G(z)) ÷ lote EOQ, ponderado por consumo de insumos")
    c7.metric(f"Costo de inventario ({horizonte} sem.)", f"US$ {R['c_inv']:,.0f}", pct(R["c_inv"], B["c_inv"]),
              delta_color="inverse", help=f"S/ {R['c_inv_s']:,.0f} · pedidos + mantener lote + stock de seguridad")
    c8.metric("Σ Puntos de reorden", f"{R['met']['ROP'].sum():,.0f} porc.",
              pct(R["met"]["ROP"].sum(), B["met"]["ROP"].sum()), delta_color="inverse")

    st.markdown("##### 3 · Resultado económico")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric(f"Ingreso ({horizonte} sem.)", f"US$ {R['ingreso']:,.0f}", pct(R["ingreso"], B["ingreso"]),
              help=f"≈ S/ {R['ingreso'] * tc:,.0f}")
    c10.metric("Costo de insumos", f"US$ {R['c_ins']:,.0f}", pct(R["c_ins"], B["c_ins"]), delta_color="inverse")
    c11.metric("Margen (ingreso − insumos − inventario)", f"US$ {R['margen']:,.0f}", pct(R["margen"], B["margen"]))
    c12.metric("Δ precio que maximiza el margen", f"{opt['d']:+.0f} %",
               f"margen US$ {opt['margen']:,.0f}", delta_color="off")
    if opt["d"] in (sens["d"].min(), sens["d"].max()):
        st.warning("El precio óptimo cae en el **borde** del rango (±30 %): con demanda inelástica el modelo siempre "
                   "sugiere subir. No incluye competencia ni caídas no lineales de la demanda; úselo como escenario.")

    cA, cB = st.columns([3, 2])
    with cA:
        st.subheader("Demanda semanal: histórico, base y escenario")
        hist = serie_sem["unidades"].tail(16).reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        ax.plot(hist["fecha"], hist["unidades"], marker="o", color=COLOR_BASE, label="Histórico")
        ax.plot(fc["fecha"], fc["prediccion"], marker="o", ls="--", color="#F4A300", label="Pronóstico base")
        ax.plot(fc["fecha"], fc["escenario"], marker="s", color=COLOR_ESC, label="Demanda escenario (precio)")
        ax.plot(fc["fecha"], fc["vendidas"], ls=":", color="#6B8E23", label="Vendidas (tras faltantes)")
        ax.axvline(serie_sem.index[-1], color="#6B8E23", ls=":")
        ax.set_ylabel("Pizzas por semana"); ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)
    with cB:
        st.subheader("Sensibilidad al precio (todo el sistema)")
        b = sens.loc[sens["d"] == 0].iloc[0]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for col, color, lab in [("ingreso", COLOR_ESC, "Ingreso"), ("dem", "#F4A300", "Demanda"),
                                ("c_inv", "#6B8E23", "Costo inventario"), ("margen", COLOR_BASE, "Margen")]:
            ax.plot(sens["d"], (sens[col] / b[col] - 1) * 100, marker="o", color=color, label=lab,
                    lw=2.5 if col == "margen" else 1.5)
        ax.axhline(0, color="gray", lw=0.8); ax.axvline(delta_global, color=COLOR_BASE, ls=":")
        ax.axvline(opt["d"], color=COLOR_ESC, ls="--", lw=1)
        ax.set_xlabel("Δ precio global (%)"); ax.set_ylabel("Variación vs Δ = 0 (%)")
        ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)

    st.subheader("Inventario: histórico, base y escenario")
    comp = pd.DataFrame({
        "Porciones por pizza": est["Porciones_por_pizza"],
        "Consumo semanal histórico": met_hist["Demanda_semanal"],
        "Consumo semanal base": met_base["Demanda_semanal"], "Consumo semanal escenario": met_esc["Demanda_semanal"],
        "ROP base": met_base["ROP"], "ROP escenario": met_esc["ROP"],
        "EOQ base": met_base["EOQ"], "EOQ escenario": met_esc["EOQ"],
        "Fill rate escenario (%)": (R["fill_i"] * 100).round(2),
        f"Costo escenario {horizonte} sem. (S/)": (met_esc["Costo_total_sem"] * horizonte).round(0),
        "Factor total (g·m)": factor_esc.round(3)})
    st.dataframe(comp)
    st.download_button("⬇️ Descargar comparación (CSV)", csv_bytes(comp), "comparacion_inventario.csv", "text/csv")

    with st.expander("🔗 Mapa de dependencias: qué mueve cada parámetro"):
        st.markdown(
            """
| Parámetro | Demanda | Inventario (SS/ROP/EOQ) | Servicio y ventas perdidas | Ingreso | Margen | Precio óptimo |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Δ precio (global / categoría) | ✔ | ✔ | ✔ | ✔ | ✔ | — |
| Elasticidad (modo / ε) | ✔ si Δ ≠ 0 | ✔ si Δ ≠ 0 | ✔ si Δ ≠ 0 | ✔ si Δ ≠ 0 | ✔ si Δ ≠ 0 | ✔ |
| Alcance, índice temporal, modelo, horizonte | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |
| N.º de insumos | — | ✔ | ✔ | ✔ | ✔ | ✔ |
| Lead time L, nivel de servicio | — | ✔ | ✔ | ✔ | ✔ | ✔ |
| Costos S y H | — | ✔ | ✔ | ✔ | ✔ | ✔ |
| Costo de insumos (%), tipo de cambio | — | — | — | — | ✔ | ✔ |
"""
        )
# ============================ INDICADORES CLAVE ==============================
with tab_ind:
    st.header("Indicadores clave del problema")
    st.write("Cada indicador es el resultado de una técnica analítica distinta y responde a una parte del problema: "
             "¿cuánto reacciona la demanda al precio?, ¿cuánto se venderá?, ¿cuánto y cuándo comprar?, "
             "¿qué servicio y qué margen se obtienen? Los valores corresponden al escenario actual del panel lateral.")

    ic = ols_g["tabla"].loc["ln_precio"]
    mape_u = bt_res[modelo_usado][2]
    resumen_ind = pd.DataFrame([
        {"Indicador": "Elasticidad precio-demanda (ε)", "Valor": f"{ols_g['eps']:.3f}",
         "Técnica analítica": "Regresión OLS log-log con controles de tamaño y categoría",
         "Lectura": "Inelástica (|ε| < 1): un alza de precio reduce poco las unidades" if abs(ols_g["eps"]) < 1
         else "Elástica (|ε| ≥ 1): un alza de precio reduce el ingreso"},
        {"Indicador": f"Demanda proyectada ({horizonte} sem.)", "Valor": f"{R['dem']:,.0f} pizzas",
         "Técnica analítica": f"Series de tiempo · {modelo_usado} (pronóstico recursivo)",
         "Lectura": f"Promedio de {R['dem'] / horizonte:,.0f} pizzas por semana"},
        {"Indicador": "Error del pronóstico (MAPE)", "Valor": f"{mape_u:.2f} %",
         "Técnica analítica": "Backtesting con origen móvil (4 semanas por corte)",
         "Lectura": "Margen de error esperado al planificar compras"},
        {"Indicador": "Punto de reorden total (Σ ROP)", "Valor": f"{R['met']['ROP'].sum():,.0f} porciones",
         "Técnica analítica": "Modelo de inventario de revisión continua (s, Q)",
         "Lectura": f"Incluye {R['met']['SS'].sum():,.0f} porciones de stock de seguridad"},
        {"Indicador": "Lote económico total (Σ EOQ)", "Valor": f"{R['met']['EOQ'].sum():,.0f} porciones",
         "Técnica analítica": "EOQ = √(2DS/H)",
         "Lectura": f"Pedido cada {R['met']['Semanas_entre_pedidos'].mean():.1f} semanas en promedio"},
        {"Indicador": "Fill rate (demanda atendida)", "Valor": f"{R['fill'] * 100:.2f} %",
         "Técnica analítica": "Función de pérdida normal G(z) sobre el inventario",
         "Lectura": f"{R['perdidas']:,.1f} pizzas perdidas por faltantes en el horizonte"},
        {"Indicador": f"Margen del escenario ({horizonte} sem.)", "Valor": f"US$ {R['margen']:,.0f}",
         "Técnica analítica": "Integración de los tres modelos (ingreso − insumos − inventario)",
         "Lectura": f"{(R['margen'] / B['margen'] - 1) * 100:+.1f} % frente a la base"},
    ]).set_index("Indicador")
    st.dataframe(resumen_ind, use_container_width=True)
    st.download_button("⬇️ Descargar indicadores (CSV)", csv_bytes(resumen_ind), "indicadores.csv", "text/csv")

    # --- Indicador 1: elasticidad
    st.subheader("1 · Elasticidad precio-demanda")
    i1, i2, i3 = st.columns(3)
    i1.metric("ε (OLS log-log)", f"{ols_g['eps']:.3f}")
    i2.metric("IC 95 %", f"[{ic['IC95_inf']:.2f} ; {ic['IC95_sup']:.2f}]")
    i3.metric("p-valor / R²", f"{ols_g['p']:.3f} / {ols_g['r2']:.3f}")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for c, col in zip(CATS, PALETA):
        sub = agg[agg["pizza_category"] == c]
        ax.scatter(sub["ln_precio"], sub["ln_cantidad"], color=col, label=c, alpha=0.8)
    xs = np.linspace(agg["ln_precio"].min(), agg["ln_precio"].max(), 20)
    ax.plot(xs, agg["ln_cantidad"].mean() + ols_g["eps"] * (xs - agg["ln_precio"].mean()),
            color=COLOR_BASE, lw=2, label=f"Pendiente ε = {ols_g['eps']:.2f}")
    ax.set_xlabel("ln(precio)"); ax.set_ylabel("ln(cantidad)"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)
    st.caption("La pendiente es la elasticidad estimada con controles de tamaño y categoría. "
               + ("No es significativa al 5 %: se usa como escenario referencial." if ols_g["p"] > 0.05 else ""))

    # --- Indicador 2: demanda y error
    st.subheader("2 · Demanda proyectada y error del pronóstico")
    j1, j2 = st.columns([3, 2])
    with j1:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        h16 = serie_sem["unidades"].tail(16).reset_index()
        ax.plot(h16["fecha"], h16["unidades"], marker="o", color=COLOR_BASE, label="Histórico")
        ax.plot(fc["fecha"], fc["prediccion"], marker="o", ls="--", color="#F4A300", label="Pronóstico")
        ax.fill_between(fc["fecha"], fc["prediccion"] * (1 - mape_u / 100), fc["prediccion"] * (1 + mape_u / 100),
                        color="#F4A300", alpha=0.2, label=f"± MAPE ({mape_u:.1f} %)")
        ax.set_ylabel("Pizzas por semana"); ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)
    with j2:
        mapes = pd.Series({n: r[2] for n, r in bt_res.items()}).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.barh(mapes.index[::-1], mapes.values[::-1],
                color=[COLOR_ESC if n == modelo_usado else "#BBBBBB" for n in mapes.index[::-1]])
        for k, v in enumerate(mapes.values[::-1]):
            ax.text(v, k, f" {v:.2f} %", va="center", fontsize=8)
        ax.set_xlabel("MAPE (%)"); ax.set_title("Error por modelo (backtesting)", fontsize=10); estilo(ax)
        mostrar(fig)

    # --- Indicador 3: inventario
    st.subheader("3 · Política de inventario por insumo (ROP, SS y EOQ)")
    k1, k2 = st.columns(2)
    with k1:
        mi = R["met"].sort_values("ROP")
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.barh(mi.index, mi["ROP"] - mi["SS"], color="#F4A300", label="Demanda en el lead time (μ·L)")
        ax.barh(mi.index, mi["SS"], left=mi["ROP"] - mi["SS"], color=COLOR_ESC, label="Stock de seguridad")
        ax.set_xlabel("Porciones"); ax.set_title("Punto de reorden = μ·L + SS", fontsize=10)
        ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)
    with k2:
        me = R["met"].sort_values("EOQ")
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.barh(me.index, me["EOQ"], color="#2b83ba")
        ax.set_xlabel("Porciones por pedido"); ax.set_title("Lote económico (EOQ)", fontsize=10); estilo(ax)
        mostrar(fig)

    # --- Indicador 4: servicio vs costo
    st.subheader("4 · Nivel de servicio: fill rate vs costo de inventario")
    curva = pd.DataFrame([{"Servicio": n, "Z": z, **{k: v for k, v in evaluar(d_efectivos, z).items()
                                                      if k in ("fill", "c_inv", "margen", "perdidas")}}
                          for n, z in Z_TABLA.items()])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(curva["Servicio"], curva["c_inv"], color="#6B8E23", alpha=0.7, label="Costo de inventario (US$)")
    ax.set_ylabel(f"Costo inventario {horizonte} sem. (US$)")
    ax2 = ax.twinx()
    ax2.plot(curva["Servicio"], curva["fill"] * 100, marker="o", color=COLOR_ESC, label="Fill rate (%)")
    ax2.set_ylabel("Fill rate (%)")
    ax.set_title("Más servicio = más stock de seguridad (costo) y menos ventas perdidas", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left"); estilo(ax)
    mostrar(fig)

    # --- Indicador 5: margen
    st.subheader(f"5 · Composición del margen ({horizonte} semanas, US$)")
    partes = pd.Series({"Ingreso": R["ingreso"], "Costo de insumos": -R["c_ins"],
                        "Costo de inventario": -R["c_inv"], "Margen": R["margen"]})
    fig, ax = plt.subplots(figsize=(9, 3.4))
    acum, x = 0.0, 0
    for nombre, v in partes.items():
        if nombre == "Margen":
            ax.bar(x, v, color=COLOR_BASE); ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.bar(x, v, bottom=acum if v < 0 else 0, color=COLOR_ESC if v < 0 else "#F4A300")
            ax.text(x, acum if v < 0 else v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
            acum += v
        x += 1
    ax.set_xticks(range(len(partes))); ax.set_xticklabels(partes.index); ax.set_ylabel("US$"); estilo(ax)
    mostrar(fig)

# ========================= ESCENARIOS PRESCRIPTIVOS ==========================
with tab_esc:
    st.header("Escenarios prescriptivos: ¿qué decisión conviene tomar?")
    st.write("Se evalúan varias decisiones de precio y de nivel de servicio con el **mismo sistema integrado** "
             "(elasticidad → demanda → inventario → margen). Se recomienda el escenario de **mayor margen** que "
             "cumpla las restricciones operativas. Puede editar, agregar o quitar escenarios en la tabla.")

    cat_principal = pes_u.idxmax()
    nivel_actual = nivel
    esc_default = pd.DataFrame([
        {"Escenario": "E0 · Base (situación actual)", "Δ precio global (%)": 0, "Categoría con ajuste": "—",
         "Ajuste categoría (pp)": 0, "Nivel de servicio": nivel_actual},
        {"Escenario": "E1 · Alza de precio 10 %", "Δ precio global (%)": 10, "Categoría con ajuste": "—",
         "Ajuste categoría (pp)": 0, "Nivel de servicio": nivel_actual},
        {"Escenario": "E2 · Promoción −10 %", "Δ precio global (%)": -10, "Categoría con ajuste": "—",
         "Ajuste categoría (pp)": 0, "Nivel de servicio": nivel_actual},
        {"Escenario": "E3 · Alza 10 % + servicio 99 %", "Δ precio global (%)": 10, "Categoría con ajuste": "—",
         "Ajuste categoría (pp)": 0, "Nivel de servicio": "99 %"},
        {"Escenario": f"E4 · Alza 10 % solo en {cat_principal}", "Δ precio global (%)": 0,
         "Categoría con ajuste": cat_principal, "Ajuste categoría (pp)": 10, "Nivel de servicio": nivel_actual},
        {"Escenario": "E5 · Ahorro: servicio 90 %", "Δ precio global (%)": 0, "Categoría con ajuste": "—",
         "Ajuste categoría (pp)": 0, "Nivel de servicio": "90 %"},
    ])
    esc_in = st.data_editor(
        esc_default, num_rows="dynamic", use_container_width=True, hide_index=True, key="editor_escenarios",
        column_config={
            "Δ precio global (%)": st.column_config.NumberColumn(min_value=-30, max_value=30, step=1),
            "Categoría con ajuste": st.column_config.SelectboxColumn(options=["—"] + CATS),
            "Ajuste categoría (pp)": st.column_config.NumberColumn(min_value=-30, max_value=30, step=1),
            "Nivel de servicio": st.column_config.SelectboxColumn(options=list(Z_TABLA.keys())),
        })

    r1, r2 = st.columns(2)
    fill_min = r1.slider("Restricción: fill rate mínimo (%)", 95.0, 100.0, 99.90, 0.01)
    caida_max = r2.slider("Restricción: caída máxima de demanda vs E0 (%)", 0, 30, 10, 1)

    filas = []
    for _, e in esc_in.dropna(subset=["Escenario"]).iterrows():
        dg = float(e.get("Δ precio global (%)") or 0)
        cat_e, aj = e.get("Categoría con ajuste") or "—", float(e.get("Ajuste categoría (pp)") or 0)
        dd = {c: float(np.clip(dg + (aj if c == cat_e else 0), -90, 200)) for c in CATS}
        zz = Z_TABLA.get(e.get("Nivel de servicio"), Z)
        r = evaluar(dd, zz)
        filas.append({"Escenario": e["Escenario"], "Demanda (pizzas)": r["dem"], "Fill rate (%)": r["fill"] * 100,
                      "Pizzas perdidas": r["perdidas"], "Ingreso (US$)": r["ingreso"],
                      "Costo insumos (US$)": r["c_ins"], "Costo inventario (US$)": r["c_inv"],
                      "Margen (US$)": r["margen"], "Σ ROP (porc.)": r["met"]["ROP"].sum(),
                      "Σ SS (porc.)": r["met"]["SS"].sum(), "Σ EOQ (porc.)": r["met"]["EOQ"].sum()})
    tabla_esc = pd.DataFrame(filas).set_index("Escenario")

    if tabla_esc.empty:
        st.warning("Agregue al menos un escenario en la tabla.")
    else:
        dem_ref = tabla_esc["Demanda (pizzas)"].iloc[0]
        tabla_esc["Δ demanda vs E0 (%)"] = (tabla_esc["Demanda (pizzas)"] / dem_ref - 1) * 100
        tabla_esc["Δ margen vs E0 (%)"] = (tabla_esc["Margen (US$)"] / tabla_esc["Margen (US$)"].iloc[0] - 1) * 100
        tabla_esc["Cumple restricciones"] = ((tabla_esc["Fill rate (%)"] >= fill_min)
                                             & (tabla_esc["Δ demanda vs E0 (%)"] >= -caida_max))
        factibles = tabla_esc[tabla_esc["Cumple restricciones"]]
        reco = factibles["Margen (US$)"].idxmax() if not factibles.empty else None

        st.subheader("Comparación de escenarios")
        st.dataframe(tabla_esc.style.format({
            "Demanda (pizzas)": "{:,.0f}", "Fill rate (%)": "{:.3f}", "Pizzas perdidas": "{:,.1f}",
            "Ingreso (US$)": "{:,.0f}", "Costo insumos (US$)": "{:,.0f}", "Costo inventario (US$)": "{:,.0f}",
            "Margen (US$)": "{:,.0f}", "Σ ROP (porc.)": "{:,.0f}", "Σ SS (porc.)": "{:,.0f}",
            "Σ EOQ (porc.)": "{:,.0f}", "Δ demanda vs E0 (%)": "{:+.1f}", "Δ margen vs E0 (%)": "{:+.1f}"})
            .apply(lambda f: ["background-color: #E7F0D6" if f.name == reco else "" for _ in f], axis=1),
            use_container_width=True)
        st.download_button("⬇️ Descargar escenarios (CSV)", csv_bytes(tabla_esc), "escenarios.csv", "text/csv")

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            colores = [COLOR_BASE if n == reco else ("#F4A300" if ok else "#BBBBBB")
                       for n, ok in tabla_esc["Cumple restricciones"].items()]
            ax.barh(tabla_esc.index[::-1], tabla_esc["Margen (US$)"][::-1], color=colores[::-1])
            ax.set_xlabel(f"Margen {horizonte} semanas (US$)")
            ax.set_title("Margen por escenario (gris = no cumple restricciones)", fontsize=10); estilo(ax)
            mostrar(fig)
        with g2:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            ax.scatter(tabla_esc["Δ demanda vs E0 (%)"], tabla_esc["Margen (US$)"], s=80,
                       c=colores, edgecolor=COLOR_BASE)
            etiquetas = {}
            for n, rr in tabla_esc.iterrows():  # agrupa escenarios que caen en el mismo punto
                pos = (round(rr["Δ demanda vs E0 (%)"], 1), round(rr["Margen (US$)"] / 50) * 50)
                etiquetas.setdefault(pos, []).append(n.split(" · ")[0])
            for (xx, yy), labs in etiquetas.items():
                ax.annotate(" / ".join(labs), (xx, yy), xytext=(5, 4), textcoords="offset points", fontsize=8)
            ax.axvline(-caida_max, color=COLOR_ESC, ls="--", lw=1, label="Caída máxima permitida")
            ax.set_xlabel("Δ demanda vs E0 (%)"); ax.set_ylabel("Margen (US$)")
            ax.set_title("Trade-off: volumen vs margen", fontsize=10); ax.legend(fontsize=8); estilo(ax)
            mostrar(fig)

        st.subheader("Recomendación")
        if reco is None:
            st.error("Ningún escenario cumple las restricciones. Relaje el fill rate mínimo o la caída máxima de demanda.")
        else:
            rr, r0 = tabla_esc.loc[reco], tabla_esc.iloc[0]
            st.success(
                f"**Escenario recomendado: {reco}.** Margen de **US$ {rr['Margen (US$)']:,.0f}** en {horizonte} semanas "
                f"({rr['Δ margen vs E0 (%)']:+.1f} % vs E0), con demanda de {rr['Demanda (pizzas)']:,.0f} pizzas "
                f"({rr['Δ demanda vs E0 (%)']:+.1f} %) y fill rate de {rr['Fill rate (%)']:.2f} %.")
            st.markdown(
                f"- **Compras:** reordenar cuando el stock llegue a **{rr['Σ ROP (porc.)']:,.0f} porciones** en total "
                f"(con {rr['Σ SS (porc.)']:,.0f} de seguridad) y pedir lotes que suman **{rr['Σ EOQ (porc.)']:,.0f} "
                "porciones**; el detalle por insumo está en la pestaña Inventario al fijar este escenario en el panel.\n"
                f"- **Frente a la base:** ingreso {rr['Ingreso (US$)'] - r0['Ingreso (US$)']:+,.0f} US$, costo de "
                f"inventario {rr['Costo inventario (US$)'] - r0['Costo inventario (US$)']:+,.0f} US$ y "
                f"{rr['Pizzas perdidas'] - r0['Pizzas perdidas']:+,.1f} pizzas perdidas.\n"
                f"- **Descartados por restricciones:** "
                + (", ".join(tabla_esc.index[~tabla_esc["Cumple restricciones"]]) or "ninguno") + ".")
        st.caption(f"Elasticidad usada: {modo} (panel lateral). La recomendación es un escenario, no una predicción: "
                   "depende de la elasticidad estimada (no significativa) y de los supuestos de costos.")

# ============================== DEMANDA ======================================
with tab_dem:
    st.header(f"Proyección de demanda con series de tiempo — {serie_sel}")
    tv = ts_v
    bt_v = tv["comp"]
    val_v = [n for n in MODELOS_TS if np.isfinite(bt_v.loc[n, "MAPE (%)"])]
    modelo_v = min(val_v, key=lambda n: bt_v.loc[n, "MAPE (%)"]) if modelo_sel == M_AUTO else modelo_sel
    es_diario = indice_sel == IDX_DIA
    unidad_t = "día" if es_diario else "semana"
    st.write(f"Variable proyectada: **{var_label}** · índice temporal: **{indice_sel}** · modelo: **{modelo_v}**"
             + (" (elegido automáticamente por menor MAPE)" if modelo_sel == M_AUTO else "")
             + f" · horizonte: **{horizonte} semanas**.")
    if serie_sel == OPC_TOP5:
        st.write("Top 5 por unidades: " + ", ".join(f"**{p}** ({int(u):,})" for p, u in top5.items()))
    st.info(f"Preparación: serie continua de {len(tv['y'])} periodos ({unidad_t}). Se completaron por interpolación "
            f"{tv['n_faltantes']} día(s) sin ventas y se descartaron {tv['sobrantes']} día(s) iniciales para formar "
            "semanas completas de 7 días. Así no hay huecos ni semanas incompletas.")

    # 1 · Serie y descriptivos
    st.subheader("1 · Serie de tiempo y análisis descriptivo")
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(tv["y"].index, tv["y"].values, color=COLOR_BASE, lw=1, label=f"{var_label} (real)")
    if modelo_v in tv["ajustados"]:
        aj = tv["ajustados"][modelo_v].iloc[tv["m"]:]
        ax.plot(aj.index, aj.values, color="#F4A300", lw=1, alpha=0.9, label=f"Ajuste {modelo_v}")
    ax.set_ylabel(f"{var_label} por {unidad_t}"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)
    dd1, dd2 = st.columns(2)
    with dd1:
        st.markdown("**Variables numéricas** (media, moda, desviación estándar, mínimo, máximo)")
        base_d = construir_series(d_scope, clave_s)[0].add_suffix(" / día")
        base_s = tv["semanal"].add_suffix(" / semana")
        st.dataframe(pd.concat([descriptivos(base_d, base_d.columns), descriptivos(base_s, base_s.columns)]))
    with dd2:
        st.markdown("**Variables categóricas** (moda y su frecuencia)")
        st.dataframe(pd.DataFrame([{"Variable": c, "Moda": d_scope[c].mode().iloc[0],
                                    "Frecuencia de la moda (%)": round((d_scope[c] == d_scope[c].mode().iloc[0]).mean() * 100, 1),
                                    "N.º categorías": d_scope[c].nunique()}
                                   for c in ["pizza_name", "pizza_category", "pizza_size"]]).set_index("Variable"))

    # 2 · Descomposición
    st.subheader(f"2 · Descomposición aditiva (periodo m = {tv['m']})")
    deco = tv["deco"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
    for ax, comp, tit, col in [(axes[0], deco.trend, "Tendencia", COLOR_BASE),
                               (axes[1], deco.seasonal.iloc[:4 * tv["m"]], "Estacionalidad (4 ciclos)", "#F4A300"),
                               (axes[2], deco.resid, "Residuo", "#6B8E23")]:
        ax.plot(comp.index, comp.values, color=col, lw=1.2); ax.set_title(tit, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=30); estilo(ax)
    mostrar(fig)
    amp = float(deco.seasonal.max() - deco.seasonal.min())
    if es_diario:
        perfil = deco.seasonal.groupby(deco.seasonal.index.dayofweek).mean()
        dias_n = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        st.caption(f"Amplitud estacional ≈ {amp:,.1f} por día. Día más alto: **{dias_n[int(perfil.idxmax())]}** "
                   f"({perfil.max():+.1f}); más bajo: **{dias_n[int(perfil.idxmin())]}** ({perfil.min():+.1f}) "
                   "respecto a la tendencia.")
    else:
        st.caption(f"Amplitud del ciclo de 4 semanas ≈ {amp:,.1f}. Con un solo año de datos no se puede estimar la "
                   "estacionalidad anual (se necesitan al menos dos ciclos de 52 semanas).")

    # 3 · Estacionariedad y autocorrelación
    st.subheader("3 · Estacionariedad y autocorrelación")
    e1, e2, e3 = st.columns([1, 2, 2])
    e1.metric("ADF p-valor", f"{tv['adf_p']:.4f}")
    e1.caption("Estacionaria al 5 %" if tv["adf_p"] < 0.05 else "No estacionaria: conviene diferenciar (d ≥ 1)")
    lim = 1.96 / np.sqrt(len(tv["y"]))
    for colw, vals, tit in [(e2, tv["acf"][1:], "ACF"), (e3, tv["pacf"][1:], "PACF")]:
        with colw:
            fig, ax = plt.subplots(figsize=(6, 2.8))
            ax.bar(range(1, len(vals) + 1), vals, color="#F4A300")
            ax.axhline(lim, color="gray", ls="--", lw=1); ax.axhline(-lim, color="gray", ls="--", lw=1)
            ax.set_title(f"{tit} (rezagos en {unidad_t}s)", fontsize=10); estilo(ax)
            mostrar(fig)

    # 4 · Modelos
    st.subheader("4 · Modelos de series de tiempo ajustados")
    o = tv["ordenes"]
    st.dataframe(pd.DataFrame([
        {"Modelo": M_HW, "Especificación": f"Tendencia aditiva amortiguada + estacionalidad aditiva (m = {tv['m']})",
         "Selección": "Parámetros α, β, γ, φ por máxima verosimilitud"},
        {"Modelo": M_ARIMA, "Especificación": f"ARIMA{o['arima']}",
         "Selección": f"Menor AIC en rejilla p,q ∈ {{0,1,2}}, d ∈ {{0,1}} (AIC = {o['aic_arima']:,.1f})"},
        {"Modelo": M_SARIMA, "Especificación": f"SARIMA{o['sarima'][0]}×{o['sarima'][1][:3]}{o['sarima'][1][3]}",
         "Selección": f"Menor AIC sobre la parte estacional (AIC = {o['aic_sarima']:,.1f})"},
    ]).set_index("Modelo"), use_container_width=True)
    st.caption("Los órdenes se eligen solo con el primer tramo de entrenamiento (28 semanas), para no usar "
               "información del periodo de prueba.")

    # 5 · Validación
    st.subheader("5 · Validación: backtesting con origen móvil")
    st.caption("Se entrena con al menos 28 semanas, se pronostican las 4 siguientes y se avanza 4 semanas; el error "
               "se mide siempre en totales semanales, así los índices diario y semanal son comparables.")
    st.dataframe(bt_v.round(2), use_container_width=True)
    mejor_v = bt_v["MAPE (%)"].idxmin()
    st.success(f"Menor MAPE: **{mejor_v}** ({bt_v.loc[mejor_v, 'MAPE (%)']:.2f} %). "
               f"Modelo usado: **{modelo_v}** ({bt_v.loc[modelo_v, 'MAPE (%)']:.2f} %).")
    if bt_v.loc[[M_NAIVE, M_MA4], "MAPE (%)"].min() < bt_v.loc[modelo_v, "MAPE (%)"]:
        st.warning("Una referencia simple tiene menor error que el modelo usado: la serie tiene poca estructura "
                   "semanal-a-semanal que explotar con un solo año de datos.")
    tb = tv["tablas"][modelo_v]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.plot(tb["fecha"], tb["real"], marker="o", color=COLOR_ESC, label="Real (semanal)")
    ax.plot(tb["fecha"], tb["prediccion"], marker="o", color="#F4A300", label=f"Predicción — {modelo_v}")
    ax.set_ylabel(f"{var_label} por semana"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)

    # 6 · Pronóstico
    st.subheader(f"6 · Pronóstico a {horizonte} semanas — {var_label}")
    pv = tv["pron"][MODELOS_TS].round(1)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    hs = tv["semanal"][var_sel].tail(16)
    ax.plot(hs.index, hs.values, marker="o", color=COLOR_BASE, label="Histórico (semanal)")
    for n, col in zip(MODELOS_TS, ["#6B8E23", "#2b83ba", COLOR_ESC]):
        ax.plot(pv.index, pv[n], marker="o", ls="--", color=col, lw=2.5 if n == modelo_v else 1, label=n)
    ax.axvline(hs.index[-1], color="gray", ls=":")
    ax.set_ylabel(f"{var_label} por semana"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)
    tabla_pv = pv.copy(); tabla_pv.index = tabla_pv.index.strftime("%d/%m/%Y"); tabla_pv.index.name = "Semana (fin)"
    st.dataframe(tabla_pv, use_container_width=True)
    st.download_button("⬇️ Descargar pronósticos (CSV)", csv_bytes(tabla_pv), "pronosticos.csv", "text/csv")

    st.markdown(f"**Pronóstico que usa el sistema integrado (unidades, {modelo_usado}) y escenario de precio:**")
    tabla_fc = fc.rename(columns={"fecha": "Semana (fin)", "prediccion": "Unidades (base)",
                                  "escenario": "Unidades (escenario precio)", "vendidas": "Vendidas (tras faltantes)",
                                  "ingreso_base": "Ingreso base (US$)", "ingreso_escenario": "Ingreso escenario (US$)"})
    tabla_fc["Semana (fin)"] = tabla_fc["Semana (fin)"].dt.strftime("%d/%m/%Y")
    st.dataframe(tabla_fc.set_index("Semana (fin)").round(1), use_container_width=True)
    st.caption("El efecto de un cambio de precio entra por la elasticidad (columna «escenario»), no por el modelo "
               "de series de tiempo, para no contarlo dos veces.")

# ============================== ELASTICIDAD ==================================
with tab_el:
    st.header("Elasticidad precio-demanda (regresión log-log)")
    st.subheader("1 · Análisis de ventas")
    st.dataframe(descriptivos(df, ["unit_price", "quantity", "total_price"]))
    cc1, cc2 = st.columns(2)
    cc1.write("**Categoría (líneas de pedido)**"); cc1.dataframe(df["pizza_category"].value_counts().rename("líneas"))
    cc2.write("**Tamaño (líneas de pedido)**"); cc2.dataframe(df["pizza_size"].value_counts().rename("líneas"))
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for ax, v in zip(axes[:3], ["unit_price", "quantity", "total_price"]):
        ax.hist(df[v], bins=20, color="#F4A300", edgecolor=COLOR_BASE); ax.set_title(f"Histograma — {v}", fontsize=9); estilo(ax)
    for ax, v in zip(axes[3:], ["unit_price", "total_price"]):
        bp = ax.boxplot(df[v], patch_artist=True); bp["boxes"][0].set_facecolor(COLOR_ESC)
        ax.set_title(f"Caja — {v}", fontsize=9); estilo(ax)
    mostrar(fig)

    st.subheader("2 · Modelo global OLS: ln(cantidad) ~ ln(precio) + tamaño + categoría")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Elasticidad global ε", f"{ols_g['eps']:.3f}")
    m2.metric("p-valor de ε", f"{ols_g['p']:.4f}")
    m3.metric("R² / R² ajustado", f"{ols_g['r2']:.3f} / {ols_g['r2_aj']:.3f}")
    m4.metric("Variantes (n)", f"{ols_g['n']}")
    st.dataframe(ols_g["tabla"].round(4))
    if ols_g["p"] > 0.05:
        st.warning("ε global **no es significativa al 5 %**: el precio, por sí solo, explica poco de la variación entre "
                   "variantes. Úsela como escenario referencial, no como estimación definitiva.")
    st.write("Como |ε| < 1, la demanda es **inelástica**: un alza de precio aumenta el ingreso." if abs(ols_g["eps"]) < 1
             else "Como |ε| ≥ 1, la demanda es **elástica**: un alza de precio reduce el ingreso.")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for c, col in zip(CATS, PALETA):
        sub = agg[agg["pizza_category"] == c]
        ax.scatter(sub["ln_precio"], sub["ln_cantidad"], color=col, label=c, alpha=0.8)
    ax.set_xlabel("ln(precio promedio)"); ax.set_ylabel("ln(cantidad total)"); ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)

    st.subheader("3 · Elasticidad por segmento")
    e1, e2 = st.columns(2)
    e1.write("**Por categoría** (OLS con control de tamaño)"); e1.dataframe(tabla_cat.round(4))
    e2.write("**Por tamaño** (OLS con control de categoría)"); e2.dataframe(tabla_tam.round(4))
    st.caption("«No identificable»: dentro del segmento el precio queda determinado por el tamaño (p. ej., todas las "
               "Chicken cuestan lo mismo para un tamaño dado), por lo que no hay variación para estimar ε. "
               "Los segmentos con pocas variantes producen estimaciones inestables.")
    if modo.startswith("Por categoría"):
        st.warning("Está usando elasticidades por categoría. Con 18–25 variantes por categoría pueden resultar muy "
                   "dispares; verifique su significancia antes de decidir.")
    if respaldo:
        st.info("Se usó la elasticidad global para: " + ", ".join(respaldo))

    st.subheader("4 · Benchmark: OLS vs Ridge vs Random Forest")
    bench, n_cel = benchmark_elasticidad(df, clave)
    st.dataframe(bench.round(3))
    st.caption(f"Validación cruzada de 5 folds sobre {n_cel:,} celdas (variante × día de la semana × mes). "
               "ε del Random Forest = elasticidad implícita al simular ±10 % de precio.")

    st.subheader("5 · Simulador de precios (con la elasticidad del panel lateral)")
    f1, f2 = st.columns(2)
    seg_cat = f1.selectbox("Categoría", ["Todas"] + CATS)
    seg_tam = f2.selectbox("Tamaño", ["Todos"] + sorted(df["pizza_size"].unique()))
    df_seg = df.copy()
    if seg_cat != "Todas":
        df_seg = df_seg[df_seg["pizza_category"] == seg_cat]
    if seg_tam != "Todos":
        df_seg = df_seg[df_seg["pizza_size"] == seg_tam]
    if df_seg.empty:
        st.warning("El segmento seleccionado no tiene ventas.")
    else:
        sim = simular_precio(df_seg, eps, list(range(-30, 31, 5)))
        opt_sim = sim.loc[sim["Ingreso ($)"].idxmax()]
        k1, k2, k3 = st.columns(3)
        k1.metric("Ingreso base", f"${df_seg['total_price'].sum():,.0f}")
        k2.metric("Δ precio que maximiza el ingreso", f"{opt_sim['Δ precio (%)']:+.0f} %")
        k3.metric("Ingreso en el óptimo", f"${opt_sim['Ingreso ($)']:,.0f}", f"{opt_sim['Δ ingreso (%)']:+.1f} %")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(sim["Δ precio (%)"], sim["Ingreso ($)"], marker="o", color=COLOR_ESC, label="Ingreso")
        ax.axvline(delta_global, color=COLOR_BASE, ls=":", label="Escenario del panel")
        ax.set_xlabel("Δ precio (%)"); ax.set_ylabel("Ingreso ($)"); ax.legend(); estilo(ax)
        mostrar(fig)
        st.dataframe(sim.round(2).set_index("Δ precio (%)"))
        if opt_sim["Δ precio (%)"] in (sim["Δ precio (%)"].min(), sim["Δ precio (%)"].max()):
            st.warning("El óptimo cae en el **borde** del rango simulado: con |ε| < 1 el ingreso crece con el precio. "
                       "El modelo no incluye costos, competencia ni caídas no lineales de la demanda; no lo extrapole.")

        st.subheader("6 · Simulador por historia (solo precios observados)")
        hist_p = (df_seg.assign(v=df_seg["pizza_name"] + "|" + df_seg["pizza_size"]).groupby("unit_price")
                  .agg(Unidades=("quantity", "sum"), Ingreso=("total_price", "sum"), Variantes=("v", "nunique")))
        hist_p["Ingreso por variante"] = hist_p["Ingreso"] / hist_p["Variantes"]
        st.dataframe(hist_p.round(2))
        st.success(f"Precio histórico con mayor ingreso total: **${hist_p['Ingreso'].idxmax():.2f}** · "
                   f"con mayor ingreso por variante: **${hist_p['Ingreso por variante'].idxmax():.2f}**.")

# ============================== INVENTARIO ===================================
with tab_inv:
    st.header("Gestión de inventarios — revisión continua (s, Q)")
    st.write(f"Alcance: **{serie_sel}** (los mismos productos del pronóstico). Todo el inventario se expresa en la "
             "**misma base que la demanda: semanas**. El consumo de insumos sale de la demanda en pizzas mediante "
             "las porciones que cada pizza usa de cada insumo.")

    st.subheader("Conversión de unidades: demanda → inventario")
    st.dataframe(pd.DataFrame([
        {"Magnitud": "Demanda", "Unidad": "pizzas / semana", "Valor": f"{fc['prediccion'].mean():,.1f}",
         "Origen": f"Pronóstico de series de tiempo ({modelo_usado})"},
        {"Magnitud": "Consumo del insumo i", "Unidad": "porciones / semana",
         "Valor": "pizzas/semana × porciones por pizza × m_i", "Origen": "Receta histórica (pizza_ingredients) y elasticidad"},
        {"Magnitud": "Lead time L", "Unidad": "semanas", "Valor": f"{L} días ÷ 7 = {L / 7:.3f}", "Origen": "Panel lateral"},
        {"Magnitud": "Costo de mantener H", "Unidad": "S/ por porción·semana",
         "Valor": f"{H_men:,.2f} ÷ {sem_anio} = {H_men / sem_anio:.4f}", "Origen": "Panel lateral (H anual)"},
        {"Magnitud": "Costo por pedido S", "Unidad": "S/ por pedido", "Valor": f"{S_ped:,.2f}", "Origen": "Panel lateral"},
        {"Magnitud": "Costo de inventario", "Unidad": f"S/ por semana × {horizonte} semanas → US$",
         "Valor": f"S/ {R['c_inv_s']:,.0f} ÷ {tc:.2f} = US$ {R['c_inv']:,.0f}", "Origen": "Tipo de cambio del panel"},
    ]).set_index("Magnitud"), use_container_width=True)

    st.subheader("Consumo histórico de insumos (porciones por semana)")
    st.dataframe(est, use_container_width=True)
    st.caption("Porciones_por_pizza = consumo semanal medio del insumo ÷ pizzas semanales medias del alcance.")

    st.subheader("Reducción de dimensiones: PCA del consumo de insumos")
    evr, cargas = pca_insumos(consumo)
    n95 = int(np.searchsorted(evr.cumsum(), 0.95) + 1)
    q1, q2 = st.columns([3, 2])
    with q1:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(range(1, len(evr) + 1), evr * 100, color="#6B8E23", label="Varianza por componente")
        ax.plot(range(1, len(evr) + 1), evr.cumsum() * 100, marker="o", color=COLOR_ESC, label="Acumulada")
        ax.axhline(95, color="gray", ls="--", lw=1)
        ax.set_xlabel("Componente principal"); ax.set_ylabel("% de varianza"); ax.legend(fontsize=8); estilo(ax)
        mostrar(fig)
    with q2:
        st.metric("Componentes para ≥ 95 % de la varianza", f"{n95} de {len(evr)}")
        st.metric("Varianza explicada por CP1", f"{evr[0] * 100:.1f} %")
        st.dataframe(cargas, use_container_width=True)
    st.caption(f"Los {len(evr)} insumos se resumen en {n95} componentes: su consumo no es independiente, lo determinan "
               f"las pizzas vendidas ({d_scope['pizza_name'].nunique()} productos en el alcance). Esto respalda calcular "
               "el consumo de insumos a partir de la demanda de pizzas (porciones por pizza).")

    st.subheader("Métricas de inventario — escenario actual (por semana)")
    out = met_esc.copy()
    out.insert(0, "Factor (g·m)", factor_esc.round(3))
    out["Faltante_por_ciclo"] = (met_esc["Desv_semanal"] * np.sqrt(L / 7.0) * G_Z).round(2)
    out["Fill_rate_%"] = (R["fill_i"] * 100).round(3)
    out[f"Costo_{horizonte}_sem"] = (met_esc["Costo_total_sem"] * horizonte).round(0)
    st.dataframe(out, use_container_width=True)
    st.download_button("⬇️ Descargar métricas (CSV)", csv_bytes(out), "metricas_inventario.csv", "text/csv")
    st.caption("μ, σ en porciones/semana · L en semanas · H en S/ por porción·semana · "
               "SS = Z·σ·√L · ROP = μ·L + SS · EOQ = √(2·μ·S/H) · Costo_EOQ_sem = √(2·μ·S·H) · "
               "Costo_total_sem = Costo_EOQ_sem + H·SS · Faltante por ciclo = σ·√L·G(z) · Fill rate = 1 − faltante ÷ EOQ. "
               "El EOQ no cambia si se calcula con datos anuales o semanales, siempre que D y H usen el mismo periodo.")

    g1, g2 = st.columns(2)
    for ax_col, campo, color, titulo in [(g1, "ROP", "#e66101", "Punto de reorden (ROP, porciones)"),
                                         (g2, "EOQ", "#2b83ba", "Lote económico (EOQ, porciones)")]:
        with ax_col:
            idx = np.arange(len(met_esc))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(idx + 0.2, met_base[campo].values, 0.4, color="#BBBBBB", label="Base (pronóstico, precio actual)")
            ax.barh(idx - 0.2, met_esc[campo].values, 0.4, color=color, label="Escenario")
            ax.set_yticks(idx); ax.set_yticklabels(met_esc.index, fontsize=8); ax.invert_yaxis()
            ax.set_title(titulo); ax.legend(fontsize=8); estilo(ax)
            mostrar(fig)

    st.subheader("Ciclo de inventario simulado")
    ins = st.selectbox("Insumo", met_esc.index.tolist())
    r = met_esc.loc[ins]
    x_sem, ciclo = simular_ciclo(r["Demanda_semanal"], r["ROP"], r["EOQ"], r["SS"], L, semanas=horizonte)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(x_sem, ciclo, color=COLOR_ESC, label="Stock")
    ax.axhline(r["ROP"], color="#F4A300", ls="--", label=f"ROP = {r['ROP']:.0f}")
    ax.axhline(r["SS"], color="#6B8E23", ls=":", label=f"SS = {r['SS']:.0f}")
    ax.set_xlabel(f"Semanas del horizonte (1–{horizonte})"); ax.set_ylabel("Porciones")
    ax.legend(fontsize=8); estilo(ax)
    mostrar(fig)
    st.caption(f"Consumo de {r['Demanda_semanal']:,.1f} porciones/semana; se pide un lote de {r['EOQ']:,.0f} "
               f"porciones cada {r['Semanas_entre_pedidos']:.1f} semanas.")

    st.subheader(f"Requerimiento semanal proyectado por insumo ({horizonte} semanas)")
    pizzas_t = fc["prediccion"].values if aplicar_pron else np.full(horizonte, media_hist)
    plan = pd.DataFrame(np.ceil(np.outer(est["Porciones_por_pizza"].values * m_insumo.values, pizzas_t)),
                        index=est.index, columns=fc["fecha"].dt.strftime("%d/%m/%Y")).astype(int)
    st.dataframe(plan, use_container_width=True)
    st.download_button("⬇️ Descargar plan (CSV)", csv_bytes(plan), "plan_requerimiento.csv", "text/csv")
    st.caption("Requerimiento (porciones/semana) = pizzas pronosticadas en la semana × porciones por pizza × "
               "multiplicador de precio del insumo.")

# ============================== METODOLOGÍA ==================================
with tab_met:
    st.header("Metodología, supuestos y limitaciones")
    st.markdown(
        """
**Datos.** 48 620 líneas de pedido de 2015 (fecha en formato día/mes/año).

**Elasticidad.** OLS log-log sobre variantes (pizza × categoría × tamaño):
`ln(cantidad) = β₀ + ε·ln(precio) + tamaño + categoría`. Benchmarks: Ridge y Random Forest.

**Demanda (series de tiempo).** Serie diaria continua (índice order_date; días sin ventas interpolados) o
semanal (bloques de 7 días). Se analiza con **descomposición aditiva**, prueba **ADF** de estacionariedad y
**ACF/PACF**, y se ajustan tres modelos: **Holt-Winters** (tendencia amortiguada + estacionalidad aditiva),
**ARIMA(p,d,q)** y **SARIMA(p,d,q)(P,D,Q)m**, con órdenes elegidos por AIC en el primer tramo de entrenamiento.
Se validan con backtesting de origen móvil (4 semanas por corte) contra dos referencias simples (ingenuo y media
móvil) usando MAE, RMSE y MAPE; en modo automático se usa el de menor MAPE.

**Reducción de dimensiones.** PCA sobre el consumo semanal estandarizado de los insumos: pocas componentes
explican casi toda la varianza porque el consumo lo determinan las pizzas vendidas.

**Inventario.** Revisión continua en base **semanal** (la misma de la demanda): μ y σ en porciones/semana,
L en semanas (días ÷ 7), H semanal (H anual ÷ semanas del año). SS = Z·σ·√L, ROP = μ·L + SS, EOQ = √(2μS/H).

**Conexión entre modelos.**
1. Precio: cada categoría recibe Δ_c; la demanda se multiplica por (1+Δ_c)^ε_c y el ingreso por (1+Δ_c)^(1+ε_c).
2. Pronóstico: g = pronóstico medio semanal ÷ media semanal histórica de la serie.
3. Inventario: μ' = μ × g × m_insumo, con σ' = σ × g × m_insumo (coeficiente de variación constante).

4. Inventario → ventas: con σ, L y Z se calcula el faltante esperado por ciclo (σ·√L·G(z)) y el *fill rate*
   (1 − faltante ÷ EOQ). Las pizzas vendidas = demanda × fill rate: un menor nivel de servicio o un mayor lead time
   reduce ventas e ingreso.
5. Economía: margen = ingreso − costo de insumos (% del precio actual, fijo por pizza) − costo de inventario del
   horizonte (costo semanal × h semanas, convertido a US$). El **precio óptimo** es el Δ que maximiza ese
   margen, por lo que S, H, L, Z, el pronóstico y la elasticidad influyen en la recomendación de precio.
6. Alcance común: la serie elegida (Top 5, todas las pizzas o un producto) define las pizzas del pronóstico, del
   ingreso y de los insumos del inventario.

**Escenarios y base.** Histórico = consumo 2015 · **Base** = pronóstico con el precio actual (factor g) ·
Escenario = base × efecto del precio (factor m). Todas las variaciones «vs base» se miden contra la base.

**Unidades y monedas.** Tiempo: semanas en todo el sistema (horizonte, pronóstico, consumo, lead time, costos).
Cantidades: la demanda en pizzas se convierte a porciones de cada insumo con la receta histórica (porciones por
pizza). Dinero: ventas e ingresos en US$ (moneda del dataset); S y H en S/, convertidos con el tipo de cambio.

**Limitaciones.**
- Los precios no cambian en el tiempo dentro de 2015: la elasticidad se identifica con variación *entre* variantes
  (corte transversal), por lo que puede mezclarse con diferencias de popularidad entre productos.
- ε global no es significativa; las elasticidades por categoría son inestables. Trate el simulador como escenario.
- No hay datos de costos: el «óptimo de ingreso» no es un óptimo de utilidad.
- El fill rate por pizza se aproxima con el promedio ponderado del fill rate de sus insumos.
- El costo de insumos por pizza es un supuesto (el dataset no trae costos).
- Los costos S y H son supuestos del informe, no datos observados.
        """
    )
