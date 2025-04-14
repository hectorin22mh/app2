import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

################################################
tokenAI = st.secrets["tokenAI"]
newsapi_key = st.secrets["newsapi_key"]
################################################

def translate_with_gemini(text):
    try:
        client = genai.Client(api_key=tokenAI)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=f"Traduce al español este texto sin encabezado ni introducción, solo la traducción directa: {text}"
        )
        return response.text.strip()
    except Exception:
        return text  # En caso de error, devolver el texto original

def get_similar_tickers(ticker):
    all_tickers = yf.Ticker("AAPL").history(period="1d").index  # Listado de tickers disponibles
    return [t for t in all_tickers if ticker.lower() in t.lower()][:5]  # Devolver los más similares

def get_news_from_newsapi(company_name):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "language": "es",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": newsapi_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return []

def fetch_stock_data(symbol, period, full_history=False):
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period="2d")  # Obtener los últimos 2 días para comparación
        info = stock.info
        market_cap = info.get('marketCap', 'N/A')
        volume = info.get('volume', 'N/A')

        if len(history) > 1:
            last_price = history.iloc[-1]['Close']
            prev_close_price = history.iloc[-2]['Close']  # Tomar el cierre del día anterior
            percent_change = ((last_price - prev_close_price) / prev_close_price) * 100
        elif len(history) == 1:
            last_price = history.iloc[-1]['Close']
            open_price = history.iloc[-1]['Open']
            percent_change = ((last_price - open_price) / open_price) * 100
        else:
            last_price, percent_change = None, None

        day_high = history.iloc[-1]['High']  # Máximo del día
        day_low = history.iloc[-1]['Low']  # Mínimo del día
        return last_price, percent_change, market_cap, volume, day_high, day_low
    except Exception:
        return None, None, None, None

def get_investment_recommendation(symbol, last_price, day_high, day_low, market_cap, volume):
    try:
        # Obtener datos históricos de 1 año
        stock = yf.Ticker(symbol)
        history = stock.history(period="1y")

        if history.empty:
            return "No hay datos históricos suficientes para generar una recomendación."

        # Cálculo de métricas clave
        annual_return = ((history['Close'][-1] - history['Close'][0]) / history['Close'][0]) * 100
        volatility = history['Close'].pct_change().std() * (252**0.5) * 100  # Volatilidad anualizada
        max_close = history['Close'].max()
        min_close = history['Close'].min()
        avg_volume = int(history['Volume'].mean())

        beta = stock.info.get("beta", None)
        risk_free_rate = 0.04  # Tasa libre de riesgo aproximada (bono a 10 años USA)
        market_return = 0.08   # Rentabilidad promedio del mercado (S&P500)

        capm_estimate = None
        if beta is not None:
            capm_estimate = risk_free_rate + beta * (market_return - risk_free_rate)

        # Construir resumen para el prompt
        metrics_summary = f"""
        Análisis histórico del activo {symbol} (últimos 12 meses):
        - Rentabilidad anual estimada: {annual_return:.2f}%
        - Volatilidad anual: {volatility:.2f}%
        - Precio máximo: {max_close:.2f} USD
        - Precio mínimo: {min_close:.2f} USD
        - Volumen promedio: {avg_volume:,}
        """

        if beta is not None and capm_estimate is not None:
            metrics_summary += f"""
            - Beta estimada: {beta:.2f}
            - CAPM estimado (Rentabilidad esperada): {capm_estimate:.2%}
            """

        # Prompt con base en datos reales
        prompt = f"""
        Eres un analista financiero experto en análisis de riesgos de inversión. Tu especialidad es evaluar el riesgo-retorno de activos financieros a nivel global.

        A continuación se presentan los datos de un activo financiero. Analiza y proporciona un informe claro sin incluir código Python:

        {metrics_summary}

        Información adicional actual:
        - Última cotización: {last_price} USD
        - Máximo del día: {day_high} USD
        - Mínimo del día: {day_low} USD
        - Capitalización de mercado: {market_cap}
        - Volumen del día: {volume}

        Tu análisis debe incluir:
        - Un resumen del nivel de riesgo del activo.
        - Un reporte claro con hallazgos clave.
        - Un análisis estimado del CAPM (Capital Asset Pricing Model) basado en el riesgo sistemático del activo.
        - Una recomendación diferenciada para tres perfiles de inversionista: conservador, moderado y agresivo.

        🚨 Restricciones:
        - No devuelvas código Python.
        - No expliques cómo se calculan las métricas.
        - Usa lenguaje técnico, pero comprensible para personas con conocimientos intermedios en finanzas.
        """

        genai.configure(api_key=tokenAI)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "No se pudo generar una recomendación en este momento."


def plot_stock_chart(symbol, period):
    try:
        stock = yf.Ticker(symbol)

        history = stock.history(period=period)

        if history.empty:
            return None  

        first_price = history.iloc[0]['Close']
        last_price = history.iloc[-1]['Close']
        line_color = "#28A745" if last_price > first_price else "#DC3545"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history.index, y=history['Close'], 
            mode='lines', name='Precio de Cierre', 
            line=dict(color=line_color, width=2)
        ))
        fig.update_layout(
            title=f"Historial de Precios de {symbol} ({period})",
            xaxis_title="Fecha",
            yaxis_title="Precio (USD)",
            hovermode="x",
            template="plotly_dark"
        )
        return fig
    except Exception:
        return None

def calculate_period_change(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period=period)

        if history.empty or len(history) < 2:
            return None  

        first_price = history.iloc[0]['Close']
        last_price = history.iloc[-1]['Close']
        percent_change = ((last_price - first_price) / first_price) * 100

        return percent_change
    except Exception:
        return None

st.title("📈 Información de Empresas en la Bolsa")

ticker = st.text_input("Escribe el símbolo bursátil", placeholder="Ejemplo: AAPL", help="Ingresa el símbolo de la empresa que deseas buscar")

# Definir opciones de período ANTES de usarlas
period_options = {
    "1 Día": "1d",
    "1 Semana": "5d",
    "1 Mes": "1mo",
    "3 Meses": "3mo",
    "6 Meses": "6mo",
    "1 Año": "1y",
    "2 Años": "2y",
    "5 Años": "5y",
    "10 Años": "10y",
    "Todo": "max"
}

if ticker:
    last_price, percent_change, market_cap, volume, day_high, day_low = fetch_stock_data(ticker, "1d", full_history=True)
    
    info = yf.Ticker(ticker).info
    if 'longName' in info:
        # Centrar el título de la empresa
        st.markdown(
            f"<div style='text-align: center; font-size: 32px; font-weight: bold;'>{info['longName']}</div>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([3, 1, 1])

        # Reducir tamaño y alinear a la izquierda con CSS
        st.markdown(
            """
            <style>
            div[data-baseweb="select"] {
                max-width: 200px !important;
                text-align: left !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        market_cap_str = f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else "N/D"
        volume_str = f"{volume:,}" if isinstance(volume, (int, float)) else "N/D"

        st.markdown(
            f"""
            <div style='text-align: center; font-size: 18px; margin-bottom: 10px;'>
                <strong>Último:</strong> {last_price:.2f} USD | 
                <strong>Máximo:</strong> {day_high:.2f} USD | 
                <strong>Mínimo:</strong> {day_low:.2f} USD | 
                <strong>Market Cap:</strong> {market_cap_str} | 
                <strong>Volumen:</strong> {volume_str}
            </div>
            """,
            unsafe_allow_html=True
        )


        # Mostrar gráfica de la acción
        selected_period = st.selectbox("Selecciona el período del gráfico", list(period_options.keys()), key="period_select")
        
        period_change = calculate_period_change(ticker, period_options[selected_period])
        
        if period_change is not None:
            change_color = "green" if period_change > 0 else "red"
            change_symbol = "+" if period_change > 0 else ""
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; font-size: 18px; font-weight: bold;'>
                    <div></div>
                    <div style='color: {change_color};'>{change_symbol}{period_change:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        chart = plot_stock_chart(ticker, period_options[selected_period])
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.write("No se encontraron datos para graficar.")


        if 'logo_url' in info and info['logo_url']:
            try:
                response = requests.get(info['logo_url'])
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    st.image(image, width=100)
            except Exception:
                pass
        
        if 'longBusinessSummary' in info:
            translated_text = translate_with_gemini(info['longBusinessSummary'])
            st.markdown(f"<div class='summary'>{translated_text}</div>", unsafe_allow_html=True)

        # Mover el bloque del sitio web aquí
        if 'website' in info and info['website']:
            st.markdown(f"<div class='link'><a href='{info['website']}' target='_blank'>Visitar sitio web</a></div>", unsafe_allow_html=True)

        # Sección de recomendación de inversión
        st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; margin-top: 40px;'>Análisis de Riesgo y Evaluación del Activo</div>", unsafe_allow_html=True)

        recommendation = get_investment_recommendation(ticker, last_price, day_high, day_low, market_cap, volume)

        st.markdown(f"<div class='summary'>{recommendation}</div>", unsafe_allow_html=True)

        # Noticias relacionadas
        st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; margin-top: 40px;'>📰 Noticias Recientes</div>", unsafe_allow_html=True)
        news_articles = get_news_from_newsapi(info['longName'])

        if news_articles:
            for article in news_articles:
                st.markdown(f"**[{article['title']}]({article['url']})**  \n{article['description']}", unsafe_allow_html=True)
        else:
            st.info("No se encontraron noticias recientes.")

    else:
        st.error("No se encontró información para el símbolo ingresado.")
        similar_tickers = get_similar_tickers(ticker)
        if similar_tickers:
            st.write("Tal vez quisiste decir:")
            for t in similar_tickers:
                st.write(f"🔹 {t}")

# Estilos personalizados
st.markdown(
    """
    <style>
    .stTextInput > div > div > input {
        text-align: center;
        font-size: 24px !important;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .summary {
        text-align: justify;
        font-size: 18px;
    }
    .link {
        text-align: center;
        font-size: 16px;
        margin-top: 20px;
    }
    .info-container {
        display: flex;
        justify-content: space-around;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .market-volume-container {
        display: flex;
        justify-content: space-around;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
    }
    .price {
        color: #FFFFFF !important;
        font-size: 22px !important;
        font-weight: bold;
    }
    .change {
        text-align: center;
    }
    .marketcap, .volume {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
