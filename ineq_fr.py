import streamlit as st
import pandas as pd
import numpy as np
from millify import millify
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
import plotly.express as px
import os
import base64

# Configuration de la page
st.set_page_config(page_title="Module de Calcul des Indices d'Inégalité", layout="wide")

# Add the company logo to the top right
base_path = os.path.dirname(__file__)
logo_path = os.path.join(base_path, 'ISASD-Logo.png')

# Insert the logo at the top of the page
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>Module de calcul des indices d'inégalité</h1>
        <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="Company Logo" style="height: 100px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Fonction pour souligner le titre en bleu
def underlined_title(title):
    return f"<h3 style='border-bottom: 2px solid #1f77b4; padding-bottom: 4px; color:#1f77b4'>{title}</h3>"

# Custom sidebar styling with white background and blue border
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: white;
            border: 2px solid #1f77b4;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Fonction safe_millify pour gérer les valeurs infinies
def safe_millify(value, precision=3):
    if np.isinf(value):
        return "Infinity"
    else:
        return millify(round(value, precision), precision=precision)

# Fonction pour calculer l'indice de Gini pondéré
def gini_weighted(x, weights=None, na_rm=True):
    if na_rm:
        x = np.array(x).flatten()
        weights = np.array(weights) if weights is not None else np.ones_like(x)
        missing = ~(np.isnan(x) | np.isnan(weights))
        x = x[missing]
        weights = weights[missing]
    else:
        x = np.array(x).flatten()
        weights = np.array(weights) if weights is not None else np.ones_like(x)
    
    if np.any(weights < 0):
        raise ValueError("Au moins un poids est négatif")
    if np.all(weights == 0):
        raise ValueError("Tous les poids sont nuls")
    
    weights = weights / np.sum(weights)
    order = np.argsort(x)
    x = x[order]
    weights = weights[order]

    p = np.cumsum(weights)
    nu = np.cumsum(weights * x)
    nu = nu / nu[-1]

    gini = np.sum(nu[1:] * p[:-1]) - np.sum(nu[:-1] * p[1:])
    return gini, p, nu

# Fonction pour calculer l'indice d'Atkinson
def atkinson(x, parameter=0.5, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    if parameter == 0:
        return 1 - np.exp(np.mean(np.log(x))) / np.mean(x)
    else:
        return 1 - (np.mean(x**parameter))**(1/parameter) / np.mean(x)

# Fonction pour calculer l'indice de Theil avec un paramètre par défaut de 0.5
def theil(x, parameter=0.5, na_rm=True):
    x = np.array(x, dtype=float)
    
    # Handle zero values to avoid log(0) issues
    x = x[x > 0]  # Filter out zero values
    if len(x) == 0:
        return np.nan  # If all values are zero, return NaN
    
    mean_x = np.mean(x)
    if mean_x == 0:
        return np.nan  # If the mean is zero, return NaN
    
    theil_t = np.mean((x / mean_x) * np.log(x / mean_x))
    return theil_t

# Fonction pour calculer le coefficient de variation
def var_coeff(x, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    mean_x = np.mean(x)
    if mean_x == 0:
        return np.nan  # Évite la division par zéro
    return np.std(x) / mean_x

# Fonction pour calculer l'indice de Kolm
def kolm_inequality(x, epsilon=1, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    mean_x = np.mean(x)
    max_diff = np.max(x) - mean_x
    if epsilon * max_diff > 700:
        scaling_factor = epsilon * max_diff / 700
        scaled_epsilon = epsilon / scaling_factor
        scaled_x = (x - mean_x) / scaling_factor
        result = (1/epsilon) * np.log(np.mean(np.exp(-scaled_epsilon * scaled_x)))
    else:
        result = (1/epsilon) * np.log(np.mean(np.exp(-epsilon * (x - mean_x))))
    return result

# Fonction pour calculer l'indice de Palma
def palma_ratio(x, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    x_sorted = np.sort(x)
    n = len(x_sorted)
    top_10_percent = x_sorted[int(0.9 * n):]
    bottom_40_percent = x_sorted[:int(0.4 * n)]
    palma = top_10_percent.sum() / bottom_40_percent.sum()
    return palma

# Fonction pour calculer l'entropie
def entropy(x, parameter=0.5, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    if parameter == 0:
        return theil(x, parameter=1)
    elif parameter == 1:
        return theil(x, parameter=0)
    else:
        k = parameter
        return np.mean((x / np.mean(x))**k - 1) / (k * (k - 1))

# Fonction pour générer la courbe de Lorenz avec Plotly
def plot_lorenz_curve(p, nu, gini_value):
    fig = go.Figure()

    # Ajout de la courbe de Lorenz
    fig.add_trace(go.Scatter(
        x=p, 
        y=nu, 
        mode='lines', 
        name=f'Courbe de Lorenz (Gini = {gini_value:.2f})', 
        line=dict(color='blue')
    ))

    # Ajout de la ligne d'égalité parfaite
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1], 
        mode='lines', 
        name='Égalité parfaite', 
        line=dict(color='grey', dash='solid')
    ))

    # Mise à jour de la mise en page pour ajouter les axes et un fond gris clair
    fig.update_layout(
        title='Courbe de Lorenz',
        xaxis_title='Proportion cumulée de la population',
        yaxis_title='Proportion cumulée des revenus',
        showlegend=True,
        width=1000,
        height=600,
        plot_bgcolor='rgba(240, 240, 240, 0.9)',  # Fond légèrement gris
        xaxis=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )

    return fig

# Fonction pour générer l'histogramme de distribution
def plot_distribution_histogram(data, column_name):
    fig = px.histogram(data, x=column_name, nbins=50, title=f"Distribution de {column_name}", color_discrete_sequence=["blue"])
    fig.update_traces(marker=dict(line=dict(color="grey", width=1)))  # Contour gris autour des barres

    # Calcul des pourcentages au-dessus des barres
    counts = data[column_name].value_counts(bins=50, sort=False)
    total = len(data[column_name])
    percentages = (counts / total * 100).round(1).astype(str) + '%'
    
    fig.update_traces(text=percentages, textposition='outside')
    fig.update_layout(
        xaxis_title=column_name,
        yaxis_title="Fréquence",
        plot_bgcolor='rgba(240, 240, 240, 0.9)',  # Fond légèrement gris
        width=1000,
        height=600
    )
    return fig

# Fonction pour générer une table des déciles avec pourcentages cumulés
def generate_decile_table(data):
    deciles = pd.qcut(data, 10, labels=False, duplicates='drop')
    counts = deciles.value_counts().sort_index()
    cum_counts = counts.cumsum()
    total = len(data)
    
    decile_table = pd.DataFrame({
        'Décile': [f'Décile {i+1}' for i in range(len(counts))],
        'Min': [data[deciles == i].min() for i in range(len(counts))],
        'Max': [data[deciles == i].max() for i in range(len(counts))],
        'Moyenne': [data[deciles == i].mean() for i in range(len(counts))],
        'Pourcentage cumulé': (cum_counts / total * 100).round(0).astype(int).astype(str) + '%'
    })

    return decile_table

# Introduction générale
st.write("""
Bienvenue dans ce module de calcul des indices d'inégalité économique. Cet outil a été conçu pour permettre une analyse approfondie des disparités économiques en utilisant plusieurs indices standardisés et largement reconnus.
Les indices disponibles dans ce module incluent :
- **Gini pondéré**
- **Atkinson**
- **Theil**
- **Kolm**
- **Palma**
- **Entropie**
""")

# Ajout du titre de la barre latérale
st.sidebar.subheader("Sélection des paramètres")

# Ajout du choix de la valeur du paramètre pour l'indice d'Atkinson
parameter_atkinson = st.sidebar.number_input(
    "Choisissez la valeur du paramètre pour l'indice d'Atkinson", 
    min_value=0.0, 
    value=0.5, 
    step=0.1,
    format="%.2f"  # Utilisation du point pour le séparateur décimal
)

# Ajout du choix de la valeur du paramètre pour l'indice de Theil
parameter_theil = st.sidebar.number_input(
    "Choisissez la valeur du paramètre pour l'indice de Theil", 
    min_value=0.0, 
    value=0.5, 
    step=0.1,
    format="%.2f"  # Utilisation du point pour le séparateur décimal
)

# Ajout du choix de la valeur du paramètre pour l'indice de Kolm
epsilon_kolm = st.sidebar.number_input(
    "Choisissez la valeur du paramètre epsilon pour l'indice de Kolm", 
    min_value=1e-6,  # une petite valeur positive pour éviter epsilon = 0
    value=1.0, 
    step=0.1,
    format="%.2f"  # Utilisation du point pour le séparateur décimal
)

# Ajout du choix de la valeur du paramètre pour l'entropie
parameter_entropy = st.sidebar.number_input(
    "Choisissez la valeur du paramètre pour l'entropie", 
    min_value=0.0, 
    value=0.5, 
    step=0.1,
    format="%.2f"  # Utilisation du point pour le séparateur décimal
)

# Ajout des définitions avec menus déroulants
with st.expander("Définition de l'indice d'Atkinson"):
    st.markdown("""
    **L'indice d'Atkinson** est une mesure de l'inégalité économique qui tient compte des préférences sociales pour l'égalité. 
    Il permet de mettre l'accent sur l'impact des inégalités parmi les plus pauvres ou les plus riches selon la valeur du paramètre \( \epsilon \).
    
    ###### Formule de l'indice d'Atkinson
    """)
    
    st.markdown(r"""
    L'indice d'Atkinson $A(\epsilon)$ pour une distribution de $n$ individus ayant des revenus $y_i$ est défini par:
    
    $$
    A(\epsilon) = 
    \begin{cases}
    1 - \frac{1}{\mu} \left( \frac{1}{n} \sum_{i=1}^{n} y_i^{1-\epsilon} \right)^{\frac{1}{1-\epsilon}} & \text{si } \epsilon \neq 1 \\
    1 - \frac{1}{\mu} \exp\left(\frac{1}{n} \sum_{i=1}^{n} \ln(y_i)\right) & \text{si } \epsilon = 1
    \end{cases}
    $$
    
    où $\mu$ est la moyenne des revenus, $y_i$ est le revenu de l'individu $i$, et $\epsilon$ est un paramètre d'aversion à l'inégalité.
    """)

with st.expander("Définition de l'indice de Gini"):
    st.markdown("""
    **L'indice de Gini** est une mesure de l'inégalité de la distribution des revenus ou des richesses parmi les membres d'une société. 
    Il varie de 0 à 1, où 0 représente l'égalité parfaite (tout le monde a le même revenu) et 1 représente l'inégalité maximale (une seule personne a tout le revenu).
    
    ###### Formule de l'indice de Gini
    """)
    
    st.markdown(r"""
    Pour une distribution de $n$ individus avec des revenus $y_1, y_2, \dots, y_n$ ordonnés tels que $ y_1 \leq y_2 \leq \dots \leq y_n $, l'indice de Gini est calculé par:
    
    $$
    G = 1 - 2 \int_0^1 L(p) \, dp
    $$
    
    où $L(p)$ est la courbe de Lorenz, qui représente la proportion cumulée du revenu total détenue par les individus les plus pauvres.
    """)


with st.expander("Définition de l'indice de Theil"):
    st.markdown("""
    **L'indice de Theil** est une mesure de l'inégalité économique qui appartient à la famille des indices d'entropie généralisée. 
    Il est sensible aux différences de revenus dans la partie supérieure de la distribution des revenus. Un indice de Theil de 0 indique une égalité parfaite, tandis qu'une valeur plus élevée indique une inégalité accrue.
    
    ###### Formule de l'indice de Theil
    """)

    st.markdown(r"""
    Pour une distribution de $n$ individus avec des revenus $y_1, y_2, \dots, y_n$, l'indice de Theil peut être calculé de deux manières principales, selon le paramètre choisi :
    
    - **Indice de Theil T** : Il est défini par :
    
    $$
    T = \frac{1}{n} \sum_{i=1}^{n} \frac{y_i}{\mu} \ln\left(\frac{y_i}{\mu}\right)
    $$
    
    - **Indice de Theil L** : Il est défini par :
    
    $$
    L = \frac{1}{n} \sum_{i=1}^{n} \ln\left(\frac{\mu}{y_i}\right)
    $$

    où $ \mu $ est la moyenne des revenus, $ y_i $ est le revenu de l'individu $ i $.

    L'indice $ T $ mesure les inégalités en mettant l'accent sur les hauts revenus, tandis que l'indice $ L $ est plus sensible aux faibles revenus.
    """)

with st.expander("Définition de l'indice de Kolm"):
    st.markdown("""
    **L'indice de Kolm** est une mesure de l'inégalité économique qui met l'accent sur les différences de revenus en bas de la distribution. 
    Contrairement à d'autres indices qui se concentrent sur les revenus élevés, l'indice de Kolm est plus sensible aux inégalités affectant les personnes les plus défavorisées.
    
    ###### Formule de l'indice de Kolm
    """)

    st.markdown(r"""
    Pour une distribution de $n$ individus ayant des revenus $y_1, y_2, \dots, y_n$, l'indice de Kolm $K_\epsilon$ est défini par :
    
    $$
    K_\epsilon = \frac{1}{\epsilon} \log \left(\frac{1}{n} \sum_{i=1}^{n} \exp(-\epsilon (y_i - \mu)) \right)
    $$

    où $ \mu $ est la moyenne des revenus, $ y_i $ est le revenu de l'individu $ i $, et $ \epsilon $ est un paramètre d'aversion à l'inégalité. 
    Un $ \epsilon $ élevé rend l'indice plus sensible aux faibles revenus, accentuant ainsi les inégalités dans le bas de la distribution.
    """)

with st.expander("Définition de l'indice de Palma"):
    st.markdown("""
    **L'indice de Palma** est une mesure de l'inégalité économique qui se concentre sur la comparaison entre les revenus des plus riches et des plus pauvres. 
    Il est défini comme le ratio des revenus des 10% les plus riches de la population sur les revenus des 40% les plus pauvres.
    
    ###### Formule de l'indice de Palma
    """)

    st.markdown(r"""
    L'indice de Palma est défini par :
    
    $$
    P = \frac{\text{Revenu des 10\% les plus riches}}{\text{Revenu des 40\% les plus pauvres}}
    $$

    Cet indice met l'accent sur les extrêmes de la distribution des revenus, en particulier sur l'écart entre les très riches et les plus pauvres. 
    Un indice de Palma élevé indique une forte inégalité, tandis qu'un indice de Palma proche de 1 indiquerait une répartition plus égale des revenus entre les extrêmes.
    """)

with st.expander("Définition de l'indice d'entropie"):
    st.markdown("""
    **L'indice d'Entropie** fait partie de la famille des indices d'entropie généralisée. 
    Il permet de mesurer l'inégalité en fonction d'un paramètre d'aversité à l'inégalité $\\alpha$, qui détermine la sensibilité de l'indice aux différences de revenus.
    
    ###### Formule de l'indice d'Entropie
    """)

    st.markdown(r"""
    L'indice d'Entropie pour une distribution de $n$ individus ayant des revenus $y_i$ est défini par :
    
    $$
    E(\alpha) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{\alpha(\alpha-1)} \left[\left(\frac{y_i}{\mu}\right)^\alpha - 1\right] \text{ pour } \alpha \neq 0, 1
    $$

    où $ \mu $ est la moyenne des revenus, $ y_i $ est le revenu de l'individu $ i $, et $ \alpha $ est un paramètre d'aversité à l'inégalité.

    Pour $ \alpha = 0 $, l'indice d'entropie se réduit à l'indice de Theil $ L $, et pour $ \alpha = 1 $, il se réduit à l'indice de Theil $ T $.

    L'indice d'Entropie peut également être exprimé en fonction des pourcentages de la population en définissant les valeurs de $ \alpha $ spécifiques :
    
    $$
    E(\alpha) = \frac{1}{\alpha(\alpha-1)} \left[\left(\frac{y_{10\%}}{\mu}\right)^\alpha - 1\right]
    $$

    Cet indice permet de pondérer les inégalités selon les préférences sociales, où une valeur plus élevée de $ \alpha $ donne plus de poids aux inégalités à l'extrémité inférieure ou supérieure de la distribution des revenus.
    """)


# Chargement du fichier
st.markdown(underlined_title("Chargement du fichier"), unsafe_allow_html=True)
separator_option = st.radio("Choisissez le séparateur pour le fichier CSV :", 
                            options=["Tabulation", "Point-virgule", "Virgule", "Espace", "Autre (spécifier)"],
                            index=2)

separator_map = {
    "Tabulation": "\t",
    "Point-virgule": ";",
    "Virgule": ",",
    "Espace": " "
}

if separator_option == "Autre (spécifier)":
    separator = st.text_input("Spécifiez le séparateur")
else:
    separator = separator_map[separator_option]

uploaded_file = st.file_uploader("Charger un fichier CSV ou Excel", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Lecture du fichier
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=separator)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Aperçu des données :")
    st.write(df.head())  # Afficher un aperçu des données

    # Sélection de la colonne numérique
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Sélectionner une colonne pour vérifier les données", numeric_columns)

    if selected_column:
        # Vérification des valeurs négatives
        if (df[selected_column] < 0).any():
            st.error("Attention, vos données comprennent une ou plusieurs valeurs négatives. La procédure ne peut continuer.")
        else:
            st.success("Les données sont valides. Vous pouvez procéder au calcul.")
        
            # Ajout des options pour supprimer les données manquantes et utiliser les poids personnalisés au-dessus des statistiques de base
            st.subheader("Sélection des paramètres")

            col1, col2 = st.columns(2)

            with col1:
                na_rm = st.checkbox("Supprimer les données manquantes ?", value=True)

            with col2:
                use_weights = st.checkbox("Utiliser des poids personnalisés")

            weights = None
            if use_weights:
                selected_weight_column = st.selectbox("Sélectionner une colonne de poids", df.columns.tolist())
                weights = df[selected_weight_column]
            
            # Calculer les indices d'inégalité
            gini_value, p, nu = gini_weighted(df[selected_column], weights, na_rm=na_rm)
            st.session_state.gini_value = gini_value
            st.session_state.p = p
            st.session_state.nu = nu
            st.session_state.atkinson_value = atkinson(df[selected_column], parameter=parameter_atkinson, na_rm=na_rm)
            st.session_state.theil_value = theil(df[selected_column], parameter=parameter_theil, na_rm=na_rm)
            st.session_state.kolm_value = kolm_inequality(df[selected_column], epsilon=epsilon_kolm, na_rm=na_rm)
            st.session_state.palma_value = palma_ratio(df[selected_column], na_rm=na_rm)
            st.session_state.entropy_value = entropy(df[selected_column], parameter=parameter_entropy, na_rm=na_rm)

            # Présentation des statistiques de base
            st.subheader("Statistiques de base")
            min_value = df[selected_column].min()
            max_value = df[selected_column].max()
            mean_value = df[selected_column].mean()
            median_value = df[selected_column].median()
            cv_value = var_coeff(df[selected_column])

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(label="Minimum", value=safe_millify(min_value, precision=3))

            with col2:
                st.metric(label="Maximum", value=safe_millify(max_value, precision=3))

            with col3:
                st.metric(label="Moyenne", value=safe_millify(mean_value, precision=3))

            with col4:
                st.metric(label="Médiane", value=safe_millify(median_value, precision=3))

            with col5:
                st.metric(label="Coefficient de Variation", value=safe_millify(cv_value, precision=3))

            # Affichage des résultats en 2 lignes et 3 colonnes
            st.subheader("Résultats des indices d'inégalité")

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            with col1:
                st.metric(label="Indice de Gini Pondéré", value=safe_millify(st.session_state.gini_value, precision=3))

            with col2:
                st.metric(label=f"Indice d'Atkinson (parameter={parameter_atkinson:.2f})", value=safe_millify(st.session_state.atkinson_value, precision=3))

            with col3:
                st.metric(label=f"Indice de Theil (parameter={parameter_theil:.2f})", value=safe_millify(st.session_state.theil_value, precision=3))

            with col4:
                st.metric(label=f"Indice de Kolm (epsilon={epsilon_kolm:.2f})", value=safe_millify(st.session_state.kolm_value, precision=3))

            with col5:
                st.metric(label="Indice de Palma", value=safe_millify(st.session_state.palma_value, precision=3))

            with col6:
                st.metric(label=f"Entropie (parameter={parameter_entropy:.2f})", value=safe_millify(st.session_state.entropy_value, precision=3))

            # Affichage de l'histogramme de distribution
            st.subheader(f"Distribution de {selected_column}")
            histogram_fig = plot_distribution_histogram(df, selected_column)
            st.plotly_chart(histogram_fig)

            # Affichage du tableau des déciles
            st.subheader("Tableau des déciles")
            decile_table = generate_decile_table(df[selected_column])
            st.table(decile_table)

            # Affichage de la courbe de Lorenz si calculée
            st.subheader("Courbe de Lorenz")
            fig = plot_lorenz_curve(st.session_state.p, st.session_state.nu, st.session_state.gini_value)
            st.plotly_chart(fig)

            # Simplified styling of the metric cards
            style_metric_cards()

st.subheader("Bibliographie")

st.markdown("""
**F A Cowell**: *Measurement of Inequality*, 2000, in A B Atkinson / F Bourguignon (Eds): Handbook of Income Distribution, Amsterdam.

**F A Cowell**: *Measuring Inequality*, 1995, Prentice Hall/Harvester Wheatsheaf.

**Marshall / Olkin**: *Inequalities: Theory of Majorization and Its Applications*, New York, 1979, Academic Press.
""")

st.markdown("""
Développé par l'[ISASD](http://www.isasd.fr)
""")
