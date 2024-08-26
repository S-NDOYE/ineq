import streamlit as st
import pandas as pd
import numpy as np
from millify import millify
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
import plotly.express as px
import os
import base64

# Page configuration
st.set_page_config(page_title="Inequality Indices Calculation Module", layout="wide")

# Add the company logo to the top right
base_path = os.path.dirname(__file__)
logo_path = os.path.join(base_path, 'ISASD-Logo.png')

# Insert the logo at the top of the page
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1>Inequality Indices Calculation Module</h1>
        <img src="data:image/png;base64,{base64.b64encode(open(logo_path, "rb").read()).decode()}" alt="Company Logo" style="height: 100px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Function to underline the title in blue
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

# Safe millify function to handle infinite values
def safe_millify(value, precision=3):
    if np.isinf(value):
        return "Infinity"
    else:
        return millify(round(value, precision), precision=precision)

# Function to calculate the weighted Gini index
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
        raise ValueError("At least one weight is negative")
    if np.all(weights == 0):
        raise ValueError("All weights are zero")
    
    weights = weights / np.sum(weights)
    order = np.argsort(x)
    x = x[order]
    weights = weights[order]

    p = np.cumsum(weights)
    nu = np.cumsum(weights * x)
    nu = nu / nu[-1]

    gini = np.sum(nu[1:] * p[:-1]) - np.sum(nu[:-1] * p[1:])
    return gini, p, nu

# Function to calculate the Atkinson index
def atkinson(x, parameter=0.5, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    if parameter == 0:
        return 1 - np.exp(np.mean(np.log(x))) / np.mean(x)
    else:
        return 1 - (np.mean(x**parameter))**(1/parameter) / np.mean(x)

# Function to calculate the Theil index with a default parameter of 0.5
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

# Function to calculate the coefficient of variation
def var_coeff(x, na_rm=True):
    if na_rm:
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
    mean_x = np.mean(x)
    if mean_x == 0:
        return np.nan  # Avoid division by zero
    return np.std(x) / mean_x

# Function to calculate the Kolm index
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

# Function to calculate the Palma ratio
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

# Function to calculate entropy
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

# Function to generate the Lorenz curve with Plotly
def plot_lorenz_curve(p, nu, gini_value):
    fig = go.Figure()

    # Adding the Lorenz curve
    fig.add_trace(go.Scatter(
        x=p, 
        y=nu, 
        mode='lines', 
        name=f'Lorenz Curve (Gini = {gini_value:.2f})', 
        line=dict(color='blue')
    ))

    # Adding the perfect equality line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1], 
        mode='lines', 
        name='Perfect Equality', 
        line=dict(color='grey', dash='solid')
    ))

    # Updating the layout to add axes and a light gray background
    fig.update_layout(
        title='Lorenz Curve',
        xaxis_title='Cumulative Population Proportion',
        yaxis_title='Cumulative Income Proportion',
        showlegend=True,
        width=1000,
        height=600,
        plot_bgcolor='rgba(240, 240, 240, 0.9)',  # Light gray background
        xaxis=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black')
    )

    return fig

# Function to generate the distribution histogram
def plot_distribution_histogram(data, column_name):
    fig = px.histogram(data, x=column_name, nbins=50, title=f"Distribution of {column_name}", color_discrete_sequence=["blue"])
    fig.update_traces(marker=dict(line=dict(color="grey", width=1)))  # Gray border around bars

    # Calculating percentages above the bars
    counts = data[column_name].value_counts(bins=50, sort=False)
    total = len(data[column_name])
    percentages = (counts / total * 100).round(1).astype(str) + '%'
    
    fig.update_traces(text=percentages, textposition='outside')
    fig.update_layout(
        xaxis_title=column_name,
        yaxis_title="Frequency",
        plot_bgcolor='rgba(240, 240, 240, 0.9)',  # Light gray background
        width=1000,
        height=600
    )
    return fig

# Function to generate a decile table with cumulative percentages
def generate_decile_table(data):
    deciles = pd.qcut(data, 10, labels=False, duplicates='drop')
    counts = deciles.value_counts().sort_index()
    cum_counts = counts.cumsum()
    total = len(data)
    
    decile_table = pd.DataFrame({
        'Decile': [f'Decile {i+1}' for i in range(len(counts))],
        'Min': [data[deciles == i].min() for i in range(len(counts))],
        'Max': [data[deciles == i].max() for i in range(len(counts))],
        'Average': [data[deciles == i].mean() for i in range(len(counts))],
        'Cumulative Percentage': (cum_counts / total * 100).round(0).astype(int).astype(str) + '%'
    })

    return decile_table

# General introduction
st.write("""
Welcome to this module for calculating economic inequality indices. This tool is designed to provide an in-depth analysis of economic disparities using several standardized and widely recognized indices.
The indices available in this module include:
- **Weighted Gini**
- **Atkinson**
- **Theil**
- **Kolm**
- **Palma**
- **Entropy**
""")

# Adding the title to the sidebar
st.sidebar.subheader("Parameter Selection")

# Adding the choice of the parameter value for the Atkinson index
parameter_atkinson = st.sidebar.number_input(
    "Choose the parameter value for the Atkinson index", 
    min_value=0.0, 
    value=0.5, 
    step=0.1,
    format="%.2f"  # Use period as the decimal separator
)

# Adding the choice of the parameter value for the Theil index
parameter_theil = st.sidebar.number_input(
    "Choose the parameter value for the Theil index", 
    min_value=0.0, 
    value=0.5, 
    step=0.1,
    format="%.2f"  # Use period as the decimal separator
)

# Adding the choice of the epsilon parameter for the Kolm index
epsilon_kolm = st.sidebar.number_input(
    "Choose the epsilon parameter value for the Kolm index", 
    min_value=1e-6,  # A small positive value to avoid epsilon = 0
    value=1.0, 
    step=0.1,
    format="%.2f"  # Use period as the decimal separator
)

# Adding the choice of the parameter value for entropy
parameter_entropy = st.sidebar.number_input(
    "Choose the parameter value for entropy", 
    min_value=0.0, 
    value=0.5, 
    step=0.1,
    format="%.2f"  # Use period as the decimal separator
)

# Adding definitions with dropdown menus
with st.expander("Definition of the Atkinson index"):
    st.markdown("""
    **The Atkinson index** is a measure of economic inequality that takes into account social preferences for equality. 
    It allows emphasis on the impact of inequalities among the poorest or the richest depending on the value of the parameter \( \epsilon \).
    
    ###### Formula of the Atkinson index
    """)
    
    st.markdown(r"""
    The Atkinson index $A(\epsilon)$ for a distribution of $n$ individuals with incomes $y_i$ is defined by:
    
    $$
    A(\epsilon) = 
    \begin{cases}
    1 - \frac{1}{\mu} \left( \frac{1}{n} \sum_{i=1}^{n} y_i^{1-\epsilon} \right)^{\frac{1}{1-\epsilon}} & \text{if } \epsilon \neq 1 \\
    1 - \frac{1}{\mu} \exp\left(\frac{1}{n} \sum_{i=1}^{n} \ln(y_i)\right) & \text{if } \epsilon = 1
    \end{cases}
    $$
    
    where $\mu$ is the average income, $y_i$ is the income of individual $i$, and $\epsilon$ is a parameter of inequality aversion.
    """)

with st.expander("Definition of the Gini index"):
    st.markdown("""
    **The Gini index** is a measure of income or wealth distribution inequality among members of a society. 
    It ranges from 0 to 1, where 0 represents perfect equality (everyone has the same income) and 1 represents maximum inequality (one person has all the income).
    
    ###### Formula of the Gini index
    """)
    
    st.markdown(r"""
    For a distribution of $n$ individuals with incomes $y_1, y_2, \dots, y_n$ ordered such that $ y_1 \leq y_2 \leq \dots \leq y_n $, the Gini index is calculated by:
    
    $$
    G = 1 - 2 \int_0^1 L(p) \, dp
    $$
    
    where $L(p)$ is the Lorenz curve, which represents the cumulative proportion of total income held by the poorest individuals.
    """)


with st.expander("Definition of the Theil index"):
    st.markdown("""
    **The Theil index** is a measure of economic inequality that belongs to the family of generalized entropy indices. 
    It is sensitive to income differences in the upper part of the income distribution. A Theil index of 0 indicates perfect equality, while a higher value indicates increased inequality.
    
    ###### Formula of the Theil index
    """)

    st.markdown(r"""
    For a distribution of $n$ individuals with incomes $y_1, y_2, \dots, y_n$, the Theil index can be calculated in two main ways, depending on the chosen parameter:
    
    - **Theil T index**: It is defined by:
    
    $$
    T = \frac{1}{n} \sum_{i=1}^{n} \frac{y_i}{\mu} \ln\left(\frac{y_i}{\mu}\right)
    $$
    
    - **Theil L index**: It is defined by:
    
    $$
    L = \frac{1}{n} \sum_{i=1}^{n} \ln\left(\frac{\mu}{y_i}\right)
    $$

    where $ \mu $ is the average income, $ y_i $ is the income of individual $ i $.

    The $ T $ index measures inequalities by focusing on high incomes, while the $ L $ index is more sensitive to low incomes.
    """)

with st.expander("Definition of the Kolm index"):
    st.markdown("""
    **The Kolm index** is a measure of economic inequality that focuses on income differences at the lower end of the distribution. 
    Unlike other indices that focus on high incomes, the Kolm index is more sensitive to inequalities affecting the most disadvantaged.
    
    ###### Formula of the Kolm index
    """)

    st.markdown(r"""
    For a distribution of $n$ individuals with incomes $y_1, y_2, \dots, y_n$, the Kolm index $K_\epsilon$ is defined by:
    
    $$
    K_\epsilon = \frac{1}{\epsilon} \log \left(\frac{1}{n} \sum_{i=1}^{n} \exp(-\epsilon (y_i - \mu)) \right)
    $$

    where $ \mu $ is the average income, $ y_i $ is the income of individual $ i $, and $ \epsilon $ is a parameter of inequality aversion. 
    A high $ \epsilon $ makes the index more sensitive to low incomes, thus highlighting inequalities at the lower end of the distribution.
    """)

with st.expander("Definition of the Palma index"):
    st.markdown("""
    **The Palma index** is a measure of economic inequality that focuses on the comparison between the incomes of the richest and the poorest. 
    It is defined as the ratio of the income of the richest 10% of the population to the income of the poorest 40%.
    
    ###### Formula of the Palma index
    """)

    st.markdown(r"""
    The Palma index is defined by:
    
    $$
    P = \frac{\text{Income of the richest 10\%}}{\text{Income of the poorest 40\%}}
    $$

    This index emphasizes the extremes of income distribution, particularly the gap between the very rich and the poorest. 
    A high Palma index indicates a high level of inequality, while a Palma index close to 1 would indicate a more equal distribution of income between the extremes.
    """)

with st.expander("Definition of the entropy index"):
    st.markdown("""
    **The Entropy index** is part of the family of generalized entropy indices. 
    It allows measuring inequality according to an inequality aversion parameter $\\alpha$, which determines the sensitivity of the index to income differences.
    
    ###### Formula of the Entropy index
    """)

    st.markdown(r"""
    The Entropy index for a distribution of $n$ individuals with incomes $y_i$ is defined by:
    
    $$
    E(\alpha) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{\alpha(\alpha-1)} \left[\left(\frac{y_i}{\mu}\right)^\alpha - 1\right] \text{ for } \alpha \neq 0, 1
    $$

    where $ \mu $ is the average income, $ y_i $ is the income of individual $ i $, and $ \alpha $ is an inequality aversion parameter.

    For $ \alpha = 0 $, the Entropy index reduces to the Theil $ L $ index, and for $ \alpha = 1 $, it reduces to the Theil $ T $ index.

    The Entropy index can also be expressed in terms of population percentages by defining specific values of $ \alpha $:
    
    $$
    E(\alpha) = \frac{1}{\alpha(\alpha-1)} \left[\left(\frac{y_{10\%}}{\mu}\right)^\alpha - 1\right]
    $$

    This index allows weighting inequalities according to social preferences, where a higher $ \alpha $ value gives more weight to inequalities at the lower or upper end of the income distribution.
    """)


# File upload
st.markdown(underlined_title("File Upload"), unsafe_allow_html=True)
separator_option = st.radio("Choose the separator for the CSV file:", 
                            options=["Tab", "Semicolon", "Comma", "Space", "Other (specify)"],
                            index=2)

separator_map = {
    "Tab": "\t",
    "Semicolon": ";",
    "Comma": ",",
    "Space": " "
}

if separator_option == "Other (specify)":
    separator = st.text_input("Specify the separator")
else:
    separator = separator_map[separator_option]

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Reading the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=separator)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(df.head())  # Show a preview of the data

    # Selecting the numeric column
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Select a column to check the data", numeric_columns)

    if selected_column:
        # Checking for negative values
        if (df[selected_column] < 0).any():
            st.error("Warning, your data includes one or more negative values. The process cannot continue.")
        else:
            st.success("The data is valid. You can proceed with the calculation.")
        
            # Adding options to remove missing data and use custom weights above the basic statistics
            st.subheader("Parameter Selection")

            col1, col2 = st.columns(2)

            with col1:
                na_rm = st.checkbox("Remove missing data?", value=True)

            with col2:
                use_weights = st.checkbox("Use custom weights")

            weights = None
            if use_weights:
                selected_weight_column = st.selectbox("Select a weight column", df.columns.tolist())
                weights = df[selected_weight_column]
            
            # Calculating inequality indices
            gini_value, p, nu = gini_weighted(df[selected_column], weights, na_rm=na_rm)
            st.session_state.gini_value = gini_value
            st.session_state.p = p
            st.session_state.nu = nu
            st.session_state.atkinson_value = atkinson(df[selected_column], parameter=parameter_atkinson, na_rm=na_rm)
            st.session_state.theil_value = theil(df[selected_column], parameter=parameter_theil, na_rm=na_rm)
            st.session_state.kolm_value = kolm_inequality(df[selected_column], epsilon=epsilon_kolm, na_rm=na_rm)
            st.session_state.palma_value = palma_ratio(df[selected_column], na_rm=na_rm)
            st.session_state.entropy_value = entropy(df[selected_column], parameter=parameter_entropy, na_rm=na_rm)

            # Displaying basic statistics
            st.subheader("Basic Statistics")
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
                st.metric(label="Mean", value=safe_millify(mean_value, precision=3))

            with col4:
                st.metric(label="Median", value=safe_millify(median_value, precision=3))

            with col5:
                st.metric(label="Coefficient of Variation", value=safe_millify(cv_value, precision=3))

            # Displaying results in 2 rows and 3 columns
            st.subheader("Inequality Indices Results")

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            with col1:
                st.metric(label="Weighted Gini Index", value=safe_millify(st.session_state.gini_value, precision=3))

            with col2:
                st.metric(label=f"Atkinson Index (parameter={parameter_atkinson:.2f})", value=safe_millify(st.session_state.atkinson_value, precision=3))

            with col3:
                st.metric(label=f"Theil Index (parameter={parameter_theil:.2f})", value=safe_millify(st.session_state.theil_value, precision=3))

            with col4:
                st.metric(label=f"Kolm Index (epsilon={epsilon_kolm:.2f})", value=safe_millify(st.session_state.kolm_value, precision=3))

            with col5:
                st.metric(label="Palma Index", value=safe_millify(st.session_state.palma_value, precision=3))

            with col6:
                st.metric(label=f"Entropy (parameter={parameter_entropy:.2f})", value=safe_millify(st.session_state.entropy_value, precision=3))

            # Displaying the distribution histogram
            st.subheader(f"Distribution of {selected_column}")
            histogram_fig = plot_distribution_histogram(df, selected_column)
            st.plotly_chart(histogram_fig)

            # Displaying the decile table
            st.subheader("Decile Table")
            decile_table = generate_decile_table(df[selected_column])
            st.table(decile_table)

            # Displaying the Lorenz curve if calculated
            st.subheader("Lorenz Curve")
            fig = plot_lorenz_curve(st.session_state.p, st.session_state.nu, st.session_state.gini_value)
            st.plotly_chart(fig)

            # Simplified styling of the metric cards
            style_metric_cards()

st.subheader("Bibliography")

st.markdown("""
**F A Cowell**: *Measurement of Inequality*, 2000, in A B Atkinson / F Bourguignon (Eds): Handbook of Income Distribution, Amsterdam.

**F A Cowell**: *Measuring Inequality*, 1995, Prentice Hall/Harvester Wheatsheaf.

**Marshall / Olkin**: *Inequalities: Theory of Majorization and Its Applications*, New York, 1979, Academic Press.
""")

st.markdown("""
Developed by [ISASD](http://www.isasd.fr)
""")
