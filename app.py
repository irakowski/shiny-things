from shiny import App, render, ui, reactive
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import shinyswatch

from conf import BASE_DIR

emission_df = (pd
            .read_csv(BASE_DIR/'data/data.csv', 
                usecols=['Country', 'Year', 'Total', 'Coal', 'Oil', 'Gas'])
            .assign(years_dt=lambda _df: pd.to_datetime(_df['Year'], format='%Y'))
            .dropna(subset=['Coal', 'Oil', 'Gas'])
)

countries = emission_df.Country.unique().tolist()

df = (
    pd.read_csv(
        filepath_or_buffer=BASE_DIR / 'data/bottle.csv',
        usecols=["Salnty", "T_degC"])[:400]
        .dropna(axis=0, how='any')
        .sample(frac=1, random_state=124)
        .reset_index(drop=True)
    )



app_ui = ui.page_fluid(
    shinyswatch.theme.darkly(),
    {"style": "margin-top: 35px"},
    ui.navset_tab(
        ui.nav("LR", 
            ui.panel_title("Linear Regression: Modeling Oceanographic Data", window_title='Shiny-demo'),
            ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_numeric('plm_degree', 'Polymonial degree', value=1),
                # ui.input_numeric("mse", "MSE", value=0.2, max=3, step=0.1)
                ),
            ui.panel_main(
                ui.output_plot("scatter_plot")),
            ),   
        ),
        ui.nav("Emission by Countries",
            ui.panel_title("Emission", window_title='Shiny-demo'),
            ui.layout_sidebar(
            ui.panel_sidebar(
                ui.input_selectize(id='country', label='Select Countries', choices=countries),
                ui.input_date_range(id='years', label='Years range', format='yyyy')
            ),
            ui.panel_main(
                ui.output_table("emission_table"))
            ),
        )
    )
)




def server(input, output, session):

    @reactive.Calc
    def polynomial_features():
        x = df['Salnty'].values.reshape(-1, 1) #-1 = all
        polynomial_degree = input.plm_degree()
        features = np.hstack([x**i for i in range(0, polynomial_degree+1)])
        return features

    @reactive.Calc
    def plot_polymonial():
        ...


    @output
    @render.plot
    def scatter_plot():
        x_train = df['Salnty'].values.reshape(-1, 1)
        y_train = df['T_degC'].values.reshape(-1, 1)
        
        # OLS + Plotting Functions
        def ordinary_least_squares(x, y):
            xTx = x.T.dot(x)
            xTx_inv = np.linalg.inv(xTx)
            w = xTx_inv.dot(x.T.dot(y))
            return w
    
        def polynomial(values, coeffs):
            assert len(values.shape) == 2
            # Coeffs are assumed to be in order 0, 1, ..., n-1
            expanded = np.hstack([coeffs[i] * (values ** i) for i in range(0, len(coeffs))])
            return np.sum(expanded, axis=-1)

        def plot_polynomial(coeffs, x_range=[x_train.min(), x_train.max()], color='darkorange', label='polynomial', alpha=1.0):
            values = np.linspace(x_range[0], x_range[1], 1000).reshape([-1, 1])
            poly = polynomial(values, coeffs)
            plt.plot(values, poly, color=color, linewidth=2, label=label, alpha=alpha)
    
       
        #loss = input.mse()
        

        features = polynomial_features()
        w = ordinary_least_squares(features, y_train)

        plt.figure(figsize=(10, 5))
        plt.xlabel("Temperature")
        plt.ylabel("Salinity")
        plt.title("Polynomial degree: {}".format(input.plm_degree()))
        
        plot_polynomial(w)
        return plt.scatter(x_train, y_train, color='cornflowerblue')
        # return plt.scatter(df["T_degC"], df["Salnty"], color='cornflowerblue')

    @reactive.Effect
    def _():
        country = input.country.get()
        years = emission_df[emission_df['Country']==country].Year.unique().tolist()
        ui.update_date_range(
            "years",
            start=pd.to_datetime(min(years), format='%Y'), 
            end=pd.to_datetime(max(years), format='%Y')
        )


    @output
    @render.table
    def emission_table():
        print(input.years.get())
        return (
            emission_df[
                (emission_df['Country'] == input.country.get()) & 
                (emission_df['years_dt'].between(pd.to_datetime(input.years.get()[0]), 
                                             pd.to_datetime(input.years.get()[1])))]
        )
app = App(app_ui, server)
