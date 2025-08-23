import matplotlib.pyplot as plt
import plotly.graph_objects as go

    def plot_weights_plotly(self):
        """Plot optimized weights using plotly."""
        check_is_fitted(self)

        weights = self.weights_.sort_values(by='weights')
        tickers = weights.index
        values = weights['weights'].values.flatten()

        colors = np.where(values >= 0, '#1f77b4', '#ff7f0e')
        text_positions = ['middle right' if v >= 0 else 'middle left' for v in values]
        text_labels = [f"{v:.1%}  {ticker}" if v >= 0 else f"{v:.1%}  {ticker}"
                       for v, ticker in zip(values, tickers)]

        fig = go.Figure()

        # Lollipop sticks
        for val, i in zip(values, range(len(tickers))):
            fig.add_trace(go.Scatter(
                x=[0, val], y=[i, i],
                mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Lollipop heads
        fig.add_trace(go.Scatter(
            x=values,
            y=list(range(len(tickers))),
            mode='markers+text',
            marker=dict(size=10, color=colors),
            text=text_labels,
            textposition=text_positions,
            textfont=dict(size=10, color=colors),
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))

        # Layout settings
        fig.update_layout(
            title=dict(
                text='Optimized Portfolio Weights',
                x=0.5, xanchor='center',
                font=dict(size=16, family='Arial', color='black')
            ),
            xaxis_title='Weight',
            xaxis_tickformat='.0%',
            plot_bgcolor='white',
            height=max(400, 20 * len(weights)),
            margin=dict(l=120, r=120, t=60, b=40)
        )

        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor='black')

        # Check plotting environment
        if os.environ.get('RUNNING_IN_EMACS') == "1":
            fig.show(renderer='png')
        else:
            fig.show(config={'responsive': True})
