import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# Set page configuration
st.set_page_config(page_title="Market Basket Analysis Dashboard", layout="wide")

# Title and description
st.title("Market Basket Analysis Dashboard")
st.write("Analyze customer purchasing patterns and discover product associations")

# Helper Functions
def load_data(file):
    """Load and preprocess the data."""
    df = pd.read_csv(file, index_col=0)  # Skip the first column (transaction ID)
    
    # Convert string 'True'/'False' to boolean if needed
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map({'True': True, 'False': False})
    
    # Convert to boolean/binary if not already
    df = df.astype(bool).astype(int)
    
    return df

def validate_data(df):
    """Validate that data contains only 0s and 1s."""
    return ((df == 0) | (df == 1)).all().all()

def generate_frequent_itemsets(df, min_support):
    """Generate frequent itemsets using Apriori algorithm."""
    return apriori(df, min_support=min_support, use_colnames=True)

def generate_association_rules(frequent_itemsets, min_confidence):
    """Generate association rules from frequent itemsets."""
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_confidence)
    return rules.sort_values(['confidence', 'lift'], ascending=[False, False])

def interpret_rule(rule_row, len_transactions):
    """Generate human-readable interpretation of a rule."""
    antecedents = rule_row['antecedents'].split(', ')
    consequents = rule_row['consequents'].split(', ')
    support = rule_row['support']
    confidence = rule_row['confidence']
    lift = rule_row['lift']
    
    interpretation = f"""
    ### Rule Interpretation
    
    📌 **Pattern Found:**
    When customers buy {', '.join(antecedents)}, they are likely to also buy {', '.join(consequents)}.
    
    📊 **Key Metrics:**
    * **Support = {support:.1%}**
        - {support:.1%} of all transactions contain these items together
        - This represents {int(support * len_transactions)} transactions in your dataset
    
    * **Confidence = {confidence:.1%}**
        - When customers buy {', '.join(antecedents)},
        - {confidence:.1%} of the time they also buy {', '.join(consequents)}
    
    * **Lift = {lift:.2f}**
        - Customers are {lift:.2f}x more likely to buy {', '.join(consequents)}
        - When they buy {', '.join(antecedents)}
        - Compared to the normal probability
    
    💡 **Business Recommendations:**
    1. **Store Layout:**
        - Consider placing {', '.join(consequents)} near {', '.join(antecedents)}
        - Create a dedicated display featuring these items together
    
    2. **Promotional Strategies:**
        - Bundle these products with a special discount
        - Create a "Frequently Bought Together" package
        - Send targeted promotions to customers who buy {', '.join(antecedents)}
    
    3. **Inventory Management:**
        - Stock extra {', '.join(consequents)} when ordering large quantities of {', '.join(antecedents)}
        - Set up automated alerts when {', '.join(antecedents)} sales spike
    
    4. **Marketing Opportunities:**
        - Show "{', '.join(consequents)}" recommendations to online shoppers
        - Create recipe or usage suggestions combining these items
        - Feature these combinations in your weekly circular
    """
    return interpretation

def create_rule_visualization(rule_row):
    """Create a detailed visualization for a specific rule."""
    antecedents = rule_row['antecedents'].split(', ')
    consequents = rule_row['consequents'].split(', ')
    
    # Create nodes for products
    nodes = antecedents + consequents
    node_colors = ['lightblue'] * len(antecedents) + ['lightgreen'] * len(consequents)
    
    # Create a layout for nodes
    pos = {}
    # Position antecedents in a circle on the left
    for i, ant in enumerate(antecedents):
        angle = 2 * np.pi * i / len(antecedents)
        pos[ant] = [np.cos(angle), np.sin(angle)]
    
    # Position consequents on the right
    for i, cons in enumerate(consequents):
        pos[cons] = [2, (i - len(consequents)/2) * 0.5]
    
    # Create traces for nodes
    node_trace = go.Scatter(
        x=[pos[node][0] for node in nodes],
        y=[pos[node][1] for node in nodes],
        mode='markers+text',
        marker=dict(
            size=30,
            color=node_colors
        ),
        text=nodes,
        textposition="middle center",
        hoverinfo='text'
    )
    
    # Create traces for edges
    edge_traces = []
    for ant in antecedents:
        for cons in consequents:
            edge_trace = go.Scatter(
                x=[pos[ant][0], pos[cons][0]],
                y=[pos[ant][1], pos[cons][1]],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        title="Rule Visualization",
        title_x=0.5,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        annotations=[
            dict(
                x=-0.5,
                y=1.2,
                text="Antecedents (IF)",
                showarrow=False,
                font=dict(size=14)
            ),
            dict(
                x=2,
                y=1.2,
                text="Consequents (THEN)",
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    return fig

# def create_network_graph(rules, min_confidence, min_lift):
#     """Create network graph visualization of rules."""
#     filtered_rules = rules[
#         (rules['confidence'] >= min_confidence) &
#         (rules['lift'] >= min_lift)
#     ]
    
#     if filtered_rules.empty:
#         return None
    
#     G = nx.Graph()
    
#     for _, row in filtered_rules.iterrows():
#         antecedents = ', '.join(list(row['antecedents']))
#         consequents = ', '.join(list(row['consequents']))
        
#         G.add_edge(
#             antecedents, 
#             consequents, 
#             weight=row['lift'],
#             confidence=row['confidence']
#         )
    
#     if not G.nodes():
#         return None
        
#     pos = nx.spring_layout(G)
    
#     # Create edges trace
#     edge_x = []
#     edge_y = []
#     edge_text = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])
#         edge_text.extend([f"Lift: {G.edges[edge]['weight']:.2f}<br>Confidence: {G.edges[edge]['confidence']:.2f}", "", ""])

#     edge_trace = go.Scatter(
#         x=edge_x, 
#         y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='text',
#         text=edge_text,
#         mode='lines'
#     )

#     # Create nodes trace
#     node_x = []
#     node_y = []
#     node_text = []
#     node_adjacencies = []
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         node_text.append(node)
#         node_adjacencies.append(len(list(G.neighbors(node))))

#     node_trace = go.Scatter(
#         x=node_x, 
#         y=node_y,
#         mode='markers+text',
#         hoverinfo='text',
#         text=node_text,
#         textposition="bottom center",
#         marker=dict(
#             showscale=True,
#             colorscale='YlOrRd',
#             size=20,
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             color=node_adjacencies
#         )
#     )

#     fig = go.Figure(
#         data=[edge_trace, node_trace],
#         layout=go.Layout(
#             showlegend=False,
#             hovermode='closest',
#             margin=dict(b=20,l=5,r=5,t=40),
#             annotations=[
#                 dict(
#                     text="Product Association Network",
#                     showarrow=False,
#                     xref="paper", 
#                     yref="paper",
#                     x=0.005, 
#                     y=-0.002
#                 )
#             ],
#             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
#         )
#     )
    
#     return fig

def main():
    """Main application function."""
    
    # File upload section (always visible)
    with st.expander("📋 Data Format Instructions"):
        st.markdown("""
        Your CSV file should follow these requirements:
        - Each row represents a transaction
        - Each column represents a product
        - Values should be either:
            - True/False
            - 1/0
            - true/false
        
        Example format:
        ```
        ,Product1,Product2,Product3
        1,True,False,True
        2,False,True,False
        3,True,True,False
        ```
        """)
    
    uploaded_file = st.file_uploader(
        "Upload your transaction data (CSV format)",
        type=['csv'],
        help="Upload a CSV file with boolean values (True/False) indicating item presence in each transaction"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload your transaction data to begin the analysis")
        st.stop()
    
    # Load and validate data
    try:
        df = load_data(uploaded_file)
        if not validate_data(df):
            st.error("Invalid data format. Please ensure your data contains only binary values (0/1 or True/False)")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Analysis parameters in sidebar (always visible)
    st.sidebar.header("Analysis Parameters")
    
    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.5,
        value=0.02,
        step=0.01,
        help="Minimum frequency of itemsets as a fraction of total transactions"
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum probability of consequent given antecedent"
    )
    
    min_lift = st.sidebar.slider(
        "Minimum Lift",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
        help="Minimum lift value (measure of rule strength)"
    )
    
    # Generate insights before creating tabs
    with st.spinner("Generating insights..."):
        frequent_itemsets = generate_frequent_itemsets(df, min_support)
        
        if frequent_itemsets.empty:
            st.warning("No frequent itemsets found with current support threshold. Try lowering the minimum support value.")
            st.stop()
        
        rules = generate_association_rules(frequent_itemsets, min_confidence)
        
        if rules.empty:
            st.warning("No rules found with current parameters. Try adjusting the minimum support, confidence, or lift values.")
            st.stop()
        
        # Prepare rules for display
        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        display_rules = display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        display_rules = display_rules.round(3)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Data Overview", 
        "Analysis", 
        "Recommendations"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Dataset Summary with cards
        st.subheader("Dataset Summary")
        
        # Custom CSS for cards
        st.markdown("""
        <style>
        .summary-card {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            border: 1px solid #ddd;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 1rem;
            color: #555;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #0f1b2a;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="summary-card">
                <div class="metric-label">Number of Transactions</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="summary-card">
                <div class="metric-label">Number of Products</div>
                <div class="metric-value">{len(df.columns):,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="summary-card">
                <div class="metric-label">Average Items per Transaction</div>
                <div class="metric-value">{df.sum(axis=1).mean():.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Product frequency chart and Itemset size distribution in one row
        st.subheader("Purchase Patterns")
        col4, col5 = st.columns(2)
        
        with col4:
            # Product frequency chart
            product_freq = df.sum().sort_values(ascending=False)
            fig_freq = px.bar(
                product_freq,
                title="Product Purchase Frequency",
                labels={'value': 'Frequency', 'index': 'Product'},
                height=400
            )
            fig_freq.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                title_x=0.5,
                title_y=0.95
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        
        with col5:
            # Itemset size distribution
            itemset_sizes = frequent_itemsets['itemsets'].apply(len)
            size_dist = itemset_sizes.value_counts().sort_index()
            fig_sizes = px.bar(
                size_dist,
                title="Frequent Itemset Size Distribution",
                labels={'index': 'Itemset Size', 'value': 'Count'},
                height=400
            )
            fig_sizes.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                title_x=0.5,
                title_y=0.95
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
            
    # Tab 2: Analysis
    # In Tab 2: Analysis
    # Tab 2: Analysis
    with tab2:
        # 1. Overview Visualizations
        st.header("Rule Pattern Analysis")
        
        # Create two columns for scatter plot and heatmap
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Scatter Plot
            scatter_fig = px.scatter(
                display_rules,
                x='confidence',
                y='lift',
                size='support',
                hover_data={
                    'antecedents': True,
                    'consequents': True,
                    'support': ':.3f',
                    'confidence': ':.3f',
                    'lift': ':.3f'
                },
                title='Rule Metrics Distribution',
                labels={
                    'confidence': 'Confidence',
                    'lift': 'Lift',
                    'support': 'Support'
                },
                height=500
            )
            
            scatter_fig.update_traces(
                marker=dict(
                    sizemin=5,
                    sizeref=2.*max(display_rules['support'])/(20.**2),
                    sizemode='area'
                ),
                selector=dict(mode='markers')
            )
            
            scatter_fig.update_layout(
                title_x=0.5,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Confidence",
                yaxis_title="Lift",
                showlegend=False,
                shapes=[
                    dict(
                        type="line",
                        yref="y",
                        y0=1,
                        y1=1,
                        xref="paper",
                        x0=0,
                        x1=1,
                        line=dict(
                            color="red",
                            width=1,
                            dash="dash",
                        )
                    )
                ],
                annotations=[
                    dict(
                        x=0.02,
                        y=1.1,
                        xref="paper",
                        yref="y",
                        text="Baseline (Lift=1)",
                        showarrow=False,
                        font=dict(size=10, color="red"),
                    )
                ]
            )
            
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        with col2:
            # Heatmap
            # Create co-occurrence matrix
            product_pairs = []
            for _, row in display_rules.iterrows():
                antecedents = row['antecedents'].split(', ')
                consequents = row['consequents'].split(', ')
                for ant in antecedents:
                    for cons in consequents:
                        product_pairs.append((ant, cons, row['lift']))

            pairs_df = pd.DataFrame(product_pairs, columns=['product1', 'product2', 'lift'])
            
            heatmap_data = pairs_df.pivot_table(
                index='product1',
                columns='product2',
                values='lift',
                aggfunc='mean'
            ).fillna(0)
            
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdBu',
                zmid=1,
                text=np.round(heatmap_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
            ))

            heatmap_fig.update_layout(
                title='Product Association Strength (Lift)',
                title_x=0.5,
                height=500,
                xaxis={
                    'tickangle': 45,
                    'title': 'Consequent Product'
                },
                yaxis={
                    'title': 'Antecedent Product'
                },
                margin=dict(l=20, r=20, t=40, b=100)
            )
            
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Add explanations in expandable sections
        col3, col4 = st.columns([1, 1])
        
        with col3:
            with st.expander("📊 Understanding the Scatter Plot"):
                st.markdown("""
                The scatter plot shows the relationship between Confidence and Lift:
                
                - **X-axis**: Confidence (likelihood of consequent following antecedent)
                - **Y-axis**: Lift (improvement over random chance)
                - **Bubble size**: Support (frequency in data)
                - Points above red line (Lift > 1) show positive associations
                """)
        
        with col4:
            with st.expander("🔥 Understanding the Heatmap"):
                st.markdown("""
                The heatmap shows product association strengths:
                
                - **Red**: Strong positive associations (Lift > 1)
                - **White**: Neutral associations (Lift ≈ 1)
                - **Blue**: Negative associations (Lift < 1)
                - Numbers show exact lift values
                """)
        
        # 2. Association Rules Table
        st.header("Association Rules")
        st.dataframe(display_rules, use_container_width=True)
        
        # 3. Detailed Rule Analysis
        st.header("Detailed Rule Analysis")
        search_term = st.text_input("🔍 Search rules (e.g., product name)", "")
        
        # Create rule text and filter based on search
        display_rules['rule_text'] = (
            display_rules['antecedents'] + ' → ' + display_rules['consequents'] +
            ' (Conf: ' + display_rules['confidence'].apply(lambda x: f'{x:.2f}') +
            ', Lift: ' + display_rules['lift'].apply(lambda x: f'{x:.2f}') + ')'
        )
        
        # Filter rules based on search
        if search_term:
            filtered_rules = display_rules[
                display_rules['rule_text'].str.lower().str.contains(search_term.lower())
            ]
        else:
            filtered_rules = display_rules.nlargest(50, 'lift')
        
        if not filtered_rules.empty:
            selected_rule = st.selectbox(
                "Select a rule to analyze:",
                options=filtered_rules['rule_text'].tolist(),
                help="Choose a rule to see detailed analysis"
            )
            
            selected_rule_data = display_rules[
                display_rules['rule_text'] == selected_rule
            ].iloc[0]
            
            # Create three columns for visualization and metrics
            col5, col6, col7 = st.columns([2, 1, 1])
            
            with col5:
                # Rule visualization
                fig = create_rule_visualization(selected_rule_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col6:
                # Confidence gauge
                conf_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=selected_rule_data['confidence'] * 100,
                    title={'text': "Confidence %"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "lightblue"},
                        'steps': [
                            {'range': [0, 50], 'color': 'lightgray'},
                            {'range': [50, 75], 'color': 'gray'},
                            {'range': [75, 100], 'color': 'darkgray'}
                        ]
                    }
                ))
                conf_fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(conf_fig, use_container_width=True)
            
            with col7:
                # Lift gauge
                lift_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=selected_rule_data['lift'],
                    title={'text': "Lift"},
                    gauge={
                        'axis': {'range': [0, max(5, selected_rule_data['lift'])]},
                        'bar': {'color': "lightgreen"},
                        'steps': [
                            {'range': [0, 1], 'color': 'lightgray'},
                            {'range': [1, 2], 'color': 'gray'},
                            {'range': [2, max(5, selected_rule_data['lift'])], 'color': 'darkgray'}
                        ]
                    }
                ))
                lift_fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(lift_fig, use_container_width=True)
        else:
            st.warning("No rules match your search. Try different terms.")

    
    # Tab 3: Recommendations
    with tab3:
        if 'selected_rule' in locals():
            st.markdown(interpret_rule(selected_rule_data, len(df)))

        else:
            st.info("Select a rule in the Analysis tab to see recommendations")
    
    # Export options (below tabs)
    st.header("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = display_rules.to_csv(index=False)
        st.download_button(
            label="Download Association Rules (CSV)",
            data=csv,
            file_name="association_rules.csv",
            mime="text/csv"
        )
    
    with col2:
        csv = frequent_itemsets.to_csv(index=False)
        st.download_button(
            label="Download Frequent Itemsets (CSV)",
            data=csv,
            file_name="frequent_itemsets.csv",
            mime="text/csv"
        )   
    
if __name__ == "__main__":
    main()
