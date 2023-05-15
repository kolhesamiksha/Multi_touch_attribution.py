import streamlit as st 
from streamlit_option_menu import option_menu
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from minio import Minio
from PIL import Image, ImageOps
from io import BytesIO
import plotly.express as px
import utils as ut
import os
import requests
import plotly.graph_objects as go
import streamlit.components.v1 as components
os.system("pip install pandas pmdarima statsmodels minio")
import statsmodels.api as sm

response = requests.get(url='https://katonic.ai/favicon.ico')
im = Image.open(BytesIO(response.content))

st.set_page_config(
    page_title='Multi-Touch-Attribution-markov-chain', 
    page_icon = im,
    initial_sidebar_state = 'auto'
)

# ACCESS_KEY = "DNPD2SAYLBELJ423HCNU"
# SECRET_KEY = "zF7F6W93HS8vt+JKen4U17+zhcHiwH47AMuO3ap0"
# PUBLIC_BUCKET = "shared-storage"

# # client = Minio(
# #         endpoint="minio-server.default.svc.cluster.local:9000",
# #         access_key=ACCESS_KEY,
# #         secret_key=SECRET_KEY,
# #         secure=False,
# #     )

# # usr_journey_path = "/multi_touch_attribution/usr_journey_attribution_data.csv"
# # usr_attribution_path = "/multi_touch_attribution/usr_attribution_data.csv"
# # transition_prob_matrix = "/multi_touch_attribution/transition_matrix.csv"
# # channel_performance_across_methods = "/multi_touch_attribution/channel_performance_across_methods.csv"
# # markov_chain_rm_data = "/multi_touch_attribution/markov_chain_attribution_update.csv"
# # conversion_data = '/multi_touch_attribution/conversion_data.csv'
# # conversions_by_date = '/multi_touch_attribution/conversions_by_date.csv'
# # attribution_by_model_type_pd = '/multi_touch_attribution/attribution_by_model_type_pd.csv'
# # cost_per_aquisition = '/multi_touch_attribution/cost_per_aquisition.csv'
# # budget_allocation = '/multi_touch_attribution/budget_allocation.csv'

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         usr_journey_path,
# #         "data/usr_journey_attribution_data.csv",
# #     )

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         usr_attribution_path,
# #         "data/usr_attribution_data.csv",
# #     )

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         transition_prob_matrix,
# #         "data/transition_matrix.csv",
# #     )

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         channel_performance_across_methods,
# #         "data/channel_performance_across_methods.csv",
# #     )

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         markov_chain_rm_data,
# #         "data/markov_chain_rm_data.csv",
# #     )

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         conversions_by_date,
# #         "data/conversion_data.csv",

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         conversions_by_date,
# #         "data/conversions_by_date.csv",

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         attribution_by_model_type_pd,
# #         "data/attribution_by_model_type_pd.csv",

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         cost_per_aquisition,
# #         "data/cost_per_aquisition.csv",

# # client.fget_object(
# #         PUBLIC_BUCKET,
# #         budget_allocation,
# #         "data/budget_allocation.csv",

def app():
    menu_data = [
        {'icon': "🚀", 'label': "Load_data-katonic-connectors"},
        {'icon': "🔃", 'label': "data-preparation", 'submenu': [
            {'icon': "fa fa-paperclip", 'label': "User_Journey"},
            {'icon': "fa fa-paperclip", 'label': "Heuristic_attribution"}
        ]},
        {'icon': "💻", 'label': "markov-chain-model", 'submenu': [
            {'icon': "fa fa-paperclip", 'label': "Markov-chain-attribution"},
            {'icon': "fa fa-paperclip", 'label': "heuristic v/s data-driven"}
        ]},
        {'icon': "far fa-chart-bar", 'label': "Spend_optimisation", 'submenu': [
            {'icon': "fa fa-paperclip", 'label': "about"},
            {'icon': "fa fa-paperclip", 'label': "View-campaign-performance"},
            {'icon': "fa fa-paperclip", 'label': "Budget-allocation"}
        ]},
    ]

    with st.sidebar:
        selected_menu = option_menu(
            menu_title = "Main Menu",
            options = ["About", "Load_data-katonic-connectors", "data-preparation", "markov-chain-model", "Spend_optimisation"]
        )

    if selected_menu == "About":
        st.title("Multi-Touch-Attribution")

        st.write("""
        Behind the growth of every consumer-facing product is the **acquisition and retention of an engaged user base**. 
        When it comes to acquisition, the goal is to attract high quality users as cost effectively as possible. With marketing dollars dispersed across a wide array of campaigns, channels, and creatives, however, measuring effectiveness is a challenge. 
        In other words, it's difficult to know how to assign credit where credit is due. """)

        st.write("""Multi-touch attribution is a method of marketing measurement that accounts for all the touchpoints on the customer journey and designates a certain amount of credit to each channel so that marketers can see the value that each touchpoint has on driving a conversion.""")

        st.write("""
        #### Heuristic based approach
        """)

        st.write("""
        heuristic methods are rule-based and consist of both single-touch and multi-touch approaches:
        1. **Single-touch**: Single-touch methods, such as first-touch and last-touch, assign credit to the first channel, or the last channel, associated with a conversion.

        2. **Multi-touch**: Multi-touch methods, such as linear and time-decay, assign credit to multiple channels associated with a conversion.
        """)

        st.write("""
        #### Data-driven approach
        data-driven methods determine assignment using probabilities and statistics. Examples of data-driven methods include **Markov Chains and SHAP**.  
        """)

    elif selected_menu == "Load_data-katonic-connectors":
        st.title("Load_data-katonic-connectors")
        st.write("##### Loaded the data residing inside **AWS S3** bucket to the katonic **file-manager** using katonic connectors")
        st.write("""
            just by providing Source as IAM access key, secrete key, bucket name and object name 

            Destination as a katonic_file manager access and secrete key, bucket_name:private/public
        """)

        st.write("""
            Loaded data from AWS S3 to Katonic-filemanager
        """)

        img1 = Image.open('image/katonic_connectors.png')
        st.image(img1)

    elif selected_menu == "data-preparation":
        submenu_items = [item['label'] for item in menu_data[1]['submenu']]
        selected_submenu = st.sidebar.radio("data-preparation", submenu_items)
        if selected_submenu == "User_Journey":
            st.title("Users Journey🚀")
            st.write("---")

            st.write("""
            #### A user journey is defined as a series of steps that represent how you want users to interact with your app. It involves the analysis of how users are interacting with the app to identify the weakest points in the path to conversion.
            #### Mapping the user journey from the time of download and their first session is important for your app’s growth 
            """)

            st.write("---")

            st.write("""

            ##### Points can be drawn up by understanding users journey map: 

            - Where the audience interacts with the product. This can be a company website, ads on third-party resources, social media, etc. 
            - How various audience segments interact with your app. 
            - What stages does the user go through before buying and what goals they have.
            
            """)
            img1 = Image.open('image/user_journey.jpg')
            st.image(img1,width=500)

            if st.button("User_journey_data"):
                usr_journ_data = pd.read_csv("data/user_journey_attribution_data.csv")
                st.dataframe(usr_journ_data)
            
        elif selected_submenu == "Heuristic_attribution":
            st.title("Heuristic approach for user attribution")
            st.write("""
                ###### Heuristic means a practical method of achieving a goal, that is not necessarily optimal, but is good enough for now. Heuristic methods are used when figuring out the optimal way to do something is too costly,
                ###### The heuristic attribution models are all based on where in the customer journey you are.. 
            """)
            st.write("---")
            st.write("""
            There are two main types of methods (or models) used to divide the revenue across channels.""")

            st.write("""
            1. **First Touch Attribution**: This model assigns 100% of the opportunity value to the first (oldest) interaction with the customer.
            First touch is helpful when you want to know which channels are most likely to bring new leads. 
            """)

            st.write("""
            2. **Last Touch Attribution**: This model assigns 100% of the opportunity value to the most recent interaction with the customer. Basically, the last campaign that the prospect interacted with gets all the credit.
            Use last touch to figure out which campaigns / channels are your best deal closers. 
            """)

            usr_attr_data = pd.read_csv("data/usr_attribution_data.csv")
            if st.button("Heuristic_attribution_analysis"):
                st.write("heuristic_inside")
                st.dataframe(usr_attr_data)

                # cnt_plot = sns.catplot(x='channel',y='attribution_percent',hue='attribution_model',data=usr_attr_data, kind='bar', aspect=2).set_xticklabels(rotation=15)
                # st.pyplot(cnt_plot)

                #df = px.data.tips()
                fig = px.histogram(usr_attr_data, x="channel", y="attribution_percent",
                                color='attribution_model', barmode='group',
                                height=700)

                st.plotly_chart(fig,use_container_width=True)
    
    elif selected_menu == "markov-chain-model":
        submenu_items = [item['label'] for item in menu_data[2]['submenu']]
        selected_submenu = st.sidebar.radio("markov-chain-model", submenu_items)
        if selected_submenu == "Markov-chain-attribution":
            st.title("Markov-chain-attribution modelling")
            st.write("""##### Markov chaining is the process/illustration of predicting the future event based on some previous conditions/event.""")
            st.write("---")
            st.write("""
            Three steps to calculate attribution using Markov chains:
            1. **Transition probability matrix**: A transition probability matrix is a matrix that contains the probabilities associated with moving from one state to another state. This is calculated using the data from all available customer journeys.

            2. **Total conversion probability**: on average, the likelihood that a given user will experience a conversion event.

            3. **Removal effect per channel**: difference between the total conversion probability (with all channels) and the conversion probability when the conversion is set to 0%.
            """)

            trans_data = pd.read_csv('data/transition_matrix.csv')
            markov_chain_data = pd.read_csv("data/markov_chain_rm_data.csv")
            if st.button("Transition_Matrix"):
                st.write("##### Transition Probability Matrix")
                transition_matrix_data = pd.read_csv('data/transition_matrix.csv')
                transition_matrix = transition_matrix_data.pivot(index='start_state', columns='end_state', values='transition_probability')

                node_labels = list(set(transition_matrix_data['start_state'].unique()) | set(transition_matrix_data['end_state'].unique()))

                # Create node and link data for the Sankey diagram
                nodes = [dict(label=node_label) for node_label in node_labels]
                links = []
                for start_state, row in transition_matrix.iterrows():
                    for end_state, probability in row.iteritems():
                        if pd.notnull(probability) and probability > 0:
                            links.append(dict(source=node_labels.index(start_state), target=node_labels.index(end_state), value=probability,
                                            hovertemplate='Transition Probability: %{value:.2f}<extra></extra>'))

                # Create Sankey diagram figure
                fig = go.Figure(data=[go.Sankey(node=dict(label=node_labels), link=dict(source=[link['source'] for link in links],
                                                                                    target=[link['target'] for link in links],
                                                                                    value=[link['value'] for link in links],
                                                                                    hovertemplate=[link['hovertemplate'] for link in links]))])

                # Update layout settings
                fig.update_layout(title='Markov Chain Visualization',
                                font=dict(size=10),
                                width=800,
                                height=600)

                # Display the plot in Streamlit app
                st.plotly_chart(fig)

                st.write("---")

                transition_matrix_pivot = trans_data.pivot(index='start_state', columns='end_state', values='transition_probability')
                fig = go.Figure(data=go.Heatmap(
                    z=transition_matrix_pivot.values,
                    x=transition_matrix_pivot.columns,
                    y=transition_matrix_pivot.index,
                    colorscale='Blues',
                    zmax=0.25,
                    zmin=0,
                    colorbar=dict(title='Transition Probability')
                ))

                fig.update_layout(
                    title='Transition Matrix Heatmap',
                    xaxis=dict(title='End State'),
                    yaxis=dict(title='Start State')
                )

                st.plotly_chart(fig)

                #st.dataframe(trans_data)
                st.write("---")
                st.write("##### Total Conversion Probability")
                st.success("0.03760307804217135")
                st.write("---")
                st.write("##### Removal Effect per channel probability")
                st.dataframe(markov_chain_data)
            
        elif selected_submenu == "heuristic v/s data-driven":
            st.title("Heuristic v/s Data-driven")
            st.write("""
            **the multi-touch models are more accurate**, since you’re not betting all your money on a single channel. 
            
            It’s only fair to say all the interactions your lead had with your company contributed to some extent to the conversion, 
            
            so to credit only the first one, the last one, or some other one in the middle may seem simplistic and reductionist.
            """)

            new_df = pd.read_csv('data/channel_performance_across_methods.csv')
            new_df = new_df.drop('Unnamed: 0',axis=1)
            if st.button("channel_performance"):
                #st.dataframe(new_df) 

                fig = px.histogram(new_df, x="channel", y="attribution_percent",
                                    color='attribution_model', barmode='group',
                                    height=700)

                st.plotly_chart(fig,use_container_width=True)

    elif selected_menu == "Spend_optimisation":
        submenu_items = [item['label'] for item in menu_data[3]['submenu']]
        selected_submenu = st.sidebar.radio("Spend_optimisation", submenu_items)
        if selected_submenu == "about":
            st.title("Spend Optimisation")
            st.write("The goal of multi-touch attribution is to understand where to allocate your **marketing spend**. When marketers can understand the role that certain touchpoints played in a conversion, they can more effectively devote funds to similar touchpoints in future media plans and divert funds from ineffective channels.")
        elif selected_submenu == "View-campaign-performance":
            base_conversion_rate = pd.read_csv('data/conversion_data.csv')
            conversion_by_date = pd.read_csv('data/conversions_by_date.csv')
            attribution_by_model_type = pd.read_csv('data/attribution_by_model_type_pd.csv')
            cpa_summary_pd = pd.read_csv('data/cost_per_aquisition.csv')
            st.table(base_conversion_rate['interaction_type'])
            
            st.title("Campaign Performance")
            cc1,cc2 = st.columns([1,1])
            with cc1:
    
                colors = ['gold', 'mediumturquoise']
                fig = px.pie(base_conversion_rate, values='count', names='interaction_type')
                fig.update_traces(hoverinfo='label+percent', textinfo ='label+percent', textfont_size=15,insidetextorientation='radial',
                                   marker=dict(colors=colors, line=dict(color='#000000', width=2)))
                
                fig.update_layout(title='Interaction Types')
                st.plotly_chart(fig)
                st.write("---")

                fig1 = px.line(conversion_by_date, x='date', y='count', title="Conversions by Date")

                fig1.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Number of Conversions',
                    font=dict(size=20),
                    autosize=True,
                    width=800,
                    height=450
                )

                st.plotly_chart(fig1)

                st.write("---")

                fig2 = go.Figure()

                for attribution_model in attribution_by_model_type['attribution_model'].unique():
                    filtered_data = attribution_by_model_type[attribution_by_model_type['attribution_model'] == attribution_model]
                    fig2.add_trace(go.Bar(x=filtered_data['channel'], y=filtered_data['conversions_attributed'], name=attribution_model))

                # Customize the plot
                fig2.update_layout(
                    barmode='group',
                    xaxis_title='Channels',
                    yaxis_title='Number of Conversions',
                    title='Channel Performance',
                    autosize=True,
                    width=800,
                    height=450
                )

                st.plotly_chart(fig2)

                st.write("---")

                fig3 = px.bar(cpa_summary_pd, x='channel', y='CPA_in_Dollars', color='attribution_model',
                    barmode='group', labels={'channel': 'Channels', 'CPA_in_Dollars': 'CPA in $'},
                    title='Channel Cost per Acquisition')

                fig3.update_layout(
                    autosize=True,
                    width=800,
                    height=450,
                    legend=dict(title='Attribution Model')
                )

                st.plotly_chart(fig3)

        elif selected_submenu == "Budget-allocation":
            st.write("---")
            st.title("Budget-allocation-optimisation")
            st.write("###### data-driven approach using Markov chain for efficient budget allocation across all channels.")
            budget_allocation = pd.read_csv('data/budget_allocation.csv')
            st.write("1. Current Spending: Initial campaign assumption to assign same credit across all channels.")
            st.write("2. Proposed Spending: The markov chain conversion probability values.")
            fig4 = px.bar(budget_allocation, x="channel", y="budget", color="spending",
                        barmode="group", height=500, width=900)
            fig4.update_layout(title="Spend Optimization per Channel", xaxis_title="Channels",
                                yaxis_title="Budget in $", font=dict(size=15))
            st.plotly_chart(fig4)
